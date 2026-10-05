"""Corrected ablations for the TCGA/CPTAC TMB cross-cohort regressor.

The assay-aware and MLP paths copy the main experiment's encoder, loss,
annealed gradient-reversal coefficient, early stopping, normalization and
seed conventions.  Ablations differ only in the supervised domain head,
heteroscedastic head, or both.  External labels are used only for scoring.
"""
from pathlib import Path
import argparse, copy, json, math, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed_tmb"
OUT = ROOT / "results" / "tmb_ablation_corrected"
(OUT / "checkpoints").mkdir(parents=True, exist_ok=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calibration(y, mean, scale):
    scale = np.maximum(np.asarray(scale, float), 1e-5)
    return {
        "coverage_50": float(np.mean(np.abs(y - mean) <= 0.67449 * scale)),
        "coverage_90": float(np.mean(np.abs(y - mean) <= 1.64485 * scale)),
        "coverage_95": float(np.mean(np.abs(y - mean) <= 1.95996 * scale)),
        "width_90": float(np.mean(2 * 1.64485 * scale)),
    }


def metrics(y, mean, scale):
    out = {
        "n": int(len(y)),
        "rmse": float(mean_squared_error(y, mean) ** 0.5),
        "mae": float(mean_absolute_error(y, mean)),
        "spearman": float(spearmanr(y, mean).statistic),
        "r2": float(r2_score(y, mean)),
    }
    out.update(calibration(y, mean, scale))
    return out


class GR(Function):
    @staticmethod
    def forward(ctx, x, c):
        ctx.c = float(c)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.c * g, None


def gr(x, c):
    return GR.apply(x, c)


class MLPRegressor(nn.Module):
    """Exact main-experiment MLP baseline (homoscedastic, no domain head)."""
    def __init__(self, p):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(p, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(.15),
            nn.Linear(512, 128), nn.GELU())
        self.mean = nn.Linear(128, 1)

    def forward(self, x, domain_coeff=0.0):
        return self.mean(self.encoder(x)).squeeze(1), None, None


class AssayAwareHetero(nn.Module):
    """Exact main-experiment heteroscedastic assay-aware model."""
    def __init__(self, p):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(p, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(.15),
            nn.Linear(512, 128), nn.GELU())
        self.mean = nn.Linear(128, 1)
        self.logvar = nn.Linear(128, 1)
        self.domain = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 1))

    def forward(self, x, domain_coeff=0.0):
        h = self.encoder(x)
        return (self.mean(h).squeeze(1), self.logvar(h).squeeze(1),
                self.domain(gr(h, domain_coeff)).squeeze(1))


class HeteroNoDomain(nn.Module):
    """Exact main encoder and heteroscedastic head with no domain branch."""
    def __init__(self, p):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(p, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(.15),
            nn.Linear(512, 128), nn.GELU())
        self.mean = nn.Linear(128, 1)
        self.logvar = nn.Linear(128, 1)

    def forward(self, x, domain_coeff=0.0):
        h = self.encoder(x)
        return self.mean(h).squeeze(1), self.logvar(h).squeeze(1), None


class AssayAwareHomo(nn.Module):
    """Main encoder/domain head with one learned homoscedastic variance."""
    def __init__(self, p):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(p, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(.15),
            nn.Linear(512, 128), nn.GELU())
        self.mean = nn.Linear(128, 1)
        self.logvar = nn.Parameter(torch.tensor(-0.3))
        self.domain = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 1))

    def forward(self, x, domain_coeff=0.0):
        h = self.encoder(x)
        lv = self.logvar.expand(x.shape[0])
        return self.mean(h).squeeze(1), lv, self.domain(gr(h, domain_coeff)).squeeze(1)


def predict(model, x, device):
    model.eval()
    means, scales = [], []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            m, lv, _ = model(torch.from_numpy(x[i:i+512]).to(device), 0.0)
            means.append(m.cpu().numpy())
            if lv is None:
                scales.append(np.ones(len(m), np.float32))
            else:
                scales.append(torch.exp(0.5 * torch.clamp(lv, -6, 4)).cpu().numpy())
    return np.concatenate(means), np.concatenate(scales)


def fit(variant, xtr, ytr, xval, yval, xtarget, epochs, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if variant == 'mlp_no_domain':
        model = MLPRegressor(xtr.shape[1])
    elif variant == 'hetero_no_domain':
        model = HeteroNoDomain(xtr.shape[1])
    elif variant == 'homo_domain':
        model = AssayAwareHomo(xtr.shape[1])
    else:
        model = AssayAwareHetero(xtr.shape[1])
    model.to(device)
    source_loader = DataLoader(
        TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr.astype(np.float32))),
        batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(torch.from_numpy(xtarget), batch_size=batch_size, shuffle=True)
    target_iter = iter(target_loader)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    bce = nn.BCEWithLogitsLoss()
    use_domain = variant in ('hetero_domain', 'homo_domain')
    # lambda0 keeps the exact domain branch but sets its supervised coefficient to zero.
    domain_weight = 0.0 if variant in ('hetero_no_domain', 'hetero_lambda0') else 1.0
    best_state, best_score, patience = None, float('inf'), 0
    for epoch in range(epochs):
        model.train()
        coeff = .35 * (2 / (1 + math.exp(-8 * epoch / max(1, epochs - 1))) - 1)
        for xb, yb in source_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            mean, logvar, _ = model(xb, 0.0)
            if variant == 'mlp_no_domain':
                loss = .5 * (mean - yb).pow(2).mean()
            else:
                lv = torch.clamp(logvar, -6, 4)
                loss = .5 * (torch.exp(-lv) * (mean - yb).pow(2) + lv).mean()
            if use_domain or variant == 'hetero_lambda0':
                try:
                    xu = next(target_iter).to(device)
                except StopIteration:
                    target_iter = iter(target_loader)
                    xu = next(target_iter).to(device)
                z = torch.cat([xb, xu])
                dom_y = torch.cat([torch.zeros(len(xb), device=device),
                                   torch.ones(len(xu), device=device)])
                dom_coeff = coeff if use_domain else 0.0
                _, _, dlogit = model(z, dom_coeff)
                # lambda0 retains the exact domain branch but sets its supervised
                # contribution to zero; full assay-aware uses BCE with annealed GR.
                loss = loss + (domain_weight if variant == 'hetero_lambda0' else 1.0) * bce(dlogit, dom_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        vm, _ = predict(model, xval, device)
        score = mean_squared_error(yval, vm)
        if score < best_score - 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                break
    if best_state is None:
        raise RuntimeError('no validation checkpoint')
    model.load_state_dict(best_state)
    return model, float(best_score), epoch + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--seeds', default='11,23,47')
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(',') if x]
    print({'torch': torch.__version__, 'cuda': torch.cuda.is_available(),
           'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}, flush=True)
    X = np.load(DATA / 'expression_log1p_tpm.npy', mmap_mode='r')
    meta = pd.read_csv(DATA / 'sample_metadata.csv')
    y = meta.log1p_tmb.to_numpy(np.float32)
    rows, predictions = [], []
    variants = ['mlp_no_domain', 'hetero_no_domain', 'hetero_lambda0', 'hetero_domain', 'homo_domain']
    for source, target in (('TCGA', 'CPTAC'), ('CPTAC', 'TCGA')):
        source_idx = np.flatnonzero(meta.cohort.to_numpy() == source)
        target_idx = np.flatnonzero(meta.cohort.to_numpy() == target)
        for seed in seeds:
            # Exactly the main experiment's split/normalization convention.
            seed_all(seed)
            tr, va = train_test_split(source_idx, test_size=.2, random_state=seed)
            xtr = np.asarray(X[tr], np.float32)
            xval = np.asarray(X[va], np.float32)
            xt = np.asarray(X[target_idx], np.float32)
            mu, sd = np.nanmean(xtr, 0), np.nanstd(xtr, 0)
            sd[~np.isfinite(sd) | (sd < 1e-5)] = 1
            xtr, xval, xt = tuple((np.nan_to_num(z, nan=0) - mu) / sd for z in (xtr, xval, xt))
            for variant in variants:
                # Match the main seed offsets: MLP uses seed, assay-aware paths use seed+10000.
                seed_all(seed if variant == 'mlp_no_domain' else seed + 10000)
                model, val_loss, used = fit(variant, xtr, y[tr], xval, y[va], xt,
                                            args.epochs, args.batch_size)
                device = next(model.parameters()).device
                val_mean, val_scale = predict(model, xval, device)
                ext_mean, ext_scale = predict(model, xt, device)
                factor = float(np.clip(np.std(y[va] - val_mean) / (np.mean(val_scale) + 1e-6), .25, 5.0))
                val_scale *= factor
                ext_scale *= factor
                ck = OUT / 'checkpoints' / f'{source}_to_{target}_{variant}_seed{seed}.pt'
                torch.save({'state_dict': model.state_dict(), 'source': source, 'target': target,
                            'seed': seed, 'variant': variant, 'val_rmse': val_loss ** .5,
                            'scale_factor': factor, 'epochs': used}, ck)
                # Fixed, seed-specific corruption masks; target labels are never used in fitting.
                rng = np.random.RandomState(seed + 777)
                mask = rng.rand(*xt.shape) < .10
                xt_mask = xt.copy(); xt_mask[mask] = 0
                xt_noise = xt + rng.normal(0, .10, xt.shape).astype(np.float32)
                evaluations = [('clean', xt), ('mask10', xt_mask), ('noise01', xt_noise)]
                for evaluation, xe in evaluations:
                    pm, ps = predict(model, xe, device)
                    ps *= factor
                    z = metrics(y[target_idx], pm, ps)
                    z.update({'direction': f'{source}_to_{target}', 'seed': seed,
                              'variant': variant, 'evaluation': evaluation,
                              'scale_factor': factor, 'epochs': used})
                    rows.append(z)
                    for ii, truth, pred, scale in zip(target_idx, y[target_idx], pm, ps):
                        predictions.append({'direction': f'{source}_to_{target}', 'seed': seed,
                                            'variant': variant, 'evaluation': evaluation,
                                            'case_id': meta.iloc[ii].case_id,
                                            'cohort': target, 'lineage': meta.iloc[ii].label,
                                            'true': float(truth), 'pred': float(pred),
                                            'scale': float(scale), 'lo90': float(pred - 1.64485 * scale),
                                            'hi90': float(pred + 1.64485 * scale)})
                    print(json.dumps(z, sort_keys=True), flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'external_predictions.csv', index=False)
    (OUT / 'run_manifest.json').write_text(json.dumps({
        'seeds': seeds, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'directions': [['TCGA', 'CPTAC'], ['CPTAC', 'TCGA']],
        'variants': variants, 'corruptions': ['mask10', 'noise01'],
        'normalization': 'source-train mean/std applied to source validation and target',
        'selection': 'source validation RMSE only',
    }, indent=2))


if __name__ == '__main__':
    main()
