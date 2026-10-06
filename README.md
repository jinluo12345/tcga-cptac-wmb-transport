# Cross-cohort WXS mutation-burden prediction

This archive contains the authoritative confirmatory manuscript, code, derived results, figures, and split case IDs for the two-way TCGA/CPTAC batch-transductive WMB study. This release is `v1.0.14-confirmatory`.

The endpoint is the log1p count of deduplicated non-synonymous variants from open masked WXS calls. It is not callable-territory-normalized clinical TMB. The assay-aware model uses source labels and an unlabeled target adaptation batch; target evaluation cases are disjoint and target labels are used only for final scoring. The CORAL-PCA-HGB control receives the same unlabeled adaptation batch.

Public data sources: TCGA (https://portal.gdc.cancer.gov/projects/TCGA), CPTAC-3 (https://portal.gdc.cancer.gov/projects/CPTAC-3), and GDC documentation (https://docs.gdc.cancer.gov/).

The archive includes a frozen Python environment description, runtime metadata, citation audit, split IDs, checksums, and per-seed prediction tables. Raw controlled-access files are not redistributed.

Permanent public repository: https://github.com/jinluo12345/tcga-cptac-wmb-transport. No DOI is assigned; repository history and release checksums identify this version.

Reproduction entry points are under `src/`. Run the CPU control and analysis from the repository root with the frozen environment in `metadata/environment.txt`; the GPU scripts require the pinned image described in `metadata/runtime_manifest.json`. To reconstruct labels after obtaining the open masked WXS MAF files from GDC, run `src/derive_wmb_labels.py --maf-root <directory containing file_id.maf.gz>`. The exact WXS file manifest and variant-class filter are in `metadata/wxs_maf_manifest.json` and `metadata/wmb_label_audit.md`. The release includes the derived expression panel and sample metadata used by the confirmatory models. Code is MIT licensed; GDC data remain subject to their source terms.

A portable CPU regeneration entry point is `src/reproduce_cpu.sh`; it runs the CORAL control, unique-case summaries, 2,000-resample paired bootstrap, and all five figure generators from the repository root. The GPU refit scripts use the same relative paths and require the pinned platform image.
