# Cross-cohort WXS mutation-burden prediction

This archive contains the authoritative confirmatory manuscript, code, derived results, figures, and split case IDs for the two-way TCGA/CPTAC batch-transductive WMB study.

The endpoint is the log1p count of deduplicated non-synonymous variants from open masked WXS calls. It is not callable-territory-normalized clinical TMB. The assay-aware model uses source labels and an unlabeled target adaptation batch; target evaluation cases are disjoint and target labels are used only for final scoring. The CORAL-PCA-HGB control receives the same unlabeled adaptation batch.

Public data sources: TCGA (https://portal.gdc.cancer.gov/projects/TCGA), CPTAC-3 (https://portal.gdc.cancer.gov/projects/CPTAC-3), and GDC documentation (https://docs.gdc.cancer.gov/).

The archive includes a frozen Python environment description, runtime metadata, citation audit, split IDs, checksums, and per-seed prediction tables. Raw controlled-access files are not redistributed.

Permanent public repository: https://github.com/jinluo12345/tcga-cptac-wmb-transport. No DOI is assigned; repository history and release checksums identify this version.
