# TMB label audit

Expression identities are exact GDC case IDs from the frozen STAR-count manifest. Mutation files are open masked WXS MAFs joined by the same case IDs. For every case, all qualifying non-silent variants were collected across its WXS files and deduplicated by chromosome, start and end positions, reference allele and alternate allele. The target is log1p of the resulting unique variant count. No expression value was used in label construction.

The active label table is `data/tmb_labels.csv`; 4,077 patients are resolved (2,431 TCGA and 1,646 CPTAC). The cohort means of log1p(TMB) are 4.379 (TCGA) and 4.249 (CPTAC). Cases without an open WXS MAF are excluded before all patient-level splits. The mutation manifest and raw MAF files remain available for reproducibility.
