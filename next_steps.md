# Next steps: Experiment 1

## Completed

- Validated all 10,015 readable HAM10000 images and created a fixed lesion-grouped split: 7,153 train, 1,431 validation, and 1,431 test images.
- Validated PAD-UFES-20 and retained 1,141 readable examples for the mapped classes bcc (845), mel (52), and nv (244). The other 1,157 diagnoses were excluded.
- Checked exact SHA-256 overlap across datasets: zero byte-identical images. Near-duplicate images have not been checked.
- Trained four seed-42 configurations: ResNet-50 and ViT-B/16, each frozen and fine-tuned.
- Evaluated each checkpoint on the locked HAM10000 test split and the mapped PAD-UFES-20 external subset.
- Generated aggregate metrics and paired, class-stratified bootstrap tables for both splits.
- Wrote the report in experiment1_report.tex, including class-wise recalls, paired intervals, resource measurements, and confusion matrices.

## Results to carry forward

- On HAM10000, fine-tuned ViT-B/16 had the highest test macro-F1 (0.705) and melanoma sensitivity (106/159, 66.7%). Its balanced accuracy (0.665) was close to fine-tuned ResNet-50 (0.662).
- ResNet-50 inference was about twice as fast and its checkpoint was smaller (94 MB versus 343 MB).
- On the external subset, macro-F1 ranged from 0.327 to 0.444 and melanoma sensitivity from 3.8% to 21.2%. Treat these as exploratory transfer results; the mapped set has only 52 melanoma images.
- The external image-level paired bootstrap is not patient-clustered. Neither external nor internal intervals capture training-seed variation.

## Recommended remaining work

1. **Estimate run-to-run variation.** If compute permits, repeat the same four configurations with seed 123 (and optionally another seed). Keep the split files and training settings fixed. Use validation macro-F1 for checkpoint selection; do not use the existing test or external results to tune the experiment.

    python -m src.training.train --model resnet50 --strategy frozen --seed 123
    python -m src.training.train --model vit_b_16 --strategy frozen --seed 123
    python -m src.training.train --model resnet50 --strategy finetune --seed 123
    python -m src.training.train --model vit_b_16 --strategy finetune --seed 123

   After training, benchmark the new checkpoints on the fixed test and external splits, then regenerate the aggregate tables. Report seed-level means and standard deviations separately from case-level bootstrap intervals.

2. **Account for grouped images in uncertainty estimates.** Add a patient-clustered bootstrap for PAD-UFES-20 and a lesion-clustered bootstrap for the HAM10000 test set. Resample groups while keeping all images in a selected group together.

3. **Check near-duplicate overlap.** The existing SHA-256 audit only detects byte-identical files. Add a documented perceptual-image comparison before making stronger external-generalization claims.

4. **Review the report and produce a PDF if needed.** Compile experiment1_report.tex from the workspace root in a LaTeX environment that has graphicx and review the included confusion matrices and references.

Do not describe this image-classification experiment as a clinical diagnostic system or as evidence of clinical readiness.
