# Experiment 1: ResNet-50 vs. ViT-B/16 for Skin-Lesion Classification

This repository contains Experiment 1 only: a comparison of a convolutional neural network (CNN) and a Vision Transformer (ViT) for skin-lesion image classification. It measures predictive performance and compute cost on HAM10000, with exploratory external evaluation on a mapped subset of PAD-UFES-20.

## Completed configurations

Four seed-42 configurations were trained and evaluated:

| Model | Pretrained weights | Strategy | Trainable parameters |
|---|---|---|---:|
| ResNet-50 | ImageNet-1K V2 | Frozen backbone | 14,343 |
| ResNet-50 | ImageNet-1K V2 | Fine-tuned | 23,522,375 |
| ViT-Base/16 | ImageNet-1K V1 | Frozen backbone | 5,383 |
| ViT-Base/16 | ImageNet-1K V1 | Fine-tuned | 85,804,039 |

### HAM10000 held-out test

| Model | Strategy | Accuracy | Macro-F1 | Balanced accuracy | Melanoma sensitivity |
|---|---|---:|---:|---:|---:|
| ResNet-50 | Frozen | 0.748 | 0.549 | 0.537 | 49.1% |
| ViT-B/16 | Frozen | 0.794 | 0.605 | 0.604 | 50.3% |
| ResNet-50 | Fine-tuned | 0.806 | 0.648 | 0.662 | 56.6% |
| ViT-B/16 | Fine-tuned | 0.838 | 0.705 | 0.665 | 66.7% |

Fine-tuned ViT-B/16 had the highest macro-F1 and melanoma sensitivity. Fine-tuned ResNet-50 had similar balanced accuracy, faster inference, and a smaller checkpoint.

### PAD-UFES-20 external subset

| Model | Strategy | Accuracy | Macro-F1 | Balanced accuracy | Melanoma sensitivity |
|---|---|---:|---:|---:|---:|
| ResNet-50 | Frozen | 0.422 | 0.407 | 0.441 | 21.2% |
| ViT-B/16 | Frozen | 0.502 | 0.444 | 0.381 | 3.8% |
| ResNet-50 | Fine-tuned | 0.340 | 0.327 | 0.352 | 9.6% |
| ViT-B/16 | Fine-tuned | 0.480 | 0.433 | 0.470 | 15.4% |

The external mapping retains 1,141 images in three classes (bcc: 845, mel: 52, nv: 244); 1,157 other diagnoses are excluded. These are exploratory transfer results, not evidence of clinical readiness.

The full report is [experiment1_report.tex](experiment1_report.tex) and [report.pdf](report.pdf).

## Data

The HAM10000 split is grouped by lesion identifier. It contains 7,153 train, 1,431 validation, and 1,431 test images, with no lesion-group overlap. The split files are included under data/splits.

Raw images are not included in Git. Place the datasets in these configured locations:

~~~text
data/raw/HAM10000/HAM10000_metadata.csv
data/raw/HAM10000/HAM10000_images_part_1/
data/raw/HAM10000/HAM10000_images_part_2/
data/raw/PAD_UFES_20/PAD-UFES-20/metadata.csv
data/raw/PAD_UFES_20/PAD-UFES-20/images/
~~~

PAD-UFES-20 label mapping is in configs/data.yaml: BCC to bcc, MEL to mel, and NEV to nv. An exact SHA-256 overlap check found no byte-identical images; near-duplicate overlap was not checked.

## Environment

Use Python 3.10 or newer. Install a PyTorch/torchvision pair matching your operating system and CUDA setup using the [PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining requirements:

~~~powershell
python -m venv .venv
& .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

The recorded environment was Python 3.12, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8, and an NVIDIA RTX 4050 Laptop GPU.

## Run the experiment

Run from the repository root. Validate the data, check image decoding and exact overlap, and regenerate the fixed split if needed:

~~~powershell
python -m src.data.validate --dataset all --verify-image-content
python -m src.data.validate --dataset all --check-external-overlap
python -m src.data.split
~~~

The overlap command hashes all images in both datasets. It detects byte-identical files, not visually similar re-encodings.

Train the four configurations:

~~~powershell
python -m src.training.train --model resnet50 --strategy frozen --seed 42
python -m src.training.train --model vit_b_16 --strategy frozen --seed 42
python -m src.training.train --model resnet50 --strategy finetune --seed 42
python -m src.training.train --model vit_b_16 --strategy finetune --seed 42
~~~

Each run selects a checkpoint using validation macro-F1 and prints its path. Training reads only the train and validation partitions. After configurations are locked, benchmark the printed checkpoint on the held-out test and external splits:

~~~powershell
python -m src.training.benchmark --checkpoint results/runs/YOUR_RUN_ID/checkpoints/best.pt --split test
python -m src.training.benchmark --checkpoint results/runs/YOUR_RUN_ID/checkpoints/best.pt --split external
~~~

Aggregate results and generate paired comparisons:

~~~powershell
python -m src.evaluation.compare_runs --split test --bootstrap-repetitions 2000
python -m src.evaluation.compare_runs --split external --bootstrap-repetitions 2000
~~~

Do not use test or external results for checkpoint selection or hyperparameter tuning.

## Training settings

Shared settings are in configs/experiments.yaml: 224 x 224 inputs, batch size 4, FP16 automatic mixed precision, AdamW, training-only class weights, horizontal and vertical flips, rotations up to 15 degrees, at most 20 epochs, and early stopping on validation macro-F1 with patience 4. The frozen head learning rate is 0.001; fine-tuning uses 0.00001 for the backbone and 0.0001 for the head.

## Repository contents

- src/ contains data validation and splitting, ResNet/ViT model adapters, training, benchmarking, metrics, and comparison code.
- configs/ contains the dataset and experiment settings.
- data/splits/ contains the fixed partitions used by the reported runs.
- results/tables/, results/evaluations/, and results/plots/ contain the metrics and evaluation artifacts.
- results/runs/ contains run summaries and histories; binary checkpoints are excluded from Git.
- experiment1_report.tex and report.pdf are the report source and compiled report.
- next_steps.md lists remaining Experiment 1 follow-up work.

Raw images and model checkpoints are excluded from Git because of their size. Run the training commands to create local checkpoints.

## Limitations

Each configuration was trained once with seed 42. HAM10000 is lesion-grouped but does not establish patient-level independence. The external subset has 52 melanoma images and differs in acquisition domain from HAM10000. Its bootstrap resamples images, not patient groups. The exact-hash audit does not detect near duplicates. These results do not establish clinical safety or deployment readiness.

## Dataset references

- Tschandl, P., Rosendahl, C., and Kittler, H. (2018). [The HAM10000 dataset](https://doi.org/10.1038/sdata.2018.161). Scientific Data.
- Pacheco, A. G. C. et al. (2020). [PAD-UFES-20](https://doi.org/10.1016/j.dib.2020.106221). Data in Brief, 32, 106221.
