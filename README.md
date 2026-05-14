# CENG467 Term Project - Group 9 Technical Checkpoint

Repository: https://github.com/alialp5959/ceng467-term-project

This package contains the technical checkpoint report and the files used for the initial baseline evaluation.

## Main submission file

- `CENG467_ProgressReport_Group9.pdf`

## Checkpoint artifacts

- `src/checkpoint_baselines.py`: script for the checkpoint baselines.
- `results/checkpoint_results.csv`: BLEU and chrF results used in the report.
- `results/checkpoint_results_table.tex`: LaTeX table used in the report.
- `results/sample_translations_preview.csv`: short preview of qualitative examples.
- `report/CENG467_ProgressReport_Group9_Revised.tex`: LaTeX source of the report.

## Reproduce checkpoint baseline evaluation

The script expects FLORES-200 files under:

```text
flores200_dataset/devtest/eng_Latn.devtest
flores200_dataset/devtest/tur_Latn.devtest
```

Then run:

```bash
pip install -r requirements.txt
python src/checkpoint_baselines.py
```
