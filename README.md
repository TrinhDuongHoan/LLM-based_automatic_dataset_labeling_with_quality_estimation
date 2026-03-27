# LLM-based Automatic Dataset Labeling With Quality Estimation

This project provides a reproducible research pipeline for studying automatic dataset labeling with pseudo-label quality estimation.

## Research Goal

We simulate an LLM-assisted annotation workflow:

1. Start with a small gold seed set.
2. Automatically label a large unlabeled pool with multiple labelers.
3. Estimate pseudo-label quality using confidence-style features.
4. Filter low-quality pseudo-labels.
5. Train downstream classifiers on `gold + pseudo-labeled` data.
6. Compare which labeler and which downstream model perform best.

## Dataset

Default dataset: `data/newsCorpora.csv`

Schema:

- `title`: news headline
- `publisher`: news source
- `label`: original class id
- `label_name`: mapped class name
- `text`: combined text field used for modeling

Class mapping:

- `b -> business`
- `e -> entertainment`
- `m -> health`
- `t -> science_tech`

## Labelers Compared

The pipeline currently compares several automatic labelers:

- `keyword_rule`: weak supervision with rule-based keywords
- `fewshot_retrieval`: nearest-seed retrieval as a lightweight proxy for in-context LLM labeling
- `seed_logreg`: logistic regression trained only on seed labels
- `seed_nb`: Multinomial Naive Bayes trained only on seed labels
- `ensemble_vote`: combines multiple pseudo-labelers by averaged probabilities
- `hf_llm_*`: optional Hugging Face instruction-tuned LLM labeler for Kaggle

## Downstream Models Compared

These models are trained on the filtered pseudo-labeled dataset:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- Random Forest

## Quality Estimation

For each pseudo-labeled sample, the pipeline computes:

- `confidence`
- `margin`
- `entropy`
- `predicted_correct` for analysis against the available gold label

A quality estimator is then trained to predict whether a pseudo-label is likely correct. Its output is used to keep only pseudo-labels above a configurable threshold.

## Project Structure

```text
.
├── data/
│   └── newsCorpora.csv
├── models/
├── outputs/
├── scripts/
│   ├── prepare_data.py
│   ├── generate_pseudo_labels.py
│   ├── train_models.py
│   ├── evaluate.py
│   └── visualize.py
└── src/ldlqe/
    ├── config.py
    ├── data.py
    ├── labelers.py
    ├── quality.py
    ├── training.py
    ├── evaluation.py
    └── visualization.py
```

## End-to-End Pipeline

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You can also install the project as a package:

```bash
pip install -e .
```

### 2. Run the full experiment

```bash
chmod +x run_pipeline.sh
PYTHON_BIN=.venv/bin/python ./run_pipeline.sh
```

### 3. Or run step by step

```bash
export PYTHONPATH=$(pwd)/src
python3 scripts/prepare_data.py --source data/newsCorpora.csv
python3 scripts/generate_pseudo_labels.py
python3 scripts/train_models.py --quality-threshold 0.55
python3 scripts/evaluate.py
python3 scripts/visualize.py --sample-labeler ensemble_vote
```

### 4. Kaggle LLM run

For Kaggle GPU, enable the optional Hugging Face LLM labeler:

```bash
pip install -r requirements.txt
export PYTHONPATH=$(pwd)/src
python3 scripts/prepare_data.py --source /kaggle/input/<your-dataset-folder>/newsCorpora.csv
python3 scripts/generate_pseudo_labels.py \
  --include-llm \
  --llm-model-name google/flan-t5-base \
  --llm-batch-size 8 \
  --llm-num-samples 3 \
  --max-unlabeled-rows 12000
python3 scripts/train_models.py --quality-threshold 0.55 --max-pseudo-rows 12000
python3 scripts/evaluate.py
python3 scripts/visualize.py --sample-labeler hf_llm_flan_t5_base
```

Recommended Kaggle models:

- `google/flan-t5-base`: safest starting point
- `google/flan-t5-large`: better quality, heavier VRAM
- `google/flan-ul2`: stronger but usually too heavy for simple Kaggle runs

## Main Outputs

After running the pipeline, you will get:

- `data/processed/*.csv`: train, seed, unlabeled, validation, test splits
- `outputs/pseudo_labels/*.csv`: pseudo-labeled datasets for each labeler
- `outputs/pseudo_labels/labeler_summary.csv`: pseudo-label accuracy comparison
- `outputs/results/validation_results.csv`: downstream validation comparison
- `outputs/results/quality_estimation_results.csv`: quality estimator metrics
- `outputs/results/best_model_summary.json`: best experiment summary
- `outputs/figures/*.png`: figures for the report

## Suggested Experiments For The Report

You can structure the thesis/demo around these questions:

1. Which automatic labeler creates the best pseudo-labels?
2. Does quality filtering improve downstream performance?
3. Which downstream classifier is most robust to pseudo-label noise?
4. How strongly do `confidence`, `margin`, and `entropy` correlate with pseudo-label correctness?
5. Does an ensemble teacher outperform individual weak labelers?

## Extension Ideas

To make the project more strongly LLM-oriented, you can extend `labelers.py` with:

- OpenAI API or local LLM prompt-based labeler
- Chain-of-thought plus self-consistency voting
- Judge model for label verification
- Active learning loop for uncertain samples
- Calibration plots for pseudo-label quality scores

## Run On Kaggle

If your local machine is too weak, you can train directly in a Kaggle Notebook by cloning the repo and installing it as a module.

### Kaggle notebook cells

```python
!git clone <YOUR_GITHUB_REPO_URL>
%cd LLM-based_automatic_dataset_labeling_with_quality_estimation
!pip install -q -r requirements.txt
```

If the dataset file is already uploaded to Kaggle input storage:

```python
from pathlib import Path

DATASET_PATH = Path("/kaggle/input/<your-dataset-folder>/newsCorpora.csv")
```

Prepare data and train:

```python
!python scripts/prepare_data.py --source /kaggle/input/<your-dataset-folder>/newsCorpora.csv
!python scripts/generate_pseudo_labels.py --include-llm --llm-model-name google/flan-t5-base --max-unlabeled-rows 12000
!python scripts/train_models.py --max-pseudo-rows 20000 --quality-threshold 0.55
!python scripts/evaluate.py
!python scripts/visualize.py --sample-labeler hf_llm_flan_t5_base
```

Or import the module directly inside the notebook:

```python
from pathlib import Path
import pandas as pd

from ldlqe.data import SplitConfig, build_splits, save_splits
from ldlqe.labelers import build_labelers
from ldlqe.training import build_downstream_models, train_and_score

source_path = Path("/kaggle/input/<your-dataset-folder>/newsCorpora.csv")
splits = build_splits(SplitConfig(source_path=source_path, seed_per_class=80))
save_splits(splits, Path("data/processed"))

seed_df = splits["seed"]
unlabeled_df = splits["unlabeled_pool"]
val_df = splits["val"]

labeler = build_labelers()[-1].fit(seed_df)  # ensemble_vote
pseudo = labeler.predict(unlabeled_df).predictions
train_df = pd.concat(
    [
        seed_df.assign(train_label=seed_df["label_name"]),
        pd.concat([unlabeled_df.reset_index(drop=True), pseudo.reset_index(drop=True)], axis=1)
        .assign(train_label=lambda df: df["pseudo_label"]),
    ],
    ignore_index=True,
)

model_name, estimator = next(iter(build_downstream_models().items()))
model, result = train_and_score(train_df, val_df, labeler.name, model_name, estimator)
print(result)
```

### Recommended Kaggle workflow

1. Push this repo to GitHub.
2. Open a Kaggle Notebook with Internet enabled.
3. `git clone` the repo.
4. Add `src` to `sys.path` or install requirements.
5. Mount your dataset from Kaggle `Input`.
6. Run the scripts with `--include-llm` on GPU.

If you want, the next step can be adding a dedicated `kaggle_train.ipynb` template with ready-made cells for Kaggle GPU usage.
