# Human Activity Recognition with Hidden Markov Models

This repository classifies six human activities from smartphone sensor features and compares a class-conditional Hidden Markov Model (HMM) with a scikit-learn Logistic Regression baseline.

The project is an educational benchmark based on the UCI Human Activity Recognition Using Smartphones dataset. It is designed to show how feature preprocessing, temporal windows, probabilistic sequence scoring, and a conventional classifier fit together.

## Activities

- `LAYING`
- `SITTING`
- `STANDING`
- `WALKING`
- `WALKING_DOWNSTAIRS`
- `WALKING_UPSTAIRS`

## Dataset

Download the dataset from the Kaggle mirror:

https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones

Place the two files below in a local `Data` directory. The raw CSV files are intentionally excluded from Git.

```text
Data/train.csv
Data/test.csv
```

The saved outputs from the original notebook show 7,352 training rows, 2,947 test rows, six activity classes, and 562 input columns before preprocessing. The `Activity` column is the target and `subject` is treated as an identifier rather than a predictive feature in the reusable pipeline.

## How The Pipeline Works

1. Load and validate matching train and test CSV schemas.
2. Remove the activity target and subject identifier from the feature matrix.
3. Fit `StandardScaler` on the training data only.
4. Fit PCA on the training data only, then transform the test data with that same PCA object.
5. Fit one diagonal-covariance Gaussian HMM per activity. Training sequences are contiguous subject/activity runs, and their lengths are passed to `hmmlearn` so transitions are learned within sequences.
6. Classify test windows by the largest HMM log likelihood.
7. Fit Logistic Regression on the same transformed features as a transparent comparison baseline.
8. Write accuracy, macro F1, weighted F1, and class-level metrics to a JSON file.

## Local Setup

Use Python 3.10 to 3.12 (macOS or Linux shown; on Windows use `.venv\Scripts\activate`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the data. This fetches the official UCI archive (no login needed, about 60 MB) and writes `Data/train.csv` and `Data/test.csv` in the Kaggle column layout:

```bash
python scripts/download_data.py
```

Run the reusable pipeline:

```bash
python scripts/run_activity_recognition.py \
  --train Data/train.csv \
  --test Data/test.csv \
  --output artifacts/metrics.json
```

Run the smoke tests without the full dataset:

```bash
python -m pytest -q
```

## Results (reusable pipeline, default settings)

80 PCA components, 3 hidden states per HMM, 16-window test chunks. Run on the official 7,352 / 2,947 train/test split:

| Activity | HMM F1 | Logistic Regression F1 |
|---|---|---|
| LAYING | 0.831 | 0.990 |
| SITTING | 0.703 | 0.878 |
| STANDING | 0.789 | 0.902 |
| WALKING | 0.857 | 0.945 |
| WALKING_DOWNSTAIRS | 0.683 | 0.930 |
| WALKING_UPSTAIRS | 0.761 | 0.909 |
| **Accuracy** | **0.773** | **0.926** |

The per-window Logistic Regression baseline beats the class-conditional HMMs. The features are already engineered summaries of each 2.56 s window, so most of the information is inside a single window, and fixed 16-window test chunks can straddle two activities.

The original notebook, `Human Activity Recognition from Smart Phone Data.ipynb`, is retained as a historical reference. The script under `scripts/` and the package under `src/` are the reproducible path for new runs.

## Historical Notebook Results

The original saved notebook reported approximately 0.873 HMM micro F1 at one hidden state and approximately 0.93 test F1 for its neural-network comparison. These values are preserved for context only. They are not fresh results from the reusable pipeline, and the old notebook contains a confusing early PCA cell that fits the test data separately.

## Project Limitations

- The benchmark contains engineered sensor windows rather than raw accelerometer streams.
- An HMM is meaningful only when rows are in temporal order and sequence boundaries are valid.
- The reusable implementation assumes contiguous rows for a subject represent temporal order.
- The dataset is a benchmark and should not be treated as validated performance for a new device, subject population, or safety-critical product.
- The current pipeline predicts activity classes; it is not a deployed real-time mobile application.
