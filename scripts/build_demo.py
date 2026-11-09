"""Run the pipeline once and build the interactive demo page.

Reads Data/train.csv and Data/test.csv, fits the same pipeline as
run_activity_recognition.py, exports what the demo shows (inputs, PCA view,
HMM parameters, per-chunk likelihoods, predictions, metrics) and inlines it
into demo/template.html, writing demo/index.html. Open that file in a browser.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from activity_hmm.pipeline import (  # noqa: E402
    ActivityRecognitionPipeline,
    PipelineConfig,
    _subject_windows,
    load_har_data,
)

# A few interpretable features out of the 561, shown on the "input" section.
SHOWCASE_FEATURES = {
    "tBodyAccMag-mean()": "Movement intensity",
    "tBodyGyroMag-mean()": "Rotation speed",
    "tGravityAcc-mean()-X": "Gravity along phone X (posture)",
    "tBodyAcc-std()-X": "Shake along X",
}
PLACEHOLDER = "/*__DEMO_DATA__*/null"


def r(values, digits=3):
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("Data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("Data/test.csv"))
    parser.add_argument("--template", type=Path, default=REPOSITORY_ROOT / "demo/template.html")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "demo/index.html")
    args = parser.parse_args()

    train, test = load_har_data(args.train, args.test)
    config = PipelineConfig()
    pipe = ActivityRecognitionPipeline(config).fit(train)
    labels = pipe.labels
    index = {name: i for i, name in enumerate(labels)}

    z_test = pipe._transform(test)
    lr_proba = pipe.baseline.predict_proba(z_test)
    lr_classes = [str(c) for c in pipe.baseline.classes_]
    lr_proba = lr_proba[:, [lr_classes.index(name) for name in labels]]
    lr_pred = lr_proba.argmax(axis=1)

    # HMM: score each 16-window chunk with all six models, decode hidden states
    # with the winning model (Viterbi) so the page can show the state path.
    chunks, hmm_pred = [], np.zeros(len(test), dtype=int)
    hidden = np.zeros(len(test), dtype=int)
    for start, end in _subject_windows(test, config.subject_column, config.window_size):
        window = z_test[start:end]
        scores = [pipe.hmm_models[name].score(window) for name in labels]
        best = int(np.argmax(scores))
        hmm_pred[start:end] = best
        hidden[start:end] = pipe.hmm_models[labels[best]].predict(window)
        chunks.append({"s": start, "e": end, "ll": r(scores, 1)})

    truth = test[config.target_column].map(index).to_numpy()
    subjects = test[config.subject_column].to_numpy()

    def metrics(pred):
        cm = confusion_matrix(truth, pred, labels=range(len(labels)))
        tp = np.diag(cm)
        precision = tp / np.maximum(cm.sum(axis=0), 1)
        recall = tp / np.maximum(cm.sum(axis=1), 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        return {
            "accuracy": round(float(tp.sum() / cm.sum()), 4),
            "f1_macro": round(float(f1.mean()), 4),
            "f1": r(f1),
            "confusion": cm.tolist(),
        }

    # Example raw rows (one per activity, all 561 values) for the input view.
    feature_cols = pipe.feature_columns
    examples = {}
    for name in labels:
        row = test.index[test[config.target_column] == name][len(test) // 50]
        examples[name] = {"row": int(row), "subject": int(subjects[row]),
                          "values": r(test.loc[row, feature_cols].to_numpy(), 3)}

    data = {
        "labels": labels,
        "config": {
            "pca": int(pipe.pca.n_components_),
            "states": config.hmm_states,
            "window": config.window_size,
            "iterations": config.hmm_iterations,
        },
        "sizes": {
            "train_rows": len(train),
            "test_rows": len(test),
            "features": len(feature_cols),
            "train_subjects": int(train[config.subject_column].nunique()),
            "test_subjects": int(test[config.subject_column].nunique()),
        },
        "pca_cumvar": r(np.cumsum(pipe.pca.explained_variance_ratio_), 4),
        "feature_names": feature_cols,
        "examples": examples,
        "showcase": {
            name: {"label": text, "values": r(test[name].to_numpy(), 3)}
            for name, text in SHOWCASE_FEATURES.items()
        },
        "rows": {
            "subject": subjects.tolist(),
            "truth": truth.tolist(),
            "hmm": hmm_pred.tolist(),
            "lr": lr_pred.tolist(),
            "state": hidden.tolist(),
            "pc1": r(z_test[:, 0], 2),
            "pc2": r(z_test[:, 1], 2),
            "lr_proba": [r(p, 3) for p in lr_proba],
        },
        "chunks": chunks,
        "hmm_params": {
            name: {
                "startprob": r(model.startprob_),
                "transmat": [r(row) for row in model.transmat_],
                "means_pc12": [r(m[:2], 2) for m in model.means_],
            }
            for name, model in pipe.hmm_models.items()
        },
        "metrics": {"hmm": metrics(hmm_pred), "lr": metrics(lr_pred)},
    }

    template = args.template.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise SystemExit(f"Placeholder {PLACEHOLDER!r} not found in {args.template}")
    payload = json.dumps(data, separators=(",", ":"))
    args.output.write_text(template.replace(PLACEHOLDER, payload), encoding="utf-8")
    print(f"HMM accuracy {data['metrics']['hmm']['accuracy']}, "
          f"LR accuracy {data['metrics']['lr']['accuracy']}")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
