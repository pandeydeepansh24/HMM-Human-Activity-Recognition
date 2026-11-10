"""Experiments and figures for the paper in paper/.

Runs the default pipeline, ablations (hidden states, chunk length, PCA size),
diagnostic variants (oracle activity-aligned chunks, chunk-voted Logistic
Regression) and per-subject accuracy. Writes paper/results.json and PDF figures
into paper/figures/.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from activity_hmm.pipeline import (  # noqa: E402
    ActivityRecognitionPipeline,
    PipelineConfig,
    _sequence_bounds,
    _subject_windows,
    load_har_data,
)

OUT = ROOT / "paper"
FIG = OUT / "figures"
SHORT = {"LAYING": "Lay", "SITTING": "Sit", "STANDING": "Stand", "WALKING": "Walk",
         "WALKING_DOWNSTAIRS": "Down", "WALKING_UPSTAIRS": "Up"}
COLORS = ["#3c6cb3", "#8660b5", "#2a9484", "#dc8424", "#c4443b", "#a88d12"]


def hmm_predict(pipe, z, bounds, labels):
    pred = np.empty(len(z), dtype=object)
    for s, e in bounds:
        scores = [pipe.hmm_models[name].score(z[s:e]) for name in labels]
        pred[s:e] = labels[int(np.argmax(scores))]
    return pred.astype(str)


def summary(truth, pred, labels):
    p, r, f, _ = precision_recall_fscore_support(truth, pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(np.mean(truth == pred)),
        "f1_macro": float(f1_score(truth, pred, labels=labels, average="macro")),
        "precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(),
        "confusion": confusion_matrix(truth, pred, labels=labels).tolist(),
    }


def fit(train, **kw):
    t0 = time.perf_counter()
    pipe = ActivityRecognitionPipeline(PipelineConfig(**kw)).fit(train)
    return pipe, time.perf_counter() - t0


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    train, test = load_har_data(ROOT / "Data/train.csv", ROOT / "Data/test.csv")
    truth = test["Activity"].to_numpy().astype(str)
    subj = test["subject"].to_numpy()
    res = {"sizes": {"train": len(train), "test": len(test),
                     "train_subjects": int(train.subject.nunique()),
                     "test_subjects": int(test.subject.nunique()),
                     "train_runs": sum(1 for _ in _sequence_bounds(train, "subject", "Activity")),
                     "test_runs": sum(1 for _ in _sequence_bounds(test, "subject", "Activity"))}}
    res["class_counts"] = {"train": dict(Counter(train.Activity)), "test": dict(Counter(test.Activity))}

    # ---- default configuration
    pipe, t_fit = fit(train)
    labels = pipe.labels
    z = pipe._transform(test)
    cfg = pipe.config
    t0 = time.perf_counter()
    chunks = list(_subject_windows(test, "subject", cfg.window_size))
    hmm = hmm_predict(pipe, z, chunks, labels)
    t_pred = time.perf_counter() - t0
    lr = pipe.baseline.predict(z).astype(str)
    res["labels"] = labels
    res["pca_var_80"] = float(pipe.pca.explained_variance_ratio_.sum())
    res["pca_cumvar"] = np.cumsum(pipe.pca.explained_variance_ratio_).tolist()
    res["timing"] = {"fit_s": t_fit, "hmm_predict_s": t_pred}
    res["hmm"] = summary(truth, hmm, labels)
    res["lr"] = summary(truth, lr, labels)
    res["n_chunks"] = len(chunks)
    mixed = sum(1 for s, e in chunks if len(set(truth[s:e])) > 1)
    res["mixed_chunks"] = mixed
    res["transmat"] = {k: m.transmat_.tolist() for k, m in pipe.hmm_models.items()}

    # ---- diagnostics: oracle chunks aligned with activity runs, LR chunk vote
    oracle = []
    for s, e in _sequence_bounds(test, "subject", "Activity"):
        oracle += [(a, min(a + cfg.window_size, e)) for a in range(s, e, cfg.window_size)]
    res["hmm_oracle_chunks"] = summary(truth, hmm_predict(pipe, z, oracle, labels), labels)
    res["hmm_single_window"] = summary(
        truth, hmm_predict(pipe, z, [(i, i + 1) for i in range(len(z))], labels), labels)
    vote = lr.copy()
    for s, e in chunks:
        vote[s:e] = Counter(lr[s:e]).most_common(1)[0][0]
    res["lr_chunk_vote"] = summary(truth, vote, labels)

    # ---- per-subject accuracy
    res["per_subject"] = {int(s): {"hmm": float(np.mean(hmm[subj == s] == truth[subj == s])),
                                   "lr": float(np.mean(lr[subj == s] == truth[subj == s]))}
                          for s in np.unique(subj)}

    # ---- ablations
    abl = {"states": {}, "window": {}, "pca": {}}
    for k in [1, 2, 3, 4, 5]:
        p, _ = fit(train, hmm_states=k)
        zz = p._transform(test)
        abl["states"][k] = float(np.mean(hmm_predict(p, zz, chunks, labels) == truth))
    for w in [1, 2, 4, 8, 16, 32, 64]:
        cw = list(_subject_windows(test, "subject", w))
        abl["window"][w] = float(np.mean(hmm_predict(pipe, z, cw, labels) == truth))
    for c in [10, 20, 40, 80, 160]:
        p, _ = fit(train, pca_components=c)
        zz = p._transform(test)
        abl["pca"][c] = {"hmm": float(np.mean(hmm_predict(p, zz, chunks, labels) == truth)),
                         "lr": float(np.mean(p.baseline.predict(zz).astype(str) == truth)),
                         "var": float(p.pca.explained_variance_ratio_.sum())}
    res["ablation"] = abl
    print(json.dumps({k: res[k] for k in ["ablation", "per_subject", "mixed_chunks", "n_chunks", "timing"]}, indent=1))
    for k in ["hmm", "lr", "hmm_oracle_chunks", "hmm_single_window", "lr_chunk_vote"]:
        print(k, round(res[k]["accuracy"], 4), round(res[k]["f1_macro"], 4))
    (OUT / "results.json").write_text(json.dumps(res, indent=1))

    # ---- figures
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    short = [SHORT[n] for n in labels]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    idx = np.random.default_rng(0).permutation(len(z))
    for i, name in enumerate(labels):
        m = idx[truth[idx] == name]
        ax.scatter(z[m, 0], z[m, 1], s=2, alpha=0.5, color=COLORS[i], label=short[i], linewidths=0)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.legend(markerscale=4, fontsize=6.5, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "pca.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, key, title in zip(axes, ["hmm", "lr"], ["(a) Class-conditional HMM", "(b) Logistic Regression"]):
        cm = np.array(res[key]["confusion"]); norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6.5,
                        color="white" if norm[i, j] > 0.55 else "black")
        ax.set_xticks(range(6), short, rotation=45, ha="right", rotation_mode="anchor"); ax.set_yticks(range(6), short)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=8)
        ax.spines[:].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "confusion.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.7))
    ks = list(abl["states"]); axes[0].plot(ks, [abl["states"][k] * 100 for k in ks], "o-", color="#0d6a88", ms=3)
    axes[0].set_xlabel("Hidden states $N$"); axes[0].set_ylabel("HMM accuracy (%)")
    ws = list(abl["window"]); axes[1].plot(ws, [abl["window"][w] * 100 for w in ws], "o-", color="#b95a2a", ms=3)
    axes[1].set_xscale("log", base=2); axes[1].set_xticks(ws, [str(w) for w in ws])
    axes[1].set_xlabel("Chunk length $T$ (windows)")
    for ax in axes: ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(); fig.savefig(FIG / "ablation.pdf"); plt.close(fig)

    s0 = 9  # one test subject as an illustration
    m = np.where(subj == s0)[0]
    fig, ax = plt.subplots(figsize=(7.0, 1.05))
    code = {n: i for i, n in enumerate(labels)}
    for r, (arr, name) in enumerate([(truth, "True"), (hmm, "HMM"), (lr, "LogReg")]):
        ax.scatter(np.arange(len(m)), np.full(len(m), 2 - r), c=[COLORS[code[v]] for v in arr[m]],
                   marker="|", s=110, linewidths=1.4)
    ax.set_yticks([2, 1, 0], ["True", "HMM", "LogReg"]); ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel(f"Window index (test subject {s0})"); ax.set_xlim(-1, len(m))
    ax.spines[["left"]].set_visible(False); ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], color=COLORS[i], lw=3) for i in range(6)]
    ax.legend(handles, short, ncol=6, fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(FIG / "timeline.pdf"); plt.close(fig)
    res["timeline_subject"] = s0
    (OUT / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
