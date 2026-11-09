"""Download the UCI HAR dataset and write Data/train.csv and Data/test.csv.

The official UCI archive needs no login (unlike the Kaggle mirror). The output
CSVs use the Kaggle layout: 561 feature columns, then `subject`, then `Activity`.
"""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
ROOT = "UCI HAR Dataset/"


def _read_table(archive: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    with archive.open(ROOT + name) as handle:
        return pd.read_csv(handle, sep=r"\s+", header=None, **kwargs)


def _unique_names(names):
    """Feature names in features.txt repeat; append a counter to duplicates."""
    seen = {}
    result = []
    for name in names:
        count = seen.get(name, 0)
        result.append(name if count == 0 else f"{name}-{count}")
        seen[name] = count + 1
    return result


def build_split(archive: zipfile.ZipFile, split: str, features, labels) -> pd.DataFrame:
    frame = _read_table(archive, f"{split}/X_{split}.txt")
    frame.columns = features
    frame["subject"] = _read_table(archive, f"{split}/subject_{split}.txt")[0].to_numpy()
    codes = _read_table(archive, f"{split}/y_{split}.txt")[0]
    frame["Activity"] = codes.map(labels).to_numpy()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("Data"))
    parser.add_argument("--url", default=UCI_URL)
    args = parser.parse_args()

    print("Downloading", args.url, "(about 60 MB) ...")
    with urllib.request.urlopen(args.url) as response:
        outer = zipfile.ZipFile(io.BytesIO(response.read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read("UCI HAR Dataset.zip")))

    features = _unique_names(_read_table(inner, "features.txt")[1].tolist())
    labels = dict(_read_table(inner, "activity_labels.txt").to_numpy().tolist())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        frame = build_split(inner, split, features, labels)
        path = args.out_dir / f"{split}.csv"
        frame.to_csv(path, index=False)
        print(f"Wrote {path}: {len(frame)} rows, {frame.shape[1]} columns")


if __name__ == "__main__":
    main()
