#!/usr/bin/env python3
"""Download official benchmark artifacts, checking recorded hashes when present."""
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URLS = {
    "bbbc038": {"stage1_test.zip": "https://data.broadinstitute.org/bbbc/BBBC038/stage1_test.zip", "stage1_solution.csv": "https://data.broadinstitute.org/bbbc/BBBC038/stage1_solution.csv", "metadata.xlsx": "https://data.broadinstitute.org/bbbc/BBBC038/metadata.xlsx"},
    "bbbc039": {name: "https://data.broadinstitute.org/bbbc/BBBC039/" + name for name in ("images.zip", "masks.zip", "metadata.zip")},
    "tnbc": {"TNBC_NucleiSegmentation.zip": "https://zenodo.org/records/2579118/files/TNBC_NucleiSegmentation.zip?download=1"},
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    for ds, urls in URLS.items():
        dest = ROOT / "data" / ds
        dest.mkdir(parents=True, exist_ok=True)
        protocol = ROOT / "outputs" / (ds + "_zero_shot") / "protocol.json"
        expected = json.loads(protocol.read_text())["files"] if protocol.exists() else {}
        for name, url in urls.items():
            path = dest / name
            if path.exists():
                if name in expected and digest(path) != expected[name]:
                    raise RuntimeError(f"Existing file has unexpected checksum; preserved: {path}")
                print("Verified/present", path)
                continue
            with tempfile.NamedTemporaryFile(dir=dest, suffix=".download", delete=False) as f:
                temp_path = Path(f.name)
            urllib.request.urlretrieve(url, temp_path)
            if name in expected and digest(temp_path) != expected[name]:
                raise RuntimeError(f"Downloaded file checksum differs from study record: {temp_path}")
            os.replace(temp_path, path)
            print("Downloaded", path)


if __name__ == "__main__":
    main()
