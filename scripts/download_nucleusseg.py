#!/usr/bin/env python3
"""Download and verify the research-only Bilkent NucleusSegData archive."""
import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path


URL = "https://www.cs.bilkent.edu.tr/~gunduz/downloads/NucleusSegData/NucleusSegData.zip"
SHA256 = "6c9dc687f6fdbb56a46cda575a36186f0186a9dc83edfccfc32ee924fedb721a"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/nucleusseg"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "NucleusSegData.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    if digest(archive) != SHA256:
        raise RuntimeError("NucleusSegData checksum mismatch")
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(args.output)
    print(f"Verified and extracted {archive}")


if __name__ == "__main__":
    main()
