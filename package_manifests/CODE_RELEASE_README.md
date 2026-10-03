# Reproducibility code bundle notes

The code bundle contains the active implementation, tests, locked configurations, run histories, per-image metrics, statistical summaries, manuscript sources and audit reports. It intentionally excludes:

- raw public/private images (distributed separately or downloaded from providers),
- prediction-mask directories,
- logs and caches,
- private annotations,
- model checkpoints (approximately 14 GB locally; individual B7 files exceed normal GitHub file limits).

To reproduce with the existing environment:

```bash
pip install -e .
python -m unittest discover -s tests -v
python scripts/download_public_data.py
PYTHONPATH=src python scripts/train_public_study.py --data data/bbbc039
PYTHONPATH=src python scripts/train_public_baselines.py --data data/bbbc039
PYTHONPATH=src python scripts/train_validation_ablation.py --data data/bbbc039
PYTHONPATH=src:scripts python scripts/evaluate_grayscale_transfer.py --data data
```

Do not treat rerunning the same test set as independent confirmation. Archive the final selected checkpoint separately with its SHA-256 hash and an immutable release DOI.
