"""Check the archived loader and model with explicitly synthetic inputs."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spec = importlib.util.spec_from_file_location("archived_eeg_model", ROOT / "scripts" / "eeg_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    torch.manual_seed(2025)
    torch.set_num_threads(2)
    rng = np.random.default_rng(2025)
    scratch = ROOT / ".local"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="synthetic-smoke-", dir=scratch) as directory:
        temporary_root = Path(directory)
        for group in ("patients", "controls"):
            subject = temporary_root / group / f"SYNTHETIC_{group}"
            subject.mkdir(parents=True)
            array = rng.standard_normal((1, 8, 40, 128)).astype(np.float32)
            np.save(subject / "synthetic_stockwell_selected8ch.npy", array)

        dataset = module.EEGDataset(
            str(temporary_root / "patients"),
            str(temporary_root / "controls"),
            downsample_factor=8,
            max_length=16,
        )
        if len(dataset) != 2:
            raise AssertionError("The loader must find both synthetic files")
        inputs, labels = module.custom_collate([dataset[0], dataset[1]])
        if tuple(inputs.shape) != (2, 8, 40, 16) or set(labels.tolist()) != {0.0, 1.0}:
            raise AssertionError("Unexpected loader shape or directory-derived labels")

        model = module.EEGSeizureModel(
            num_channels=8,
            num_frequencies=40,
            hidden_dim=16,
            transformer_layers=1,
            transformer_heads=4,
            dropout=0.0,
            use_checkpointing=False,
        ).eval()
        with torch.no_grad():
            logits = model(inputs)
        if tuple(logits.shape) != (2, 1) or not torch.isfinite(logits).all().item():
            raise AssertionError("Unexpected model output shape or non-finite logits")

    print("PASS: synthetic NPY files -> loader -> collate -> CNN/Transformer/Euler module -> (2, 1) logits")
    print("This verifies code structure only. No original data, trained weights or performance metrics were used.")


if __name__ == "__main__":
    main()
