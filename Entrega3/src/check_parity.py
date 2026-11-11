"""Verifica la paridad numérica del artefacto tras recargarlo."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

ARTIFACT = BASE / "artifacts" / "model_final.joblib"


def main() -> None:
    artifact = joblib.load(ARTIFACT)
    final_model = artifact["final_model"]
    ref = artifact["reference"]
    X = pd.DataFrame([ref["input"]], columns=artifact["features"])
    p = final_model.predict_proba(X)[0]
    diff = float(np.max(np.abs(p - np.array(ref["proba"]))))
    print(f"artifact: {ARTIFACT}")
    print(f"reference_proba: {p.tolist()}")
    print(f"parity_diff: {diff:.2e}")
    assert diff <= 1e-9, f"Fallo de paridad: {diff}"
    print("PARITY OK")


if __name__ == "__main__":
    main()
