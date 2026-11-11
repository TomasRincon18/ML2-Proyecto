"""Pipeline reproducible: preprocesamiento + regresión logística.

El mismo objeto Pipeline se usa en entrenamiento, calibración, interpretación
y serving. Así se evita que el escalamiento o el one-hot se ajusten con datos
de validación o de despliegue.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    CLASS_WEIGHT,
    LR_C,
    LR_MAX_ITER,
    NUMERIC_FEATURES,
    SEED,
)


def build_preprocessor() -> ColumnTransformer:
    """Escalamiento numérico + one-hot categórico, ajustado solo en train."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model(c: float = LR_C) -> Pipeline:
    """Pipeline completo con la configuración congelada de la fase 2."""
    return Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    class_weight=CLASS_WEIGHT,
                    max_iter=LR_MAX_ITER,
                    random_state=SEED,
                ),
            ),
        ]
    )


def transformed_feature_names(model: Pipeline) -> list[str]:
    """Nombres de las columnas después del ColumnTransformer."""
    prep = model.named_steps["prep"]
    return list(prep.get_feature_names_out())


def original_feature_names() -> list[str]:
    """Variables originales en el orden del contrato de entrada."""
    return [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
