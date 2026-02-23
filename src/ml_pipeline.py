from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class DatasetSplit:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Iterable[str],
) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=list(drop_cols))
    y = df[target_col]
    return x, y


def build_preprocessor(
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> ColumnTransformer:
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric = StandardScaler()
    return ColumnTransformer(
        [
            ("categorical", categorical, categorical_cols),
            ("numeric", numeric, numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline(
            [
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state))
            ]
        ),
        "RandomForest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=random_state,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                (
                    "model",
                    GradientBoostingClassifier(random_state=random_state),
                )
            ]
        ),
    }


def train_test_split_stratified(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> DatasetSplit:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return DatasetSplit(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def extract_feature_names(
    preprocessor: ColumnTransformer,
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> list[str]:
    cat_encoder = preprocessor.named_transformers_["categorical"]
    cat_names = list(cat_encoder.get_feature_names_out(categorical_cols))
    return cat_names + list(numeric_cols)


def to_dataframe(
    array: np.ndarray,
    columns: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(array, columns=columns)
