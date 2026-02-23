from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline

from src.ml_pipeline import (
    build_models,
    build_preprocessor,
    load_dataset,
    split_features_target,
    train_test_split_stratified,
)


def main() -> None:
    df = load_dataset("data/processed/churn_final.csv")
    target_col = "churn_binary"
    drop_cols = ["customerID", "Churn", target_col]
    x, y = split_features_target(df, target_col, drop_cols)
    categorical_cols = x.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    splits = train_test_split_stratified(x, y, test_size=0.2, random_state=42)
    models = build_models(random_state=42)
    _, model = next(iter(models.items()))
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model.named_steps["model"])])
    pipeline.fit(splits.x_train, splits.y_train)
    score = pipeline.score(splits.x_test, splits.y_test)
    print(score)


if __name__ == "__main__":
    main()
