# train.py
import os, glob
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump

DATA_DIR   = "data"
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "asl_model.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

def load_all_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        raise RuntimeError(f"No CSVs found in {DATA_DIR}/")

    # Use the FIRST file's column order as canonical
    base = pd.read_csv(files[0])
    cols = list(base.columns)
    if cols[0] != "label":
        raise RuntimeError("Expected first column to be 'label'")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Bring to the same column order (union-fill missing)
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
        extra = [c for c in df.columns if c not in cols]
        if extra:
            # if later files added new cols, extend the canonical list (fill old dfs later if needed)
            for c in extra:
                base[c] = 0.0
            cols += extra
            # also add missing extras to earlier frames when we concat below
        dfs.append(df[cols])

    data = pd.concat(dfs, axis=0, ignore_index=True)

    # Optional guard: drop all-zero feature rows (often “no hands”)
    feat_cols = [c for c in cols if c != "label"]
    zero_mask = (data[feat_cols].abs().sum(axis=1) == 0)
    if zero_mask.any():
        data = data[~zero_mask].reset_index(drop=True)

    return data, cols

def main():
    data, cols = load_all_data()
    feat_cols = [c for c in cols if c != "label"]

    X = data[feat_cols].to_numpy(dtype=np.float32)
    y_text = data["label"].astype(str).to_numpy()

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_text)

    # Train/val split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None
    )

    # Simple, strong baseline (works well on tabular):
    pipe = Pipeline([
        ("impute",  SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scale",   StandardScaler(with_mean=False)),   # sparse-friendly if needed
        ("clf",     LogisticRegression(
                        max_iter=2000,
                        multi_class="auto",
                        class_weight="balanced",        # helps with label imbalance
                        n_jobs=None
                    )
        )
    ])

    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    print("\n=== Validation ===")
    print("Accuracy:", f"{accuracy_score(yte, pred):.3f}")
    print(classification_report(yte, pred, target_names=le.classes_))

    artifact = {
        "model": pipe,
        "label_encoder_classes_": le.classes_,  # for inverse_transform at inference
        "feature_order": feat_cols             # EXACT order to build the vector
    }
    dump(artifact, MODEL_PATH)
    print("\nSaved:", MODEL_PATH)

if __name__ == "__main__":
    main()
