import os
import re
import numpy as np
import pandas as pd

CSV_IN = os.path.join(os.path.dirname(__file__), "features_with_U_and_ratios.csv")

# -------- config --------
TRAIN_MIN_C = 105
TRAIN_MAX_C = 130
TEST_EXTRA_C = 100          # alla 100°C in i test
TRAIN_FRACTION = 0.80
RANDOM_SEED = 42

# features
USE_FEATURES = ["U_A", "U_B", "U_C", "U_D", "ntc_mean"]  # ntc_mean är i °C i din CSV
TARGET = "y_tc_mean"  # target temp (°C)

SURF_VALUES = ["J1", "J2", "J3", "J4", "J5", "J6"]

def extract_surface_from_filename(s: str):
    if s is None:
        return None
    base = os.path.basename(str(s))
    m = re.match(r"^(J[1-6])", base, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"(J[1-6])", base, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    return None

def rmse(y, yhat):
    y = np.asarray(y, float).reshape(-1)
    yhat = np.asarray(yhat, float).reshape(-1)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def mae(y, yhat):
    y = np.asarray(y, float).reshape(-1)
    yhat = np.asarray(yhat, float).reshape(-1)
    return float(np.mean(np.abs(y - yhat)))

def main():
    if not os.path.exists(CSV_IN):
        raise FileNotFoundError(f"Missing: {CSV_IN}\nRun make_U_and_ratios.py first.")

    df = pd.read_csv(CSV_IN)

    # surface from file
    if "file" not in df.columns:
        raise KeyError("CSV saknar kolumnen 'file'.")
    df["surface"] = df["file"].apply(extract_surface_from_filename)
    df = df[df["surface"].isin(SURF_VALUES)].copy()

    # kräver features + target
    need = [TARGET] + USE_FEATURES
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

    # rensa NaN
    df = df.dropna(subset=need).copy()

    # välj train/test pool
    df_train_pool = df[(df[TARGET] >= TRAIN_MIN_C) & (df[TARGET] <= TRAIN_MAX_C)].copy()
    df_test_extra = df[np.isclose(df[TARGET], TEST_EXTRA_C)].copy()  # alla 100°C

    # 80/20 split på train-pool (random)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(len(df_train_pool))
    rng.shuffle(idx)

    n_train = int(np.floor(TRAIN_FRACTION * len(idx)))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    df_train = df_train_pool.iloc[train_idx].copy()
    df_test_105_130 = df_train_pool.iloc[test_idx].copy()

    # slutligt test = 20% av 105-130 + alla 100
    df_test = pd.concat([df_test_105_130, df_test_extra], ignore_index=True)

    print("\n--- Split sizes ---")
    print("Train (105-130):", len(df_train))
    print("Test  (20% of 105-130):", len(df_test_105_130))
    print("Test extra (all 100C):", len(df_test_extra))
    print("Test total:", len(df_test))

    # ----- build design matrices -----
    X_train = df_train[USE_FEATURES].to_numpy(float)
    y_train = df_train[TARGET].to_numpy(float)

    X_test = df_test[USE_FEATURES].to_numpy(float)
    y_test = df_test[TARGET].to_numpy(float)

    # add intercept column
    X_train_i = np.column_stack([np.ones(len(X_train)), X_train])
    X_test_i = np.column_stack([np.ones(len(X_test)), X_test])

    # ----- fit by least squares -----
    # p = argmin ||X p - y||
    p, *_ = np.linalg.lstsq(X_train_i, y_train, rcond=None)

    # predictions
    yhat_train = X_train_i @ p
    yhat_test = X_test_i @ p

    # metrics
    print("\n--- Linear least squares model ---")
    print("Features:", ["intercept"] + USE_FEATURES)
    print("Coefficients:", p)

    print("\nTrain RMSE (°C):", rmse(y_train, yhat_train))
    print("Train MAE  (°C):", mae(y_train, yhat_train))

    print("\nTest  RMSE (°C):", rmse(y_test, yhat_test))
    print("Test  MAE  (°C):", mae(y_test, yhat_test))

    # --- also report 100C-only performance ---
    if len(df_test_extra) > 0:
        X_100 = df_test_extra[USE_FEATURES].to_numpy(float)
        y_100 = df_test_extra[TARGET].to_numpy(float)
        X_100_i = np.column_stack([np.ones(len(X_100)), X_100])
        yhat_100 = X_100_i @ p
        print("\n100C-only RMSE (°C):", rmse(y_100, yhat_100))
        print("100C-only MAE  (°C):", mae(y_100, yhat_100))

    # --- baseline: predict mean of train ---
    y_mean = float(np.mean(y_train))
    yhat_base = np.full_like(y_test, y_mean)
    print("\n--- Baseline (predict train mean) ---")
    print("Baseline Test RMSE (°C):", rmse(y_test, yhat_base))
    print("Baseline Test MAE  (°C):", mae(y_test, yhat_base))

    # save predictions for plotting later
    out = df_test.copy()
    out["T_pred_C"] = yhat_test
    out_path = os.path.join(os.path.dirname(__file__), "temp_model_test_predictions.csv")
    out.to_csv(out_path, index=False)
    print("\nSaved test predictions:", out_path)

if __name__ == "__main__":
    main()
