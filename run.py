"""
===================================================================
  World Cup Predictor - Pickle Load Inference
  - lib.py의 기존 함수 재사용
  - 학습 없이 worldcup_home_model.pkl / worldcup_away_model.pkl load
  - historical_results.csv에서 feature 재계산
  - 2026 FIFA World Cup 경기 예측
  - 조별리그 + 토너먼트 결과 저장
===================================================================
"""

import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =====================================================================
# 1. 기존 lib.py에서 함수 / 상수 import
# =====================================================================
# lib.py에는 기존 코드가 저장되어 있어야 한다.
# 즉, 아래 함수와 상수가 lib.py 안에 정의되어 있어야 한다.
#
# - SEED
# - FEATURES
# - get_tournament_weight
# - compute_features
# - predict_all
# - format_submission
# - build_bracket_from_groups
# - simulate_full_knockouts

from lib import (
    SEED,
    FEATURES,
    get_tournament_weight,
    compute_features,
    predict_all,
    format_submission,
    build_bracket_from_groups,
    simulate_full_knockouts,
)


# =====================================================================
# 2. 경로 설정
# =====================================================================

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_PATH = BASE_DIR / "historical_results.csv"
THIRD_PLACE_PATH = BASE_DIR / "third_place_assignments_2026.csv"

HOME_MODEL_PATH = BASE_DIR / "worldcup_home_model.pkl"
AWAY_MODEL_PATH = BASE_DIR / "worldcup_away_model.pkl"

OUT_GROUP_PATH = BASE_DIR / "submission_from_pickle.csv"
OUT_FULL_PATH = BASE_DIR / "submission_from_pickle_full_tournament.csv"


# =====================================================================
# 3. pickle 모델 load
# =====================================================================

def load_models(home_model_path, away_model_path):
    """
    worldcup_home_model.pkl, worldcup_away_model.pkl을 load한다.

    pickle 내부 구조는 lib.py의 stacking_predict()에서 저장한 구조를 그대로 따른다.

    {
        "scaler": RobustScaler,
        "lgb_models": list,
        "cat_models": list,
        "xgb_models": list,
        "meta_model": Ridge
    }
    """

    if not home_model_path.exists():
        raise FileNotFoundError(f"홈 모델 pickle 파일을 찾을 수 없습니다: {home_model_path}")

    if not away_model_path.exists():
        raise FileNotFoundError(f"원정 모델 pickle 파일을 찾을 수 없습니다: {away_model_path}")

    home_model = joblib.load(home_model_path)
    away_model = joblib.load(away_model_path)

    required_keys = [
        "scaler",
        "lgb_models",
        "cat_models",
        "xgb_models",
        "meta_model",
    ]

    for model_name, model_dict in [
        ("HOME", home_model),
        ("AWAY", away_model),
    ]:
        if not isinstance(model_dict, dict):
            raise TypeError(
                f"{model_name} 모델은 dict 형태여야 합니다. "
                f"현재 타입: {type(model_dict)}"
            )

        for key in required_keys:
            if key not in model_dict:
                raise KeyError(f"{model_name} 모델에 필요한 key가 없습니다: {key}")

        print(f"\n[{model_name}] pickle load 완료")
        print("  scaler:", type(model_dict["scaler"]))
        print("  lgb_models:", len(model_dict["lgb_models"]))
        print("  cat_models:", len(model_dict["cat_models"]))
        print("  xgb_models:", len(model_dict["xgb_models"]))
        print("  meta_model:", type(model_dict["meta_model"]))

    return home_model, away_model


# =====================================================================
# 4. XGBoost CPU 보정
# =====================================================================

def set_xgb_cpu_if_possible(model_dict):
    """
    기존 학습 코드의 XGBoost는 device='cuda'로 저장되어 있다.

    GPU가 없는 다른 컴퓨터에서 pickle load 후 predict할 때 오류가 날 수 있으므로,
    예측 전에 device='cpu'로 바꾼다.

    학습된 tree 구조나 모델 파라미터를 재학습하는 것이 아니다.
    예측 실행 장치만 바꾸는 방어 코드다.
    """

    for model in model_dict["xgb_models"]:
        try:
            model.set_params(device="cpu")
        except Exception:
            pass


# =====================================================================
# 5. pickle 모델 추론 함수
# =====================================================================

def predict_lambda_from_pickle(X, model_dict, features):
    """
    pickle에서 load한 stacking 모델로 기대득점 lambda를 예측한다.

    원본 stacking_predict()의 추론 흐름을 그대로 재현한다.

    X
      → scaler.transform()
      → LightGBM fold model 평균
      → CatBoost fold model 평균
      → XGBoost fold model 평균
      → Ridge meta_model
      → clip(0.05, 12.0)
    """

    X = X.copy()

    missing_cols = [col for col in features if col not in X.columns]
    if missing_cols:
        raise ValueError(f"입력 데이터에 필요한 feature가 없습니다: {missing_cols}")

    # 학습 당시 feature 순서 고정
    X = X[features]

    if X.isna().sum().sum() > 0:
        print("\n예측 feature 결측치:")
        print(X.isna().sum()[X.isna().sum() > 0])
        raise ValueError("예측 feature에 결측치가 있습니다.")

    X_scaled = pd.DataFrame(
        model_dict["scaler"].transform(X),
        columns=features,
        index=X.index,
    )

    base_preds = np.zeros((len(X_scaled), 3))

    for model in model_dict["lgb_models"]:
        base_preds[:, 0] += model.predict(X_scaled) / len(model_dict["lgb_models"])

    for model in model_dict["cat_models"]:
        base_preds[:, 1] += model.predict(X_scaled) / len(model_dict["cat_models"])

    for model in model_dict["xgb_models"]:
        base_preds[:, 2] += model.predict(X_scaled) / len(model_dict["xgb_models"])

    final_pred = model_dict["meta_model"].predict(base_preds)
    final_pred = np.clip(final_pred, 0.05, 12.0)

    return final_pred


# =====================================================================
# 6. 데이터 load + 전처리
# =====================================================================

def load_and_prepare_data(data_path):
    """
    historical_results.csv를 load하고,
    lib.py의 compute_features()가 요구하는 기본 컬럼을 준비한다.
    """

    if not data_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

    df = pd.read_csv(data_path, encoding_errors="replace")

    required_columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"historical_results.csv에 필요한 컬럼이 없습니다: {missing_columns}")

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().sum() > 0:
        bad_count = df["date"].isna().sum()
        raise ValueError(f"date 컬럼에서 datetime 변환 실패가 발생했습니다: {bad_count}개")

    df = df.sort_values("date").reset_index(drop=True)

    df["is_neutral"] = (
        df["neutral"]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(["TRUE", "1", "YES", "Y"])
        .astype(int)
    )

    df["tournament_weight"] = df["tournament"].apply(get_tournament_weight)

    print("\n데이터 로드 완료")
    print("df.shape:", df.shape)
    print("date range:", df["date"].min(), "~", df["date"].max())

    return df


# =====================================================================
# 7. 2026 FIFA World Cup 예측 대상 분리
# =====================================================================

def split_worldcup_2026_matches(df):
    """
    전체 경기 데이터에서 2026 FIFA World Cup 경기만 예측 대상으로 분리한다.
    """

    mask_wc_2026 = (
        (df["date"].dt.year == 2026)
        & (df["tournament"] == "FIFA World Cup")
    )

    df_test = df.loc[mask_wc_2026].copy()

    if df_test.empty:
        raise ValueError(
            "2026 FIFA World Cup 예측 대상 경기가 없습니다.\n"
            "historical_results.csv 안에 date가 2026년이고 "
            "tournament == 'FIFA World Cup'인 행이 필요합니다."
        )

    print("\n예측 대상 경기 수:", len(df_test))
    print(df_test[["date", "home_team", "away_team", "tournament"]].head())

    return df_test


# =====================================================================
# 8. 조별리그 예측
# =====================================================================

def predict_group_stage_from_pickle(df_test, home_model, away_model):
    """
    pickle 모델로 lambda_home, lambda_away를 예측한 뒤,
    lib.py의 predict_all()을 사용해 Dixon-Coles Monte Carlo 시뮬레이션을 수행한다.
    """

    X_te = df_test[FEATURES].copy()
    X_te = X_te.fillna(1.0)

    df_test = df_test.copy()

    df_test["lambda_home"] = predict_lambda_from_pickle(
        X_te,
        home_model,
        FEATURES,
    )

    df_test["lambda_away"] = predict_lambda_from_pickle(
        X_te,
        away_model,
        FEATURES,
    )

    print("\n기대득점 lambda 예측 완료")
    print(
        df_test[
            [
                "home_team",
                "away_team",
                "lambda_home",
                "lambda_away",
            ]
        ].head()
    )

    # 기존 lib.py의 Dixon-Coles Monte Carlo 시뮬레이션 재사용
    df_pred = predict_all(df_test, n_sim=10000)
    df_pred = format_submission(df_pred)

    return df_pred, df_test


# =====================================================================
# 9. 토너먼트 예측
# =====================================================================

def predict_full_tournament_if_possible(df_pred, df_test):
    """
    third_place_assignments_2026.csv가 있으면 32강 이후 토너먼트까지 예측한다.

    주의:
    lib.py의 simulate_full_knockouts()는 함수 내부에서
    worldcup_home_model.pkl, worldcup_away_model.pkl을 다시 load한다.

    따라서 이 스크립트는 실행 위치를 BASE_DIR로 맞춘 뒤 실행한다.
    """

    if not THIRD_PLACE_PATH.exists():
        print("\nthird_place_assignments_2026.csv를 찾을 수 없어 토너먼트 예측은 건너뜁니다.")
        print("경로:", THIRD_PLACE_PATH)
        return None

    print("\n32강 대진표 생성 시작")
    r32_matches = build_bracket_from_groups(
        df_pred,
        str(THIRD_PLACE_PATH),
    )

    print("32강 경기 수:", len(r32_matches))

    df_knockouts = simulate_full_knockouts(
        r32_matches,
        df_test,
    )

    if df_knockouts is None or df_knockouts.empty:
        print("\n토너먼트 예측 결과가 비어 있습니다.")
        return None

    df_full = pd.concat(
        [df_pred, df_knockouts],
        ignore_index=True,
    )

    df_full = format_submission(df_full)

    return df_full


# =====================================================================
# 10. Main
# =====================================================================

def main():
    print("=" * 70)
    print("World Cup Predictor - Pickle Load Inference")
    print("=" * 70)

    print("\n경로 확인")
    print("BASE_DIR:", BASE_DIR)
    print("DATA_PATH:", DATA_PATH)
    print("HOME_MODEL_PATH:", HOME_MODEL_PATH)
    print("AWAY_MODEL_PATH:", AWAY_MODEL_PATH)
    print("THIRD_PLACE_PATH:", THIRD_PLACE_PATH)

    # lib.py의 simulate_full_knockouts()가 상대경로로 pickle을 다시 load하므로,
    # 현재 작업 디렉터리를 BASE_DIR로 맞춘다.
    os.chdir(BASE_DIR)

    np.random.seed(SEED)

    # 1. 데이터 로드
    df = load_and_prepare_data(DATA_PATH)

    # 2. feature 계산 전, 이미 feature 컬럼이 들어 있는 경우 제거
    # existing_feature_cols = [col for col in FEATURES if col in df.columns]

    # if existing_feature_cols:
    #     print("\n이미 존재하는 feature 컬럼 제거:")
    #     print(existing_feature_cols)
    #     df = df.drop(columns=existing_feature_cols)

    protected_cols = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
        "is_neutral",
        "tournament_weight",
    ]

    existing_feature_cols = [
        col for col in FEATURES
        if col in df.columns and col not in protected_cols
    ]

    if existing_feature_cols:
        print("\n이미 존재하는 feature 컬럼 제거:")
        print(existing_feature_cols)
        df = df.drop(columns=existing_feature_cols)

    # 3. Feature Engineering
    print("\n피처 계산 시작")
    df = compute_features(df)

    duplicated_cols = df.columns[df.columns.duplicated()].tolist()

    if duplicated_cols:
        raise ValueError(f"Feature Engineering 이후 중복 컬럼이 있습니다: {duplicated_cols}")

    # 4. 예측 대상 분리
    df_test = split_worldcup_2026_matches(df)

    # 5. pickle load
    home_model, away_model = load_models(
        HOME_MODEL_PATH,
        AWAY_MODEL_PATH,
    )

    # 6. XGBoost CPU 보정
    set_xgb_cpu_if_possible(home_model)
    set_xgb_cpu_if_possible(away_model)

    # 7. 조별리그 예측
    df_pred, df_test_with_lambda = predict_group_stage_from_pickle(
        df_test,
        home_model,
        away_model,
    )

    df_pred.to_csv(
        OUT_GROUP_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n조별리그 예측 저장 완료")
    print("저장 파일:", OUT_GROUP_PATH)
    print(df_pred.head(10))

    # 8. 토너먼트 예측
    df_full = predict_full_tournament_if_possible(
        df_pred,
        df_test_with_lambda,
    )

    if df_full is not None:
        df_full.to_csv(
            OUT_FULL_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print("\n전체 토너먼트 예측 저장 완료")
        print("저장 파일:", OUT_FULL_PATH)
        print(df_full.head(10))

    print("\n" + "=" * 70)
    print("Pickle load inference 완료")
    print("=" * 70)

    return df_pred, df_test_with_lambda, df_full


# =====================================================================
# 11. 실행
# =====================================================================

if __name__ == "__main__":
    df_pred, df_test_with_lambda, df_full = main()