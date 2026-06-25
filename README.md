# 2026 World Cup Bracket Predictor

2026 FIFA World Cup 경기 결과를 예측하는 축구 경기 예측 프로젝트입니다. 과거 A매치 결과 데이터로부터 Elo, 최근 득실, 폼, H2H 등 피처를 계산하고, 미리 학습해 둔 스태킹 모델 pickle을 불러와 조별리그와 토너먼트 결과를 생성합니다.

기본 실행 경로는 `run.py`입니다. 학습을 다시 하지 않고 `worldcup_home_model.pkl`, `worldcup_away_model.pkl`을 로드해 빠르게 추론합니다.

## Data Source

이 프로젝트의 기본 경기 데이터는 Kaggle의 International football results dataset을 기반으로 합니다.

* [International football results from 1872 to 2017 (by martj42)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017?select=results.csv)

현재 repository의 `historical_results.csv`는 위 데이터셋 형식의 컬럼을 사용하며, 2026 FIFA World Cup 예측 대상 경기 행을 추가로 포함합니다. 예측 대상 행은 `home_score`, `away_score`가 비어 있고, `date`가 2026년이며 `tournament` 값이 `FIFA World Cup`이어야 합니다.

## Repository Structure

```text
.
├── README.md
├── run.py
├── lib.py
├── geminifootpredict.ipynb
├── historical_results.csv
├── third_place_assignments_2026.csv
├── worldcup_home_model.pkl
├── worldcup_away_model.pkl
├── submission_from_pickle.csv
├── submission_from_pickle_full_tournament.csv
└── submission_consensus.csv
```

주요 파일:

- `run.py`: pickle 모델을 불러와 예측을 실행하는 메인 스크립트
- `lib.py`: 피처 엔지니어링, 모델 학습/예측, 몬테카를로 시뮬레이션 함수 모음
- `historical_results.csv`: 과거 A매치 결과 및 2026 월드컵 예측 대상 경기 데이터
- `third_place_assignments_2026.csv`: 2026 월드컵 3위 팀 조합별 32강 배정 규칙
- `worldcup_home_model.pkl`: 홈팀 득점 예측용 학습 모델
- `worldcup_away_model.pkl`: 원정팀 득점 예측용 학습 모델
- `geminifootpredict.ipynb`: 실험 및 학습 과정을 확인할 수 있는 Jupyter Notebook

## Requirements

- Python 3.10 이상 권장
- Git
- Python packages:
  - `pandas`
  - `numpy`
  - `scipy`
  - `scikit-learn`
  - `lightgbm`
  - `catboost`
  - `xgboost`
  - `tqdm`
  - `joblib`
  - `jupyter`, `jupyterlab`, `nbconvert`는 노트북 실행 시 필요

## Quick Start

Repository를 내려받습니다.

```bash
git clone https://github.com/syp0728/facamp_2026.git
cd facamp_2026
```

가상환경을 만들고 패키지를 설치합니다.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install pandas numpy scipy scikit-learn lightgbm catboost xgboost tqdm joblib jupyter jupyterlab nbconvert ipykernel
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install pandas numpy scipy scikit-learn lightgbm catboost xgboost tqdm joblib jupyter jupyterlab nbconvert ipykernel
```

예측을 실행합니다.

```bash
python run.py
```

실행이 끝나면 아래 파일이 생성 또는 갱신됩니다.

```text
submission_from_pickle.csv
submission_from_pickle_full_tournament.csv
```

## Expected Outputs

- `submission_from_pickle.csv`: 조별리그 예측 결과
- `submission_from_pickle_full_tournament.csv`: 조별리그부터 결승까지 포함한 전체 토너먼트 예측 결과

출력 CSV는 아래 컬럼을 포함합니다.

```text
team1, team2, team1_score, team2_score, team1_prob, team2_prob, type
```

## How It Works

`run.py`는 다음 순서로 동작합니다.

1. `historical_results.csv`를 로드합니다.
2. `lib.py`의 `compute_features()`로 Elo, 공격/수비 평균, 최근 폼, H2H 등 피처를 계산합니다.
3. `date`가 2026년이고 `tournament`가 `FIFA World Cup`인 행만 예측 대상으로 분리합니다.
4. `worldcup_home_model.pkl`, `worldcup_away_model.pkl`을 로드합니다.
5. LightGBM, CatBoost, XGBoost, Ridge meta model로 홈/원정 기대 득점 lambda를 예측합니다.
6. Dixon-Coles Monte Carlo simulation을 10,000회 실행해 경기별 스코어와 승률을 추정합니다.
7. `third_place_assignments_2026.csv`를 사용해 32강 대진을 구성하고 토너먼트 결과를 예측합니다.

## Dataset Format

`historical_results.csv`에는 최소한 아래 컬럼이 필요합니다.

```text
date, home_team, away_team, home_score, away_score, tournament, neutral
```

현재 파일은 Kaggle 원본과 같은 형식으로 다음 컬럼을 포함합니다.

```text
date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
```

2026 월드컵 예측 대상 경기는 예를 들어 다음과 같은 형태입니다.

```csv
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2026-06-27,Panama,England,NA,NA,FIFA World Cup,East Rutherford,United States,TRUE
```

## Run the Notebook

노트북에서 실험 과정을 확인하려면 다음 명령어를 사용합니다.

```bash
jupyter notebook geminifootpredict.ipynb
```

또는 JupyterLab:

```bash
jupyter lab geminifootpredict.ipynb
```

터미널에서 노트북 전체를 실행하려면:

```bash
jupyter nbconvert --to notebook --execute geminifootpredict.ipynb --output geminifootpredict_executed.ipynb
```

## Reproducibility

코드는 `SEED = 9`를 사용합니다. 동일한 입력 데이터, 동일한 pickle 모델, 동일한 패키지 버전에서 실행하면 같은 결과를 재현할 수 있도록 구성되어 있습니다.

## Troubleshooting

### `historical_results.csv`를 찾을 수 없는 경우

`run.py`와 같은 폴더에 `historical_results.csv`가 있어야 합니다.

```bash
ls historical_results.csv
```

Windows PowerShell:

```powershell
Test-Path .\historical_results.csv
```

### pickle 모델을 찾을 수 없는 경우

아래 두 파일이 repository 루트에 있는지 확인하세요.

```bash
ls worldcup_home_model.pkl worldcup_away_model.pkl
```

### 2026 월드컵 예측 대상 경기가 없다는 오류가 나는 경우

`historical_results.csv` 안에 다음 조건을 만족하는 행이 있어야 합니다.

```text
date year == 2026
tournament == "FIFA World Cup"
```

### macOS에서 LightGBM import 오류가 나는 경우

Homebrew가 설치되어 있다면 OpenMP 런타임을 설치한 뒤 LightGBM을 다시 설치해 보세요.

```bash
brew install libomp
pip install --force-reinstall lightgbm
```

## GitHub Upload Checklist

GitHub에 올릴 때는 실행에 필요한 핵심 파일이 포함되어 있는지 확인하세요.

```bash
git status
git add README.md run.py lib.py historical_results.csv third_place_assignments_2026.csv worldcup_home_model.pkl worldcup_away_model.pkl geminifootpredict.ipynb submission_from_pickle.csv submission_from_pickle_full_tournament.csv
git commit -m "Add World Cup predictor inference pipeline"
git push origin main
```

`worldcup_home_model.pkl`과 `worldcup_away_model.pkl`은 용량이 큰 파일입니다. GitHub 업로드 제한에 걸리면 Git LFS 사용을 고려하세요.
