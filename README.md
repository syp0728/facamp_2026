# 📋 2026 World Cup Bracket Predictor

과거 약 5만 건의 A매치 빅데이터를 기반으로 **Dynamic Elo Rating** 및 최근 경기력 스탯(Form)을 추출하고, **LightGBM (Poisson Regression)** 모델과 **10,000회 몬테카를로 시뮬레이션**을 결합하여 2026 월드컵 결과를 예측하는 프로젝트입니다.

본 리포지토리는 평가 환경에서의 완벽한 **결과 재현(Reproducibility)**과 빠른 검증을 최우선으로 설계되었습니다. 하이퍼파라미터 튜닝이 완료된 모델 가중치 및 빌드업된 Elo 레이팅 시점 데이터가 `pickle` 형태로 사전에 빌드되어 동봉되어 있습니다.

---

## 🛠️ 핵심 특징 (Key Features)

* **완벽한 결과 재현성:** 데이터 무작위성(`Seed=9`) 및 알고리즘 연산(`Deterministic` 옵션)이 모두 통제되어 있어, 실행 시 언제나 동일한 예측 결과 스코어가 도출됩니다.
* **하이브리드 예측 모델:** * **Dynamic Elo Rating:** 국가대표팀의 역사적 체급 및 전력 반영
  * **Recent Form 스탯:** 월드컵 직전 최근 경기력 흐름 반영
  * **LightGBM (Poisson Regression):** 국가별 경기당 득점 수 확률 예측
  * **Monte Carlo Simulation:** 10,000회 반복 연산을 통한 토너먼트 대진 및 우승 확률 산출

---

## 🚀 시작하기 (Quick Start)

### 1. 환경 구성 및 의존성 설치
본 프로젝트는 아래 환경에서 테스트 및 검증되었습니다. 의존성 라이브러리를 설치해 주세요.

```bash
pip install lightgbm pandas numpy scikit-learn