"""
===================================================================
  🏆 ULTIMATE World Cup Predictor
     Deep Stacking (LGB + Cat + XGB) + Dixon-Coles Monte Carlo
     + 34-Feature Advanced Engineering
===================================================================
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from scipy.stats import poisson
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
SEED = 9
N_FOLDS = 5

# =====================================================================
# 1. TOURNAMENT WEIGHT
# =====================================================================
def get_tournament_weight(tournament):
    t = str(tournament).strip()
    tl = t.lower()
    if t == "FIFA World Cup":            return 2.0
    if t == "FIFA World Cup qualification": return 1.3
    if t in ["UEFA Euro", "Copa América", "African Cup of Nations",
             "AFC Asian Cup", "Gold Cup", "CONCACAF Championship",
             "Oceania Nations Cup", "Confederations Cup"]:
        return 1.5
    if "qualification" in tl:           return 1.15
    if "nations league" in tl:          return 1.1
    if t == "Friendly":                 return 0.7
    return 1.0

# =====================================================================
# 2. DUAL ELO  (공격 Elo + 수비 Elo 분리)
# =====================================================================
class DualElo:
    """
    공격/수비 Elo를 분리하여 계산.
    - attack_elo  : 득점 기반 공격력
    - defense_elo : 실점 기반 수비력 (높을수록 '안 먹힘')
    """
    def __init__(self, k=20, base=1500, home_adv=100):
        self.k = k
        self.base = base
        self.home_adv = home_adv
        self.attack  = {}   # 공격 Elo
        self.defense = {}   # 수비 Elo
        self.overall = {}   # 통합 Elo

    def get(self, d, team): return d.get(team, self.base)

    def _exp(self, ra, rb): return 1 / (10 ** (-(ra - rb) / 400) + 1)

    def update(self, ht, at, hg, ag, neutral=False, tw=1.0):
        ha = 0 if neutral else self.home_adv

        # ---------- overall Elo ----------
        re_h = self._exp(self.get(self.overall, ht) + ha, self.get(self.overall, at))
        re_a = 1 - re_h
        gd = abs(hg - ag)
        G = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
        act_h = (1.0 if hg > ag else 0.0 if hg < ag else 0.5)
        self.overall[ht] = self.get(self.overall, ht) + self.k * G * tw * (act_h - re_h)
        self.overall[at] = self.get(self.overall, at) + self.k * G * tw * ((1 - act_h) - re_a)

        # ---------- attack Elo (득점 능력) ----------
        re_atk = self._exp(self.get(self.attack, ht) + ha, self.get(self.defense, at))
        self.attack[ht]  = self.get(self.attack, ht)  + self.k * tw * (act_h - re_atk)
        self.attack[at]  = self.get(self.attack, at)  + self.k * tw * ((1 - act_h) - (1 - re_atk))

        # ---------- defense Elo (수비 능력, 높을수록 좋음) ----------
        re_def = self._exp(self.get(self.defense, ht) + ha, self.get(self.attack, at))
        self.defense[ht] = self.get(self.defense, ht) + self.k * tw * (act_h - re_def)
        self.defense[at] = self.get(self.defense, at) + self.k * tw * ((1 - act_h) - (1 - re_def))

# =====================================================================
# 3. ULTRA FEATURE ENGINEERING (34 features)
# =====================================================================
def compute_features(df_sorted):
    """
    df_sorted : date 순 정렬된 전체 경기 데이터
    Returns   : df_sorted + 34개 피처 컬럼 추가
    """
    EMA_SPAN = 10          # 지수이동평균 스팬
    FORM_WINDOW = 5        # 폼 집계 최근 N경기
    HIST_WINDOW = 10       # 평균 집계 최근 N경기
    H2H_WINDOW = 5         # H2H 최근 N경기

    elo_sys = DualElo()
    team_hist = {}         # {team: {scored, conceded, pts, home_scored, home_conceded, away_scored, away_conceded}}
    h2h_hist  = {}         # {(t1,t2): list of (h_goals, a_goals)}

    rows = []
    for _, row in tqdm(df_sorted.iterrows(), total=len(df_sorted), desc="🔧 피처 계산"):
        ht, at = row['home_team'], row['away_team']
        neutral = bool(row['is_neutral'])
        tw      = row['tournament_weight']
        date    = row['date']

        # --- 초기화 ---
        for t in (ht, at):
            if t not in team_hist:
                team_hist[t] = dict(
                    scored=[], conceded=[], pts=[],
                    home_scored=[], home_conceded=[],
                    away_scored=[], away_conceded=[],
                    major_pts=[], dates=[]
                )

        def h(t):  return team_hist[t]
        def safe_mean(lst, n): return float(np.mean(lst[-n:])) if lst else 1.0
        def safe_sum(lst, n):  return float(sum(lst[-n:]))     if lst else 0.0

        # --- 1. Elo 피처 (경기 전 값) ---
        ho  = elo_sys.get(elo_sys.overall, ht)
        ao  = elo_sys.get(elo_sys.overall, at)
        hat = elo_sys.get(elo_sys.attack,  ht)
        hdf = elo_sys.get(elo_sys.defense, ht)
        aat = elo_sys.get(elo_sys.attack,  at)
        adf = elo_sys.get(elo_sys.defense, at)

        # --- 2. 공격/수비 평균 피처 ---
        h_avg_s  = safe_mean(h(ht)['scored'],   HIST_WINDOW)
        h_avg_c  = safe_mean(h(ht)['conceded'], HIST_WINDOW)
        a_avg_s  = safe_mean(h(at)['scored'],   HIST_WINDOW)
        a_avg_c  = safe_mean(h(at)['conceded'], HIST_WINDOW)

        # --- 3. EMA 피처 ---
        def ema(lst, span=EMA_SPAN):
            if not lst: return 1.0
            s = pd.Series(lst[-span*2:])
            return float(s.ewm(span=span, adjust=False).mean().iloc[-1])

        h_ema_s = ema(h(ht)['scored'])
        h_ema_c = ema(h(ht)['conceded'])
        a_ema_s = ema(h(at)['scored'])
        a_ema_c = ema(h(at)['conceded'])

        # --- 4. 홈/원정 분리 스탯 ---
        hh_s = safe_mean(h(ht)['home_scored'],    HIST_WINDOW)
        hh_c = safe_mean(h(ht)['home_conceded'],  HIST_WINDOW)
        aa_s = safe_mean(h(at)['away_scored'],    HIST_WINDOW)
        aa_c = safe_mean(h(at)['away_conceded'],  HIST_WINDOW)

        # --- 5. 폼 피처 ---
        h_form = safe_sum(h(ht)['pts'], FORM_WINDOW)
        a_form = safe_sum(h(at)['pts'], FORM_WINDOW)

        # --- 6. 주요 대회 폼 ---
        h_major = safe_sum(h(ht)['major_pts'], FORM_WINDOW)
        a_major = safe_sum(h(at)['major_pts'], FORM_WINDOW)

        # --- 7. H2H 피처 ---
        key = tuple(sorted([ht, at]))
        h2h = h2h_hist.get(key, [])
        # h2h 항목: (home, away, h_goals, a_goals)
        recent_h2h = [e for e in h2h[-H2H_WINDOW:]]
        if recent_h2h:
            # ht 기준 득실 계산
            h2h_h_goals = [e[2] if e[0]==ht else e[3] for e in recent_h2h]
            h2h_a_goals = [e[3] if e[0]==ht else e[2] for e in recent_h2h]
            h2h_wins    = sum(1 for g1,g2 in zip(h2h_h_goals,h2h_a_goals) if g1>g2)
            h2h_h_avg   = float(np.mean(h2h_h_goals))
            h2h_a_avg   = float(np.mean(h2h_a_goals))
            h2h_win_rate= h2h_wins / len(recent_h2h)
        else:
            h2h_h_avg = h2h_a_avg = 1.0
            h2h_win_rate = 0.5

        # --- 8. 경기 간격 ---
        last_date = h(ht)['dates'][-1] if h(ht)['dates'] else date
        days_gap  = max(0, (date - last_date).days)

        feat = dict(
            # Elo
            home_elo=ho, away_elo=ao, elo_diff=ho-ao,
            home_attack_elo=hat, home_defense_elo=hdf,
            away_attack_elo=aat, away_defense_elo=adf,
            atk_def_diff=(hat+adf) - (aat+hdf),   # 홈팀 공격+수비 종합 우위
            # 평균
            home_avg_scored=h_avg_s, home_avg_conceded=h_avg_c,
            away_avg_scored=a_avg_s, away_avg_conceded=a_avg_c,
            # EMA
            home_ema_scored=h_ema_s, home_ema_conceded=h_ema_c,
            away_ema_scored=a_ema_s, away_ema_conceded=a_ema_c,
            # 홈/원정 분리
            home_home_attack=hh_s,  home_home_defense=hh_c,
            away_away_attack=aa_s,  away_away_defense=aa_c,
            # 폼
            home_form=h_form, away_form=a_form,
            home_major_form=h_major, away_major_form=a_major,
            # H2H
            h2h_home_avg=h2h_h_avg, h2h_away_avg=h2h_a_avg,
            h2h_home_win_rate=h2h_win_rate,
            # 기타
            days_gap=min(days_gap, 365),
        )
        rows.append(feat)

        # --- 경기 후 업데이트 ---
        if pd.notna(row.get('home_score')) and pd.notna(row.get('away_score')):
            hs, as_ = float(row['home_score']), float(row['away_score'])
            elo_sys.update(ht, at, hs, as_, neutral=neutral, tw=tw)

            for t, sc, co, is_home in [(ht, hs, as_, True), (at, as_, hs, False)]:
                h(t)['scored'].append(sc)
                h(t)['conceded'].append(co)
                h(t)['dates'].append(date)
                if sc > co:   h(t)['pts'].append(3)
                elif sc < co: h(t)['pts'].append(0)
                else:         h(t)['pts'].append(1)

                if is_home:
                    h(t)['home_scored'].append(sc)
                    h(t)['home_conceded'].append(co)
                else:
                    h(t)['away_scored'].append(sc)
                    h(t)['away_conceded'].append(co)

                if tw >= 1.5:   # 주요 대회
                    h(t)['major_pts'].append(3 if sc>co else 0 if sc<co else 1)

            if key not in h2h_hist: h2h_hist[key] = []
            h2h_hist[key].append((ht, at, hs, as_))

    feat_df = pd.DataFrame(rows, index=df_sorted.index)
    return pd.concat([df_sorted, feat_df], axis=1)


# =====================================================================
# 4. DEEP STACKING  (5-Fold OOF → Ridge Meta)
# =====================================================================
FEATURES = [
    'home_elo','away_elo','elo_diff',
    'home_attack_elo','home_defense_elo',
    'away_attack_elo','away_defense_elo',
    'atk_def_diff',
    'home_avg_scored','home_avg_conceded',
    'away_avg_scored','away_avg_conceded',
    'home_ema_scored','home_ema_conceded',
    'away_ema_scored','away_ema_conceded',
    'home_home_attack','home_home_defense',
    'away_away_attack','away_away_defense',
    'home_form','away_form',
    'home_major_form','away_major_form',
    'h2h_home_avg','h2h_away_avg','h2h_home_win_rate',
    'is_neutral','days_gap',
]

def _lgb_model():
    return lgb.LGBMRegressor(
        objective='poisson', n_estimators=2000, learning_rate=0.02,
        num_leaves=63, min_child_samples=15, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, verbose=-1,
        deterministic=True, force_col_wise=True
    )

def _cat_model():
    return CatBoostRegressor(
        loss_function='Poisson', iterations=2000, learning_rate=0.02,
        depth=6, l2_leaf_reg=3.0, random_seed=SEED, verbose=False,
        early_stopping_rounds=100
    )

def _xgb_model():
    return xgb.XGBRegressor(
        objective='count:poisson', n_estimators=2000, learning_rate=0.02,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, early_stopping_rounds=100,
        eval_metric='poisson-nloglik', tree_method='hist', device='cuda'  # GPU
    )

def stacking_predict(X_tr, y_tr, X_te, label=""):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X_tr), 3))
    te_preds = np.zeros((len(X_te), 3))

    scaler = RobustScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    X_te_s = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns, index=X_te.index)

    lgb_models, cat_models, xgb_models = [], [], []

    for fold, (ti, vi) in enumerate(kf.split(X_tr_s)):
        print(f"  [{label}] Fold {fold+1}/{N_FOLDS}")
        Xf_tr, Xf_va = X_tr_s.iloc[ti], X_tr_s.iloc[vi]
        yf_tr, yf_va = y_tr.iloc[ti], y_tr.iloc[vi]

        # LightGBM
        m_lgb = _lgb_model()
        m_lgb.fit(Xf_tr, yf_tr,
                  eval_set=[(Xf_va, yf_va)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[vi, 0] = m_lgb.predict(Xf_va)
        te_preds[:, 0] += m_lgb.predict(X_te_s) / N_FOLDS
        lgb_models.append(m_lgb)

        # CatBoost
        m_cat = _cat_model()
        m_cat.fit(Xf_tr, yf_tr, eval_set=(Xf_va, yf_va), use_best_model=True)
        oof[vi, 1] = m_cat.predict(Xf_va)
        te_preds[:, 1] += m_cat.predict(X_te_s) / N_FOLDS
        cat_models.append(m_cat)

        # XGBoost (GPU)
        m_xgb = _xgb_model()
        m_xgb.fit(Xf_tr, yf_tr,
                  eval_set=[(Xf_va, yf_va)],
                  verbose=False)
        oof[vi, 2] = m_xgb.predict(Xf_va)
        te_preds[:, 2] += m_xgb.predict(X_te_s) / N_FOLDS
        xgb_models.append(m_xgb)

    # Meta Model (Ridge, positive=True → 기대득점 음수 방지)
    print(f"  [{label}] Meta Ridge 최적화...")
    meta = Ridge(alpha=1.0, positive=True)
    meta.fit(oof, y_tr)
    final = meta.predict(te_preds)
    final = np.clip(final, 0.05, 12.0)
    print(f"  [{label}] 메타 가중치 [LGB|Cat|XGB]: {np.round(meta.coef_, 3)}")

    import joblib
    joblib.dump({
        'scaler': scaler,
        'lgb_models': lgb_models,
        'cat_models': cat_models,
        'xgb_models': xgb_models,
        'meta_model': meta
    }, f"worldcup_{label.lower()}_model.pkl")
    print(f"  [{label}] 모델 피클 저장 완료 (worldcup_{label.lower()}_model.pkl)")

    return final


# =====================================================================
# 5. DIXON-COLES MONTE CARLO SIMULATION
# =====================================================================
def dc_simulate(lh, la, rho=-0.13, n=10000, max_g=12):
    """
    Dixon-Coles 보정 결합확률 매트릭스 → numpy 기반 초고속 시뮬레이션
    rho = -0.13 : 저득점 무승부를 현실적으로 보정하는 음의 상관관계
    """
    goals = np.arange(max_g)
    joint = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))

    # Dixon-Coles τ 보정 (0-0, 1-0, 0-1, 1-1)
    for hg, ag, tau in [
        (0, 0, 1 - lh * la * rho),
        (1, 0, 1 + la * rho),
        (0, 1, 1 + lh * rho),
        (1, 1, 1 - rho),
    ]:
        if hg < max_g and ag < max_g:
            joint[hg, ag] *= max(tau, 0.0)

    joint /= joint.sum()   # 정규화

    flat = joint.flatten()
    idx  = np.random.choice(len(flat), size=n, p=flat)
    hg_sim = idx // max_g
    ag_sim = idx %  max_g

    ph = np.mean(hg_sim > ag_sim)
    pd_ = np.mean(hg_sim == ag_sim)
    pa = np.mean(hg_sim < ag_sim)

    scores = [f"{h}-{a}" for h, a in zip(hg_sim, ag_sim)]
    best = max(set(scores), key=scores.count)
    pred_h, pred_a = map(int, best.split('-'))
    return ph, pd_, pa, pred_h, pred_a


def predict_all(df_test, n_sim=10000):
    results = []
    print(f"\n🎲 Dixon-Coles MC 시뮬레이션 ({n_sim:,}회 / 경기)")
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        ph, pd_, pa, sh, sa = dc_simulate(row['lambda_home'], row['lambda_away'], n=n_sim)

        # 무승부 → 비례 분배 (우리 코드 전통 룰)
        if ph + pa > 0:
            p1 = ph + pd_ * ph / (ph + pa)
            p2 = pa + pd_ * pa / (ph + pa)
        else:
            p1 = p2 = 0.5

        # 스코어 타이브레이커
        if sh == sa:
            if p1 > p2: sh += 1
            elif p2 > p1: sa += 1

        mt = str(row.get('type', 'group'))
        if mt.strip() == '' or mt == 'nan': mt = 'group'
        results.append(dict(
            team1=row['home_team'], team2=row['away_team'],
            team1_score=int(sh), team2_score=int(sa),
            team1_prob=round(p1, 4), team2_prob=round(p2, 4),
            type=mt
        ))
    return pd.DataFrame(results)


def format_submission(df):
    df = df.copy()
    df[["team1","team2"]] = df[["team1","team2"]].replace({
        "Cura?ao":"Curaçao","Curacao":"Curaçao",
        "DR Congo":"Congo DR","Cape Verde":"Cape Verde Islands",
        "Czech Republic":"Czechia","Bosnia and Herzegovina":"Bosnia-Herzegovina"
    })
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip().str.lower().replace(
            {"group":"Group Stage","group stage":"Group Stage"})
    df["team1_score"] = pd.to_numeric(df["team1_score"],errors="coerce").fillna(0).astype(int)
    df["team2_score"] = pd.to_numeric(df["team2_score"],errors="coerce").fillna(0).astype(int)
    df["team1_prob"] = pd.to_numeric(df["team1_prob"],errors="coerce").fillna(0.5)
    df["team2_prob"] = pd.to_numeric(df["team2_prob"],errors="coerce").fillna(0.5)
    s = df["team1_prob"] + df["team2_prob"]
    s[s<=0] = 1.0
    df["team1_prob"] = (df["team1_prob"]/s).round(4)
    df["team2_prob"] = (1-df["team1_prob"]).round(4)
    return df[["team1","team2","team1_score","team2_score","team1_prob","team2_prob","type"]]


# =====================================================================
# 6. MAIN PIPELINE
# =====================================================================
if __name__ == "__main__":
    np.random.seed(SEED)
    print("=" * 60)
    print("  🏆 ULTIMATE World Cup Predictor")
    print("  Deep Stacking (LGB+Cat+XGB) × Dixon-Coles")
    print("  34-Feature Advanced Engineering")
    print("=" * 60)

    # --- 데이터 로드 ---
    for fp in ["../data/historical_results.csv",
               "/kaggle/input/datasets/dybalar/feature/historical_results.csv"]:
        if os.path.exists(fp):
            file_path = fp; break

    df = pd.read_csv(file_path, encoding_errors='replace')
    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['is_neutral'] = (
        df['neutral'].astype(str).str.strip().str.upper()
        .isin(["TRUE","1","YES","Y"]).astype(int)
    )
    df['tournament_weight'] = df['tournament'].apply(get_tournament_weight)

    # --- 피처 계산 ---
    print("\n📐 34개 피처 계산 시작...")
    df = compute_features(df)

    # --- 분리 ---
    mask_wc = (df['date'].dt.year == 2026) & (df['tournament'] == 'FIFA World Cup')
    df_train = df[~mask_wc].dropna(subset=['home_score','away_score']).copy()
    df_test  = df[mask_wc].copy()

    # NaN 처리 (아직 통계 없는 초기 경기)
    df_train[FEATURES] = df_train[FEATURES].fillna(1.0)
    df_test[FEATURES]  = df_test[FEATURES].fillna(1.0)

    print(f"✅ 학습: {len(df_train):,}건 / 예측: {len(df_test):,}건")

    X_tr = df_train[FEATURES]
    X_te = df_test[FEATURES]

    # --- Deep Stacking: 홈 득점 ---
    print("\n" + "="*60)
    print("🔥 STEP 1 / 2 : 홈팀 기대득점 스태킹")
    print("="*60)
    lambda_home = stacking_predict(X_tr, df_train['home_score'], X_te, "HOME")

    # --- Deep Stacking: 원정 득점 ---
    print("\n" + "="*60)
    print("🔥 STEP 2 / 2 : 원정팀 기대득점 스태킹")
    print("="*60)
    lambda_away = stacking_predict(X_tr, df_train['away_score'], X_te, "AWAY")

    df_test = df_test.copy()
    df_test['lambda_home'] = lambda_home
    df_test['lambda_away'] = lambda_away

    # --- Dixon-Coles Monte Carlo ---
    print("\n" + "="*60)
    print("🎲 STEP 3 / 3 : Dixon-Coles 몬테카를로 시뮬레이션")
    print("="*60)
    df_pred = predict_all(df_test, n_sim=10000)
    df_pred = format_submission(df_pred)

    out = "submission_consensus.csv"
    df_pred.to_csv(out, index=False)

    print("\n" + "="*60)
    print(f"🎉  예측 완료! → '{out}'")
    print("="*60)
    print(df_pred.head(10))

# =====================================================================
# 7. KNOCKOUT BRACKET ALGORITHM (Round of 32)
# =====================================================================
def build_bracket_from_groups(df_pred, third_place_csv):
    from collections import defaultdict
    teams = set(df_pred['team1']).union(set(df_pred['team2']))
    
    stats = {t: {'pts':0, 'gd':0, 'gf':0} for t in teams}
    groups_graph = defaultdict(set)
    for _, row in df_pred.iterrows():
        t1, t2 = row['team1'], row['team2']
        groups_graph[t1].add(t2)
        groups_graph[t2].add(t1)
        
        s1, s2 = row['team1_score'], row['team2_score']
        stats[t1]['gf'] += s1
        stats[t2]['gf'] += s2
        stats[t1]['gd'] += (s1 - s2)
        stats[t2]['gd'] += (s2 - s1)
        if s1 > s2: stats[t1]['pts'] += 3
        elif s1 < s2: stats[t2]['pts'] += 3
        else:
            stats[t1]['pts'] += 1
            stats[t2]['pts'] += 1
            
    visited = set()
    groups = {}
    group_names = "ABCDEFGHIJKL"
    g_idx = 0
    for t in sorted(teams):
        if t not in visited:
            group_members = set([t])
            queue = [t]
            while queue:
                curr = queue.pop(0)
                for neighbor in groups_graph[curr]:
                    if neighbor not in group_members:
                        group_members.add(neighbor)
                        queue.append(neighbor)
            for m in group_members: visited.add(m)
            g_name = group_names[g_idx]
            g_idx += 1
            
            sorted_m = sorted(list(group_members), key=lambda x: (stats[x]['pts'], stats[x]['gd'], stats[x]['gf']), reverse=True)
            groups[g_name] = sorted_m
            
    firsts = {g: groups[g][0] for g in groups}
    seconds = {g: groups[g][1] for g in groups}
    thirds = {g: groups[g][2] for g in groups}
    
    sorted_thirds = sorted(thirds.items(), key=lambda x: (stats[x[1]]['pts'], stats[x[1]]['gd'], stats[x[1]]['gf']), reverse=True)
    best_8_thirds = sorted_thirds[:8]
    best_8_groups = tuple(sorted([x[0] for x in best_8_thirds]))
    
    import csv, os
    rules = {}
    slots = ("1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L")
    if os.path.exists(third_place_csv):
        with open(third_place_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assignment = {s: row[s] for s in slots}
                key = tuple(sorted(v[1:] for v in assignment.values()))
                rules[key] = assignment
                
    bracket_matches = []
    if best_8_groups in rules:
        assignment = rules[best_8_groups]
        for slot, opp in assignment.items():
            g1, g3 = slot[1], opp[1]
            t1 = firsts[g1]
            t3 = thirds[g3]
            bracket_matches.append((t1, t3))
            
    remaining = [
        (firsts.get('C'), seconds.get('B')), (firsts.get('F'), seconds.get('A')),
        (firsts.get('H'), seconds.get('G')), (firsts.get('J'), seconds.get('I')),
        (seconds.get('C'), seconds.get('D')), (seconds.get('E'), seconds.get('F')),
        (seconds.get('H'), seconds.get('K')), (seconds.get('J'), seconds.get('L'))
    ]
    for m in remaining:
        if m[0] and m[1]: bracket_matches.append(m)
        
    return bracket_matches

def simulate_full_knockouts(r32_matches, df_test):
    import joblib
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        home_model = joblib.load("worldcup_home_model.pkl")
        away_model = joblib.load("worldcup_away_model.pkl")
    except Exception as e:
        print("모델 로드 실패! 먼저 학습을 실행해 피클 파일을 생성하세요.", e)
        return
        
    def get_latest_features(team, df_all):
        last_idx = df_all[(df_all['home_team'] == team) | (df_all['away_team'] == team)].last_valid_index()
        if last_idx is None:
            # Fallback
            return {k: 1.0 for k in ['elo','attack_elo','defense_elo','avg_scored','avg_conceded','ema_scored','ema_conceded','home_attack','home_defense','form','major_form']}
        row = df_all.loc[last_idx]
        if row['home_team'] == team:
            return {
                'elo': row['home_elo'], 'attack_elo': row['home_attack_elo'], 'defense_elo': row['home_defense_elo'],
                'avg_scored': row['home_avg_scored'], 'avg_conceded': row['home_avg_conceded'],
                'ema_scored': row['home_ema_scored'], 'ema_conceded': row['home_ema_conceded'],
                'home_attack': row['home_home_attack'], 'home_defense': row['home_home_defense'],
                'form': row['home_form'], 'major_form': row['home_major_form']
            }
        else:
            return {
                'elo': row['away_elo'], 'attack_elo': row['away_attack_elo'], 'defense_elo': row['away_defense_elo'],
                'avg_scored': row['away_avg_scored'], 'avg_conceded': row['away_avg_conceded'],
                'ema_scored': row['away_ema_scored'], 'ema_conceded': row['away_ema_conceded'],
                'home_attack': row['away_away_attack'], 'home_defense': row['away_away_defense'],
                'form': row['away_form'], 'major_form': row['away_major_form']
            }

    def predict_match(t1, t2):
        f1 = get_latest_features(t1, df_test)
        f2 = get_latest_features(t2, df_test)
        
        feat = {
            'home_elo': f1['elo'], 'away_elo': f2['elo'], 'elo_diff': f1['elo'] - f2['elo'],
            'home_attack_elo': f1['attack_elo'], 'home_defense_elo': f1['defense_elo'],
            'away_attack_elo': f2['attack_elo'], 'away_defense_elo': f2['defense_elo'],
            'atk_def_diff': (f1['attack_elo'] + f2['defense_elo']) - (f2['attack_elo'] + f1['defense_elo']),
            'home_avg_scored': f1['avg_scored'], 'home_avg_conceded': f1['avg_conceded'],
            'away_avg_scored': f2['avg_scored'], 'away_avg_conceded': f2['avg_conceded'],
            'home_ema_scored': f1['ema_scored'], 'home_ema_conceded': f1['ema_conceded'],
            'away_ema_scored': f2['ema_scored'], 'away_ema_conceded': f2['ema_conceded'],
            'home_home_attack': f1['home_attack'], 'home_home_defense': f1['home_defense'],
            'away_away_attack': f2['home_attack'], 'away_away_defense': f2['home_defense'],
            'home_form': f1['form'], 'away_form': f2['form'],
            'home_major_form': f1['major_form'], 'away_major_form': f2['major_form'],
            'h2h_home_avg': 1.0, 'h2h_away_avg': 1.0, 'h2h_home_win_rate': 0.5,
            'is_neutral': 1, 'days_gap': 5
        }
        
        df_f = pd.DataFrame([feat], columns=FEATURES)
        
        def ensemble(X, model_dict):
            s = model_dict['scaler']
            X_s = pd.DataFrame(s.transform(X), columns=X.columns)
            preds = np.zeros((1, 3))
            for m in model_dict['lgb_models']: preds[:,0] += m.predict(X_s) / len(model_dict['lgb_models'])
            for m in model_dict['cat_models']: preds[:,1] += m.predict(X_s) / len(model_dict['cat_models'])
            for m in model_dict['xgb_models']: preds[:,2] += m.predict(X_s) / len(model_dict['xgb_models'])
            final = model_dict['meta_model'].predict(preds)
            return np.clip(final, 0.05, 12.0)[0]
            
        lh = ensemble(df_f, home_model)
        la = ensemble(df_f, away_model)
        
        ph, pd_, pa, sh, sa = dc_simulate(lh, la, n=10000)
        
        if ph + pa > 0:
            p1 = ph + pd_ * ph / (ph + pa)
            p2 = pa + pd_ * pa / (ph + pa)
        else:
            p1 = p2 = 0.5

        if sh == sa:
            if p1 > p2: 
                sh += 1
                winner = t1
            else: 
                sa += 1
                winner = t2
        else:
            winner = t1 if sh > sa else t2
            
        return winner, int(sh), int(sa), p1, p2

    rounds = [r32_matches]
    round_names = ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals", "Final"]
    knockout_results = []
    
    rnd_idx = 0
    while rnd_idx < len(rounds):
        current_matches = rounds[rnd_idx]
        match_type = round_names[rnd_idx] if rnd_idx < len(round_names) else "Knockout"
        print("\n" + "="*50)
        print(f" 🏆 {match_type} 진행 중...")
        print("="*50)
        
        winners = []
        for i, (t1, t2) in enumerate(current_matches):
            winner, sh, sa, p1, p2 = predict_match(t1, t2)
            print(f"  Match {i+1}: {t1} vs {t2} => {winner} 승리! ({sh}:{sa})")
            winners.append(winner)
            knockout_results.append({
                'team1': t1, 'team2': t2,
                'team1_score': sh, 'team2_score': sa,
                'team1_prob': round(p1, 4), 'team2_prob': round(p2, 4),
                'type': match_type
            })
            
        if len(winners) > 1:
            next_matches = [(winners[i], winners[i+1]) for i in range(0, len(winners), 2)]
            rounds.append(next_matches)
        else:
            print("\n" + "🌟"*25)
            print(f" 🏆 2026 WORLD CUP CHAMPION: {winners[0]}!!! 🏆")
            print("🌟"*25)
        rnd_idx += 1
        
    return pd.DataFrame(knockout_results)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏆 32강 넉아웃 대진표 매치업 생성")
    print("="*60)
    # csv 경로는 실제 환경에 맞춰 조정
    third_csv = None
    for fp in [r"c:\Users\passp\Desktop\coding\facamp\data\third_place_assignments_2026.csv",
               "../data/third_place_assignments_2026.csv",
               "/kaggle/input/datasets/dybalar/feature/third_place_assignments_2026.csv"]:
        if os.path.exists(fp):
            third_csv = fp; break
            
    if third_csv:
        try:
            r32_matches = build_bracket_from_groups(df_pred, third_csv)
            df_knockouts = simulate_full_knockouts(r32_matches, df_test)
            
            if df_knockouts is not None and not df_knockouts.empty:
                # 조별리그 + 결승까지 전체 결과 통합 저장
                df_final = pd.concat([df_pred, df_knockouts], ignore_index=True)
                out_final = "submission_consensus_full_tournament.csv"
                df_final.to_csv(out_final, index=False)
                print("\n" + "="*60)
                print(f"💾 결승전까지 싹 다 포함된 최종 파일 저장 완료: '{out_final}'")
                print("="*60)
        except Exception as e:
            print("대진표 생성 실패:", e)
    else:
        print("CSV 파일을 찾을 수 없습니다.")
