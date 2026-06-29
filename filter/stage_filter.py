"""
filter/stage_filter.py
───────────────────────
Weinstein Stage 1~4 판별
입력: {ticker: weekly_df with indicators}
출력: {ticker: weekly_df + stage 컬럼}

Stage 정의:
    1 = 횡보  (MA 평탄, ADX 낮음)
    2 = 상승  (Price > MA, MA 상승, ADX 상승)
    3 = 천장  (Price > MA 이지만 MA 기울기 꺾임)
    4 = 하락  (Price < MA, MA 하락)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

PARAMS_DIR = Path(__file__).parent.parent / "params"
PARAMS_FILE = PARAMS_DIR / "stage2_params.json"

# ── 기본 파라미터 (params 파일 없을 때 fallback) ──
DEFAULT_PARAMS = {
    "ma_period"     : 30,
    "adx_threshold" : 20,
    "ma_slope_weeks": 4,
    "vol_mult"      : 1.2,
    "hi52_pct"      : 0.02,   # 52주 고가 대비 -2% 이내
}


def load_params(ticker: str) -> dict:
    """params/stage2_params.json 에서 종목별 파라미터 로딩"""
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE) as f:
            all_params = json.load(f)
        return all_params.get(ticker, DEFAULT_PARAMS)
    return DEFAULT_PARAMS


def classify_stage(row: pd.Series, p: dict) -> int:
    """
    단일 행(최신 주봉)의 Stage 판별
    Returns: 1, 2, 3, 4
    """
    price     = row.get("close", 0)
    ma        = row.get("MA30", 0)
    slope     = row.get("MA30_slope", 0)
    adx       = row.get("ADX14", 0)

    if pd.isna(ma) or ma == 0:
        return 0   # 계산 불가

    above_ma  = price > ma
    slope_up  = slope > 0
    adx_trend = adx >= p["adx_threshold"]

    if above_ma and slope_up and adx_trend:
        return 2   # 상승추세
    elif above_ma and slope_up and not adx_trend:
        return 1   # 횡보 (MA 위지만 ADX 약함)
    elif above_ma and not slope_up:
        return 3   # 천장 (MA 위지만 꺾임)
    else:
        return 4   # 하락


def run_stage_filter(
    all_dfs: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    전체 자산 Stage 판별
    각 df 의 모든 행에 stage 컬럼 추가 (백테스트용)
    최신 행의 stage_current 도 별도 기록
    """
    print(f"\n[Stage 판별] {len(all_dfs)}개 자산")
    result = {}

    stage_summary = {1: 0, 2: 0, 3: 0, 4: 0, 0: 0}

    for ticker, df in all_dfs.items():
        p = load_params(ticker)
        d = df.copy()

        # 전체 기간 Stage (백테스트용)
        def _row_stage(row):
            return classify_stage(row, p)

        d["stage"] = d.apply(_row_stage, axis=1)

        # 현재 Stage (최신 행)
        current_stage = int(d["stage"].iloc[-1])
        d["stage_current"] = current_stage
        stage_summary[current_stage] = stage_summary.get(current_stage, 0) + 1

        result[ticker] = d

    # 요약 출력
    print(f"  Stage 1 (횡보): {stage_summary[1]}개")
    print(f"  Stage 2 (상승): {stage_summary[2]}개  ← 투자 대상")
    print(f"  Stage 3 (천장): {stage_summary[3]}개")
    print(f"  Stage 4 (하락): {stage_summary[4]}개")

    return result
