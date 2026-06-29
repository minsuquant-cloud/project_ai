"""
filter/stage2_entry.py
───────────────────────
Stage 2 진입 세부 조건 확인
Stage 판별에서 2로 분류된 자산 중
추가 진입 조건을 통과한 자산만 선별

진입 조건 (모두 충족해야 통과):
    1. Price > MA30
    2. MA30 상승 (기울기 > 0)
    3. ADX >= threshold
    4. 거래량 > 최근 평균 × vol_mult
    5. 현재가 >= 52주 고가 × (1 - hi52_pct)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

PARAMS_DIR  = Path(__file__).parent.parent / "params"
PARAMS_FILE = PARAMS_DIR / "stage2_params.json"

DEFAULT_PARAMS = {
    "ma_period"     : 30,
    "adx_threshold" : 20,
    "ma_slope_weeks": 4,
    "vol_mult"      : 1.2,
    "hi52_pct"      : 0.02,
}


def load_params(ticker: str) -> dict:
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE) as f:
            all_params = json.load(f)
        return all_params.get(ticker, DEFAULT_PARAMS)
    return DEFAULT_PARAMS


def check_entry(df: pd.DataFrame, ticker: str) -> dict:
    """
    최신 행 기준 Stage 2 진입 조건 체크
    Returns: 조건별 pass/fail 딕셔너리
    """
    p      = load_params(ticker)
    latest = df.iloc[-1]

    price     = float(latest.get("close", 0))
    ma        = float(latest.get("MA30", 0))       if not pd.isna(latest.get("MA30", np.nan))       else 0
    slope     = float(latest.get("MA30_slope", 0)) if not pd.isna(latest.get("MA30_slope", np.nan)) else 0
    adx       = float(latest.get("ADX14", 0))      if not pd.isna(latest.get("ADX14", np.nan))      else 0
    vol       = float(latest.get("volume", 0))
    vol_avg   = float(latest.get("vol_avg", 1))    if not pd.isna(latest.get("vol_avg", np.nan))    else 1
    high_52w  = float(latest.get("high_52w", price)) if not pd.isna(latest.get("high_52w", np.nan)) else price

    cond_price  = price > ma                                    # 1. Price > MA30
    cond_slope  = slope > 0                                     # 2. MA 상승
    cond_adx    = adx >= p["adx_threshold"]                    # 3. ADX 필터
    cond_vol    = (vol / vol_avg) >= p["vol_mult"] if vol_avg > 0 else False  # 4. 거래량
    cond_hi52   = price >= high_52w * (1 - p["hi52_pct"])      # 5. 52주 고가

    entry_signal = all([cond_price, cond_slope, cond_adx, cond_vol, cond_hi52])

    return {
        "ticker"       : ticker,
        "entry_signal" : entry_signal,
        "price"        : round(price, 2),
        "MA30"         : round(ma, 2),
        "MA30_slope"   : round(slope, 2),
        "ADX14"        : round(adx, 1),
        "vol_ratio"    : round(vol / vol_avg, 2) if vol_avg > 0 else 0,
        "hi52_pct"     : round((price - high_52w) / high_52w * 100, 1) if high_52w > 0 else 0,
        "cond_price"   : cond_price,
        "cond_slope"   : cond_slope,
        "cond_adx"     : cond_adx,
        "cond_vol"     : cond_vol,
        "cond_hi52"    : cond_hi52,
        "params"       : p,
    }


def check_stage2_entry(
    all_dfs: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Stage 2 자산만 추려서 진입 조건 재확인
    통과한 자산만 반환

    Returns
    -------
    {ticker: df}  — Stage 2 진입 조건 통과 자산만
    """
    print(f"\n[Stage 2 진입 필터]")
    passed  = {}
    failed  = []
    details = []

    for ticker, df in all_dfs.items():
        # Stage 2 분류된 자산만 대상
        current_stage = df["stage_current"].iloc[-1] if "stage_current" in df.columns else 0
        if current_stage != 2:
            continue

        result = check_entry(df, ticker)
        details.append(result)

        if result["entry_signal"]:
            passed[ticker] = df
        else:
            failed.append(ticker)

    # 상세 리포트 출력
    if details:
        detail_df = pd.DataFrame(details)[[
            "ticker", "entry_signal", "price", "MA30", "ADX14",
            "vol_ratio", "hi52_pct",
            "cond_price", "cond_slope", "cond_adx", "cond_vol", "cond_hi52"
        ]]
        print(detail_df.to_string(index=False))

    print(f"\n  통과: {len(passed)}개  |  탈락: {len(failed)}개")
    if failed:
        print(f"  탈락 종목: {', '.join(failed[:10])}" +
              (f" 외 {len(failed)-10}개" if len(failed) > 10 else ""))

    return passed
