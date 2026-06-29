"""
technical/indicators.py
────────────────────────
Weinstein Stage 판별 + 보조 지표 계산
입력: {ticker: weekly_df}
출력: {ticker: weekly_df + 지표 컬럼들}

계산 지표:
    MA30, MA30_slope      → Stage 기준선
    ADX14                 → 횡보 필터
    volume_ratio          → 거래량 돌파 신뢰도
    high_52w              → 52주 최고가
    ATR14                 → 변동성 (비중 결정용)
    RS                    → 상대강도 (종목 순위용)
    momentum_12           → 12주 모멘텀
"""

import pandas as pd
import numpy as np
from typing import Optional


# ══════════════════════════════════════════════
#  ADX 계산 (Wilder)
# ══════════════════════════════════════════════

def _wilder_smooth(s: pd.Series, n: int) -> pd.Series:
    result = np.full(len(s), np.nan)
    valid  = s.dropna()
    if len(valid) < n:
        return pd.Series(result, index=s.index)

    i0 = s.index.get_loc(valid.index[0])
    val = float(valid.iloc[0])
    result[i0] = val
    for i in range(i0 + 1, len(s)):
        if not np.isnan(s.iloc[i]):
            val = (val * (n - 1) + float(s.iloc[i])) / n
            result[i] = val
    return pd.Series(result, index=s.index)


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ADX 계산"""
    high, low, close = df["high"], df["low"], df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    dm_plus  = np.where((high.diff() > 0) & (high.diff() > -low.diff()),
                         high.diff().clip(lower=0), 0)
    dm_minus = np.where((-low.diff() > 0) & (-low.diff() > high.diff()),
                         (-low.diff()).clip(lower=0), 0)

    tr_s   = _wilder_smooth(tr,  period)
    dmp_s  = _wilder_smooth(pd.Series(dm_plus,  index=df.index), period)
    dmm_s  = _wilder_smooth(pd.Series(dm_minus, index=df.index), period)

    di_plus  = 100 * dmp_s / tr_s.replace(0, np.nan)
    di_minus = 100 * dmm_s / tr_s.replace(0, np.nan)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    return _wilder_smooth(dx.fillna(0), period)


# ══════════════════════════════════════════════
#  ATR 계산
# ══════════════════════════════════════════════

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range"""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ══════════════════════════════════════════════
#  단일 종목 지표 계산
# ══════════════════════════════════════════════

def calc_single(
    df: pd.DataFrame,
    ma_period:     int = 30,
    slope_weeks:   int = 4,
    adx_period:    int = 14,
    vol_avg_weeks: int = 20,
    hi52_weeks:    int = 52,
    mom_weeks:     int = 12,
) -> pd.DataFrame:
    """
    단일 종목 모든 지표 계산
    """
    d = df.copy()

    # MA30 + 기울기
    d["MA30"]       = d["close"].rolling(ma_period).mean()
    d["MA30_slope"] = d["MA30"] - d["MA30"].shift(slope_weeks)

    # ADX
    d["ADX14"] = calc_adx(d, adx_period)

    # 거래량 비율
    d["vol_avg"]   = d["volume"].rolling(vol_avg_weeks).mean()
    d["vol_ratio"] = d["volume"] / d["vol_avg"].replace(0, np.nan)

    # 52주 고가
    d["high_52w"]  = d["high"].rolling(hi52_weeks).max()
    d["hi52_pct"]  = (d["close"] - d["high_52w"]) / d["high_52w"]  # 0에 가까울수록 신고가

    # ATR (변동성)
    d["ATR14"]     = calc_atr(d, adx_period)
    d["ATR_ratio"] = d["ATR14"] / d["close"].replace(0, np.nan)   # 가격 대비 변동성

    # 모멘텀
    d["momentum_12"] = d["close"] / d["close"].shift(mom_weeks) - 1

    return d


# ══════════════════════════════════════════════
#  상대강도 (RS) — 전체 자산 대비
# ══════════════════════════════════════════════

def calc_rs_cross_assets(
    all_dfs: dict[str, pd.DataFrame],
    period:  int = 12,
) -> dict[str, pd.DataFrame]:
    """
    전체 자산 간 상대강도 계산
    RS = 해당 자산의 12주 수익률 순위 (0~1, 높을수록 강함)
    """
    # 최근 공통 날짜 기준 수익률 집계
    returns = {}
    for ticker, df in all_dfs.items():
        if len(df) >= period + 1:
            ret = df["close"].iloc[-1] / df["close"].iloc[-(period+1)] - 1
            returns[ticker] = ret

    if not returns:
        return all_dfs

    # 순위 → 0~1 정규화
    sorted_tickers = sorted(returns, key=returns.get)
    rs_rank = {t: i / (len(sorted_tickers) - 1)
               for i, t in enumerate(sorted_tickers)}

    # 각 df에 RS 컬럼 추가 (최신행에만 의미 있음)
    result = {}
    for ticker, df in all_dfs.items():
        d = df.copy()
        d["RS"] = rs_rank.get(ticker, 0.5)
        result[ticker] = d

    return result


# ══════════════════════════════════════════════
#  전체 파이프라인
# ══════════════════════════════════════════════

def calc_indicators(
    all_dfs: dict[str, pd.DataFrame],
    ma_period: int = 30,
) -> dict[str, pd.DataFrame]:
    """
    전체 자산 지표 계산 진입점
    main.py 에서 호출
    """
    print(f"\n[지표 계산] {len(all_dfs)}개 자산")
    result = {}

    for ticker, df in all_dfs.items():
        if len(df) < ma_period + 10:
            print(f"  [SKIP] {ticker}: 데이터 부족")
            continue
        result[ticker] = calc_single(df, ma_period=ma_period)

    # 전체 자산 간 상대강도
    result = calc_rs_cross_assets(result)

    print(f"  완료: {len(result)}개")
    return result
