"""
portfolio/allocator.py
───────────────────────
최종 자산 비중 결정
Trend Score / Risk Score → 비중 계산

3가지 전략 모드:
    aggressive : Trend Score 비례 집중 투자 (추세 강한 자산에 몰빵)
    balanced   : Trend / Risk 동시 고려 (기본값, 추천)
    defensive  : 안정 자산 비중 확대 + 변동성 낮은 자산 선호

비트코인 특이사항:
    - ATR이 타 자산 대비 3~5배 → risk_score 자동으로 높아짐
    - 별도 상한선(BTC_MAX_WEIGHT) 추가 적용
"""

import pandas as pd
import numpy as np

# ── 비중 제한 설정 ─────────────────────────────
BTC_MAX_WEIGHT   = 0.15   # 비트코인 최대 15%
SINGLE_MAX_WEIGHT = 0.35  # 단일 자산 최대 35%
CASH_TICKER      = "CASH" # Stage 2 자산 없을 때 현금 보유

# ── 안정 자산 정의 (defensive 모드에서 최소 비중 보장) ──
SAFE_ASSETS = {"TLT", "IEF", "GLD"}
SAFE_MIN_WEIGHT = 0.30    # defensive 모드: 안전 자산 합계 최소 30%


def _normalize_weights(weights: dict) -> dict:
    """비중 합계 = 1.0 정규화"""
    total = sum(weights.values())
    if total == 0:
        n = len(weights)
        return {k: 1/n for k in weights}
    return {k: v / total for k, v in weights.items()}


def _apply_caps(weights: dict) -> dict:
    """비중 상한선 적용 후 재정규화"""
    # 비트코인 상한
    if "BTC-USD" in weights:
        weights["BTC-USD"] = min(weights["BTC-USD"], BTC_MAX_WEIGHT)

    # 단일 자산 상한
    weights = {k: min(v, SINGLE_MAX_WEIGHT) for k, v in weights.items()}

    return _normalize_weights(weights)


# ══════════════════════════════════════════════
#  공격형
# ══════════════════════════════════════════════

def _aggressive(all_dfs: dict) -> dict:
    """
    Trend Score 비례 비중
    추세가 강한 자산에 집중
    """
    weights = {}
    for ticker, df in all_dfs.items():
        score = float(df.iloc[-1].get("trend_score", 0.5))
        weights[ticker] = max(score, 0.01)   # 최소값 보장
    return _apply_caps(weights)


# ══════════════════════════════════════════════
#  균형형 (추천)
# ══════════════════════════════════════════════

def _balanced(all_dfs: dict) -> dict:
    """
    Trend Score / Risk Score 비율로 비중 결정
    위험 대비 추세가 강한 자산에 비중 집중
    """
    weights = {}
    for ticker, df in all_dfs.items():
        latest = df.iloc[-1]
        trend  = float(latest.get("trend_score", 0.5))
        risk   = float(latest.get("risk_score",  0.5))

        # risk가 0이면 분모 보호
        score = trend / (risk + 0.1)
        weights[ticker] = max(score, 0.01)

    return _apply_caps(weights)


# ══════════════════════════════════════════════
#  보수형
# ══════════════════════════════════════════════

def _defensive(all_dfs: dict) -> dict:
    """
    Risk Score 역수 비중 + 안정 자산 최소 비중 보장
    변동성 낮은 자산 선호
    """
    weights = {}
    for ticker, df in all_dfs.items():
        risk  = float(df.iloc[-1].get("risk_score", 0.5))
        score = 1 / (risk + 0.1)   # risk 역수
        weights[ticker] = max(score, 0.01)

    weights = _apply_caps(weights)

    # 안전 자산 최소 비중 보장
    safe_present = [t for t in SAFE_ASSETS if t in weights]
    if safe_present:
        safe_total = sum(weights.get(t, 0) for t in safe_present)
        if safe_total < SAFE_MIN_WEIGHT:
            boost = (SAFE_MIN_WEIGHT - safe_total) / len(safe_present)
            for t in safe_present:
                weights[t] = weights.get(t, 0) + boost
            weights = _normalize_weights(weights)

    return weights


# ══════════════════════════════════════════════
#  메인 함수
# ══════════════════════════════════════════════

def calc_weights(
    all_dfs: dict[str, pd.DataFrame],
    mode:    str = "balanced",
) -> pd.DataFrame:
    """
    비중 계산 진입점

    Parameters
    ----------
    mode : 'aggressive' | 'balanced' | 'defensive'

    Returns
    -------
    DataFrame: ticker, weight, trend_score, risk_score, stage, price, ...
    """
    print(f"\n[비중 결정] 전략 모드: {mode.upper()}")

    if not all_dfs:
        print("  [WARN] Stage 2 통과 자산 없음 → 현금 100%")
        return pd.DataFrame([{
            "ticker": CASH_TICKER, "weight": 1.0,
            "trend_score": 0, "risk_score": 0
        }])

    # 전략별 비중 계산
    if mode == "aggressive":
        weights = _aggressive(all_dfs)
    elif mode == "defensive":
        weights = _defensive(all_dfs)
    else:
        weights = _balanced(all_dfs)

    # 결과 DataFrame 조립
    rows = []
    for ticker, df in all_dfs.items():
        latest = df.iloc[-1]
        rows.append({
            "ticker"      : ticker,
            "weight"      : round(weights.get(ticker, 0), 4),
            "weight_pct"  : f"{weights.get(ticker, 0)*100:.1f}%",
            "price"       : round(float(latest.get("close", 0)), 2),
            "MA30"        : round(float(latest.get("MA30", 0)), 2)       if not pd.isna(latest.get("MA30", float("nan"))) else None,
            "ADX14"       : round(float(latest.get("ADX14", 0)), 1)      if not pd.isna(latest.get("ADX14", float("nan"))) else None,
            "trend_score" : round(float(latest.get("trend_score", 0)), 4),
            "risk_score"  : round(float(latest.get("risk_score",  0)), 4),
            "RS"          : round(float(latest.get("RS", 0.5)), 3),
            "momentum_12" : round(float(latest.get("momentum_12", 0))*100, 1) if not pd.isna(latest.get("momentum_12", float("nan"))) else None,
            "vol_ratio"   : round(float(latest.get("vol_ratio", 0)), 2)  if not pd.isna(latest.get("vol_ratio", float("nan"))) else None,
            "ATR_ratio_pct": round(float(latest.get("ATR_ratio", 0))*100, 2) if not pd.isna(latest.get("ATR_ratio", float("nan"))) else None,
            "date"        : str(df.index[-1].date()),
            "mode"        : mode,
        })

    portfolio_df = pd.DataFrame(rows).sort_values("weight", ascending=False)

    # 출력
    print(portfolio_df[[
        "ticker", "weight_pct", "trend_score", "risk_score",
        "momentum_12", "ATR_ratio_pct", "RS"
    ]].to_string(index=False))
    print(f"\n  총 비중 합계: {portfolio_df['weight'].sum():.4f}")

    return portfolio_df
