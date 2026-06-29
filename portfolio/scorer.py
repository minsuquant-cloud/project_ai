"""
portfolio/scorer.py
────────────────────
Trend Score / Risk Score / Quality Score 계산
Stage 2 통과 자산 중 "얼마나 강한가" / "얼마나 위험한가" / "재무가 얼마나 좋은가" 수치화

Trend Score:
    RS (상대강도) 40% + 모멘텀 12주 30% + MA30 기울기 30%

Risk Score:
    ATR_ratio 50% + MDD 30% + 베타 20%

Quality Score (converted_data4 기반):
    매출 YoY 성장률 25% + 영업이익률 25% + ROE 20% + 부채안정성 15% + 현금흐름 15%

최종 비중:
    Trend Score 60% + Quality Score 40%  (섹터 분산 제한 적용)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# ── 데이터 경로 ────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA1_PATH = BASE_DIR / "data" / "processed" / "converted_data1.csv"
DATA4_PATH = BASE_DIR / "data" / "processed" / "converted_data4.csv"


# ══════════════════════════════════════════════
#  공통 헬퍼
# ══════════════════════════════════════════════

def _minmax(s: pd.Series) -> pd.Series:
    """0~1 정규화"""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


def _calc_mdd(df: pd.DataFrame, window: int = 52) -> float:
    """최근 window주 최대낙폭"""
    closes = df["close"].tail(window)
    peak   = closes.cummax()
    dd     = (closes - peak) / peak
    return float(dd.min())


def _safe_col(df: pd.DataFrame, keyword: str) -> Optional[pd.Series]:
    """컬럼명 부분 매칭"""
    matches = [c for c in df.columns if keyword in c]
    return df[matches[0]] if matches else None


# ══════════════════════════════════════════════
#  1. Trend Score
# ══════════════════════════════════════════════

def calc_trend_score(
    all_dfs: dict[str, pd.DataFrame],
    w_rs:    float = 0.40,
    w_mom:   float = 0.30,
    w_slope: float = 0.30,
) -> dict[str, pd.DataFrame]:
    """RS + 모멘텀 + MA기울기 → Trend Score"""
    print(f"\n[Trend Score 계산]")

    if not all_dfs:
        print("  [WARN] 자산 없음")
        return all_dfs

    records = []
    for ticker, df in all_dfs.items():
        latest = df.iloc[-1]
        records.append({
            "ticker"  : ticker,
            "rs"      : float(latest.get("RS", 0.5))            if not pd.isna(latest.get("RS", np.nan))            else 0.5,
            "momentum": float(latest.get("momentum_12", 0))     if not pd.isna(latest.get("momentum_12", np.nan))   else 0.0,
            "slope"   : float(latest.get("MA30_slope", 0))      if not pd.isna(latest.get("MA30_slope", np.nan))    else 0.0,
        })

    scores_df = pd.DataFrame(records).set_index("ticker").astype(float)
    scores_df["rs_norm"]    = _minmax(scores_df["rs"])
    scores_df["mom_norm"]   = _minmax(scores_df["momentum"])
    scores_df["slope_norm"] = _minmax(scores_df["slope"])
    scores_df["trend_score"] = (
        scores_df["rs_norm"]    * w_rs   +
        scores_df["mom_norm"]   * w_mom  +
        scores_df["slope_norm"] * w_slope
    )

    result = {}
    for ticker, df in all_dfs.items():
        d = df.copy()
        d["trend_score"] = scores_df.loc[ticker, "trend_score"] if ticker in scores_df.index else 0.5
        result[ticker] = d

    print(scores_df[["trend_score"]].sort_values("trend_score", ascending=False).to_string())
    return result


# ══════════════════════════════════════════════
#  2. Risk Score
# ══════════════════════════════════════════════

def calc_risk_score(
    all_dfs:          dict[str, pd.DataFrame],
    w_atr:            float = 0.50,
    w_mdd:            float = 0.30,
    w_beta:           float = 0.20,
    benchmark_ticker: Optional[str] = "SPY",
) -> dict[str, pd.DataFrame]:
    """ATR + MDD + 베타 → Risk Score"""
    print(f"\n[Risk Score 계산]")

    if not all_dfs:
        print("  [WARN] 자산 없음")
        return all_dfs

    bench_rets = None
    if benchmark_ticker and benchmark_ticker in all_dfs:
        bench_rets = all_dfs[benchmark_ticker]["close"].pct_change().dropna()

    records = []
    for ticker, df in all_dfs.items():
        latest    = df.iloc[-1]
        atr_ratio = float(latest.get("ATR_ratio", 0.02)) if not pd.isna(latest.get("ATR_ratio", np.nan)) else 0.02
        mdd       = abs(_calc_mdd(df))

        beta = 1.0
        if bench_rets is not None and ticker != benchmark_ticker:
            asset_rets = df["close"].pct_change().dropna()
            common_idx = bench_rets.index.intersection(asset_rets.index)
            if len(common_idx) > 20:
                b_r = bench_rets.loc[common_idx]
                a_r = asset_rets.loc[common_idx]
                cov = np.cov(a_r, b_r)[0, 1]
                var = np.var(b_r)
                beta = cov / var if var > 0 else 1.0
                beta = max(0, min(beta, 5.0))

        records.append({"ticker": ticker, "atr_ratio": atr_ratio, "mdd": mdd, "beta": beta})

    scores_df = pd.DataFrame(records).set_index("ticker").astype(float)
    scores_df["atr_norm"]  = _minmax(scores_df["atr_ratio"])
    scores_df["mdd_norm"]  = _minmax(scores_df["mdd"])
    scores_df["beta_norm"] = _minmax(scores_df["beta"])
    scores_df["risk_score"] = (
        scores_df["atr_norm"]  * w_atr +
        scores_df["mdd_norm"]  * w_mdd +
        scores_df["beta_norm"] * w_beta
    )

    result = {}
    for ticker, df in all_dfs.items():
        d = df.copy()
        if ticker in scores_df.index:
            d["risk_score"] = scores_df.loc[ticker, "risk_score"]
            d["beta"]       = scores_df.loc[ticker, "beta"]
            d["mdd_52w"]    = -scores_df.loc[ticker, "mdd"]
        result[ticker] = d

    print(scores_df[["atr_ratio","mdd","beta","risk_score"]].round(3).sort_values("risk_score", ascending=False).to_string())
    return result


# ══════════════════════════════════════════════
#  3. Quality Score (converted_data1 + data4)
# ══════════════════════════════════════════════

def _load_sector(path: Path) -> pd.DataFrame:
    """converted_data1.csv → 섹터 정보"""
    if not path.exists():
        print(f"  [WARN] 섹터 파일 없음: {path.name}")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["코드"] = df["코드"].astype(str).str.replace("A", "", regex=False).str.zfill(6)
    df = df.set_index("코드")
    rename = {
        "코드명": "name", "상장된 시장": "market",
        "FnGuide Sector": "sector",
        "FnGuide Industry Group": "industry_group",
        "FnGuide Industry": "industry",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = [c for c in ["name","market","sector","industry_group","industry"] if c in df.columns]
    print(f"  [OK] 섹터 데이터: {len(df)}개 종목")
    return df[keep]


def _load_financial(path: Path, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """converted_data4.csv → 종목별 분기 재무 DataFrame"""
    if not path.exists():
        print(f"  [WARN] 재무 파일 없음: {path.name}")
        return {}

    print(f"  [로딩] {path.name} ...")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["코드"] = df["코드"].astype(str).str.replace("A", "", regex=False).str.zfill(6)

    meta_cols = ["코드", "코드명", "아이템명"]
    date_cols = [c for c in df.columns if c not in meta_cols]

    # Stage 2 종목만 필터
    df = df[df["코드"].isin(tickers)]
    if df.empty:
        print(f"  [WARN] 해당 종목 재무 데이터 없음")
        return {}

    result = {}
    for ticker, group in df.groupby("코드"):
        pivot = group.set_index("아이템명")[date_cols].T
        pivot.index = pd.to_datetime(pivot.index)
        pivot.index.name = "date"
        pivot = pivot.apply(pd.to_numeric, errors="coerce").sort_index()
        result[ticker] = pivot

    print(f"  [OK] 재무 데이터: {len(result)}개 종목 × {len(date_cols)}분기")
    return result


def _yoy_growth(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 5:
        return 0.0
    recent   = float(s.iloc[-1])
    year_ago = float(s.iloc[-5])
    return (recent - year_ago) / abs(year_ago) if year_ago != 0 else 0.0


def _quality_single(fin_df: pd.DataFrame) -> dict:
    revenue   = _safe_col(fin_df, "매출액")
    op_profit = _safe_col(fin_df, "영업이익")
    roe       = _safe_col(fin_df, "ROE")
    debt      = _safe_col(fin_df, "부채총계")
    equity    = _safe_col(fin_df, "자본총계")
    cfo       = _safe_col(fin_df, "영업활동으로인한현금흐름")

    # 매출 YoY 성장률
    rev_growth = _yoy_growth(revenue) if revenue is not None else 0.0

    # 영업이익률 (최근 4분기 평균)
    op_margin = 0.0
    if revenue is not None and op_profit is not None:
        margin = (op_profit / revenue.replace(0, np.nan)).dropna()
        op_margin = float(margin.iloc[-4:].mean()) if len(margin) >= 4 else float(margin.mean()) if len(margin) > 0 else 0.0

    # ROE 최근값
    roe_val = 0.0
    if roe is not None:
        roe_clean = roe.dropna()
        roe_val   = float(roe_clean.iloc[-1]) / 100 if len(roe_clean) > 0 else 0.0

    # 부채비율 역수
    debt_inv = 0.5
    if debt is not None and equity is not None:
        dr = (debt / equity.replace(0, np.nan)).dropna()
        if len(dr) > 0:
            debt_inv = 1 / (1 + max(float(dr.iloc[-1]), 0))

    # 영업현금흐름 성장률
    cfo_growth = _yoy_growth(cfo) if cfo is not None else 0.0

    return {
        "rev_growth": rev_growth,
        "op_margin" : op_margin,
        "roe"       : roe_val,
        "debt_inv"  : debt_inv,
        "cfo_growth": cfo_growth,
    }


def calc_quality_score(
    tickers:  list[str],
    w_rev:    float = 0.25,
    w_op:     float = 0.25,
    w_roe:    float = 0.20,
    w_debt:   float = 0.15,
    w_cfo:    float = 0.15,
) -> pd.DataFrame:
    """
    converted_data1 + converted_data4 로딩 후 Quality Score 계산

    Parameters
    ----------
    tickers : Stage 2 통과 종목 코드 리스트

    Returns
    -------
    DataFrame: ticker, quality_score, sector 등
    """
    print(f"\n[Quality Score 계산] {len(tickers)}개 종목")

    # 섹터 데이터
    sector_df = _load_sector(DATA1_PATH)

    # 재무 데이터 (KOSPI/KOSDAQ 종목만 — 6자리 숫자 코드)
    stock_tickers = [t for t in tickers if t.isdigit() or (len(t) == 6 and t[:3].isdigit())]
    fin_data = _load_financial(DATA4_PATH, stock_tickers)

    rows = []
    for ticker in tickers:
        if ticker in fin_data:
            q = _quality_single(fin_data[ticker])
        else:
            # ETF / 비트코인 등 재무 데이터 없는 자산 → 중간값
            q = {"rev_growth": 0.0, "op_margin": 0.0, "roe": 0.0,
                 "debt_inv": 0.5, "cfo_growth": 0.0}

        # 섹터 정보
        sector = sector_df.loc[ticker, "sector"] if (not sector_df.empty and ticker in sector_df.index) else "기타"

        q["ticker"] = ticker
        q["sector"] = sector
        rows.append(q)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("ticker")

    # 정규화
    num_cols = ["rev_growth", "op_margin", "roe", "debt_inv", "cfo_growth"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["rev_norm"]  = _minmax(df["rev_growth"].clip(-1, 2))
    df["op_norm"]   = _minmax(df["op_margin"].clip(-0.5, 0.5))
    df["roe_norm"]  = _minmax(df["roe"].clip(-0.5, 1.0))
    df["debt_norm"] = _minmax(df["debt_inv"])
    df["cfo_norm"]  = _minmax(df["cfo_growth"].clip(-1, 2))

    df["quality_score"] = (
        df["rev_norm"]  * w_rev  +
        df["op_norm"]   * w_op   +
        df["roe_norm"]  * w_roe  +
        df["debt_norm"] * w_debt +
        df["cfo_norm"]  * w_cfo
    )

    # 출력
    show = df[["rev_growth","op_margin","roe","debt_inv","cfo_growth","quality_score","sector"]].copy()
    show.columns = ["매출성장률","영업이익률","ROE","부채안정성","현금흐름","Quality점수","섹터"]
    print(show.round(3).sort_values("Quality점수", ascending=False).to_string())

    return df.reset_index()


# ══════════════════════════════════════════════
#  4. Trend + Quality 결합
# ══════════════════════════════════════════════

def blend_scores(
    portfolio_df:  pd.DataFrame,
    quality_df:    pd.DataFrame,
    w_trend:       float = 0.60,
    w_quality:     float = 0.40,
    sector_cap:    float = 0.40,
) -> pd.DataFrame:
    """
    Trend Score + Quality Score 결합 후 섹터 분산 제한 적용

    Parameters
    ----------
    portfolio_df : allocator.py 출력
    quality_df   : calc_quality_score() 출력
    w_trend      : Trend Score 가중치
    w_quality    : Quality Score 가중치
    sector_cap   : 단일 섹터 최대 비중
    """
    if quality_df.empty:
        print("  [WARN] Quality Score 없음 → Trend Score만 사용")
        portfolio_df["blended_score"] = portfolio_df.get("trend_score", 0.5)
        portfolio_df["weight_final"]  = portfolio_df["weight"]
        portfolio_df["weight_final_pct"] = portfolio_df["weight_pct"]
        return portfolio_df

    merged = portfolio_df.merge(
        quality_df[["ticker","quality_score","sector"]],
        on="ticker", how="left"
    )
    merged["quality_score"] = merged["quality_score"].fillna(0.5)
    merged["sector"]        = merged["sector"].fillna("기타")

    trend_col = "trend_score" if "trend_score" in merged.columns else "weight"
    merged["blended_score"] = (
        merged[trend_col]       * w_trend  +
        merged["quality_score"] * w_quality
    )

    # 비중 재계산
    total = merged["blended_score"].sum()
    merged["weight_final"] = (merged["blended_score"] / total).round(4) if total > 0 else merged["weight"]

    # ── 섹터 분산 제한 ──────────────────────────
    sector_weights = merged.groupby("sector")["weight_final"].sum()
    over_sectors   = sector_weights[sector_weights > sector_cap]

    if not over_sectors.empty:
        print(f"\n[섹터 분산 조정]")
        for sector, total_w in over_sectors.items():
            mask  = merged["sector"] == sector
            ratio = sector_cap / total_w
            merged.loc[mask, "weight_final"] *= ratio
            print(f"  {sector}: {total_w:.1%} → {sector_cap:.1%}")
        # 재정규화
        total = merged["weight_final"].sum()
        merged["weight_final"] = (merged["weight_final"] / total).round(4)

    merged["weight_final_pct"] = (merged["weight_final"] * 100).round(1).astype(str) + "%"

    print(f"\n[Trend + Quality 결합] trend={w_trend} | quality={w_quality}")
    show_cols = ["ticker","weight_final_pct","trend_score","quality_score","blended_score","sector"]
    show_cols = [c for c in show_cols if c in merged.columns]
    print(merged[show_cols].sort_values("blended_score", ascending=False).to_string(index=False))

    return merged