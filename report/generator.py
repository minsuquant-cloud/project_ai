"""
report/generator.py
────────────────────
포트폴리오 리포트 생성
data/processed/portfolio_report.csv + 콘솔 출력
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

REPORT_DIR = Path(__file__).parent.parent / "data" / "processed"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def generate_report(
    portfolio_df: pd.DataFrame,
    stage_df:     dict[str, pd.DataFrame],
    mode:         str = "balanced",
) -> str:
    """
    최종 포트폴리오 리포트 생성

    Parameters
    ----------
    portfolio_df : allocator.py 출력 DataFrame
    stage_df     : {ticker: df with stage}  전체 자산 Stage 현황
    mode         : 전략 모드

    Returns
    -------
    리포트 텍스트 (파일 저장 + 콘솔 출력)
    """
    now = datetime.now()
    lines = []

    lines.append("=" * 62)
    lines.append(f"  Weinstein Stage 포트폴리오 리포트")
    lines.append(f"  생성일시: {now:%Y-%m-%d %H:%M}  |  전략: {mode.upper()}")
    lines.append("=" * 62)

    # ── 1. 최종 포트폴리오 ────────────────────────────
    lines.append("\n【 최종 포트폴리오 】")
    if portfolio_df.empty or (len(portfolio_df) == 1 and portfolio_df.iloc[0]["ticker"] == "CASH"):
        lines.append("  ※ Stage 2 통과 자산 없음 → 현금 100% 보유")
    else:
        for _, row in portfolio_df.iterrows():
            bar_len = int(row["weight"] * 30)
            bar     = "█" * bar_len + "░" * (30 - bar_len)
            lines.append(
                f"  {row['ticker']:10s} │{bar}│ "
                f"{row['weight_pct']:>6s}  "
                f"가격:{row['price']:>10,.2f}"
            )

    # ── 2. 전체 Stage 현황 ────────────────────────────
    lines.append("\n【 전체 자산 Stage 현황 】")
    stage_counts = {1: [], 2: [], 3: [], 4: []}
    for ticker, df in stage_df.items():
        s = int(df["stage_current"].iloc[-1]) if "stage_current" in df.columns else 0
        if s in stage_counts:
            stage_counts[s].append(ticker)

    stage_names = {1: "횡보", 2: "상승(투자)", 3: "천장", 4: "하락"}
    for s, name in stage_names.items():
        tickers = stage_counts[s]
        short   = ", ".join(tickers[:8]) + (f" 외 {len(tickers)-8}개" if len(tickers) > 8 else "")
        lines.append(f"  Stage {s} ({name:8s}): {len(tickers):4d}개  {short}")

    # ── 3. 주요 지표 요약 ─────────────────────────────
    if not portfolio_df.empty and "trend_score" in portfolio_df.columns:
        lines.append("\n【 보유 자산 지표 요약 】")
        cols = ["ticker", "weight_pct", "trend_score", "risk_score",
                "momentum_12", "ATR_ratio_pct", "RS"]
        avail = [c for c in cols if c in portfolio_df.columns]
        lines.append(portfolio_df[avail].to_string(index=False))

    # ── 4. 매매 신호 ──────────────────────────────────
    lines.append("\n【 매매 신호 】")
    lines.append(f"  ▶ 매수 신호 ({len(portfolio_df)}개):")
    for _, row in portfolio_df.iterrows():
        if row["ticker"] != "CASH":
            lines.append(f"    BUY  {row['ticker']:10s} → 비중 {row['weight_pct']}")

    # Stage 3으로 전환된 자산 (청산 준비)
    stage3 = stage_counts.get(3, [])
    if stage3:
        lines.append(f"\n  ▶ 청산 준비 (Stage 3 전환, {len(stage3)}개):")
        for t in stage3:
            lines.append(f"    SELL {t}")

    lines.append("\n" + "=" * 62)

    report_text = "\n".join(lines)

    # 콘솔 출력
    print(report_text)

    # CSV 저장
    csv_path = REPORT_DIR / f"portfolio_{now:%Y%m%d}.csv"
    portfolio_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 텍스트 저장
    txt_path = REPORT_DIR / f"report_{now:%Y%m%d_%H%M}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n  📄 리포트 저장 → {txt_path.name}")
    print(f"  📊 CSV 저장    → {csv_path.name}")

    return report_text
