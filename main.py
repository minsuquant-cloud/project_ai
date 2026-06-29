"""
main.py
───────
Weinstein Stage 멀티에셋 포트폴리오 파이프라인

사용법:
    python main.py                    # 전체 실행 (기본: balanced)
    python main.py --mode aggressive  # 공격형
    python main.py --mode balanced    # 균형형 (기본값)
    python main.py --mode defensive   # 보수형
    python main.py --step collect     # 멀티에셋 수집만
    python main.py --no-quality       # Quality Score 없이 실행
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

from utils.logger         import setup_logger
from utils.loader         import load_stocks, load_assets, merge_all
from technical.indicators import calc_indicators
from filter.stage_filter  import run_stage_filter
from filter.stage2_entry  import check_stage2_entry
from portfolio.scorer     import calc_trend_score, calc_risk_score, calc_quality_score, blend_scores
from portfolio.allocator  import calc_weights
from report.generator     import generate_report

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def main(step: str = "all", mode: str = "balanced", use_quality: bool = True):
    logger = setup_logger("main", LOG_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.log")
    logger.info(f"=== 파이프라인 시작 | step={step} | mode={mode} | quality={use_quality} ===")

    # ── 01: 멀티에셋 수집 ──────────────────────────
    if step in ("all", "collect"):
        logger.info("[01] 멀티에셋 데이터 수집")
        from data.collect_assets import collect_all_assets
        collect_all_assets()

    # ── 02: 데이터 로딩 ────────────────────────────
    logger.info("[02] 데이터 로딩 및 전처리")
    stocks_df = load_stocks()
    assets_df = load_assets()
    all_df    = merge_all(stocks_df, assets_df)

    # ── 03: 기술적 지표 계산 ───────────────────────
    if step in ("all", "stage", "portfolio"):
        logger.info("[03] 기술적 지표 계산")
        indicators_df = calc_indicators(all_df)

    # ── 04: Stage 판별 ─────────────────────────────
    if step in ("all", "stage", "portfolio"):
        logger.info("[04] Weinstein Stage 판별")
        stage_df = run_stage_filter(indicators_df)

    # ── 05: Stage 2 진입 필터 ──────────────────────
    if step in ("all", "stage", "portfolio"):
        logger.info("[05] Stage 2 진입 조건 필터")
        stage2_df = check_stage2_entry(stage_df)
        logger.info(f"     Stage 2 통과: {len(stage2_df)}개 자산")

    # ── 06~09: 포트폴리오 구성 ─────────────────────
    if step in ("all", "portfolio"):

        # Stage 2 없으면 현금
        if not stage2_df:
            logger.warning("     Stage 2 통과 자산 없음 → 현금 100%")
            portfolio_df = pd.DataFrame([{
                "ticker": "CASH", "weight": 1.0, "weight_pct": "100.0%",
                "price": 0, "MA30": 0, "ADX14": 0,
                "trend_score": 0, "risk_score": 0,
                "RS": 0, "momentum_12": 0, "vol_ratio": 0,
                "ATR_ratio_pct": 0, "date": "", "mode": mode,
            }])
            generate_report(portfolio_df, stage_df, mode=mode)
            logger.info("=== 파이프라인 완료 ===")
            return

        # ── 06: Trend / Risk Score ─────────────────
        logger.info("[06] Trend Score / Risk Score 계산")
        scored_df = calc_trend_score(stage2_df)
        scored_df = calc_risk_score(scored_df)

        # ── 07: 기술적 비중 결정 ───────────────────
        logger.info("[07] 기술적 비중 결정")
        portfolio_df = calc_weights(scored_df, mode=mode)

        # ── 08: Quality Score 결합 ─────────────────
        if use_quality:
            logger.info("[08] Quality Score 결합 (converted_data1 + data4)")
            stage2_tickers = list(stage2_df.keys())
            quality_df = calc_quality_score(stage2_tickers)
            portfolio_df = blend_scores(
                portfolio_df, quality_df,
                w_trend=0.60, w_quality=0.40, sector_cap=0.40
            )
        else:
            logger.info("[08] Quality Score 건너뜀 (--no-quality)")

        # ── 09: 리포트 ─────────────────────────────
        logger.info("[09] 리포트 생성")
        generate_report(portfolio_df, stage_df, mode=mode)

    logger.info("=== 파이프라인 완료 ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="all",
                        choices=["all", "collect", "stage", "portfolio"])
    parser.add_argument("--mode", default="balanced",
                        choices=["aggressive", "balanced", "defensive"])
    parser.add_argument("--no-quality", action="store_true",
                        help="Quality Score 없이 기술적 분석만 실행")
    args = parser.parse_args()
    main(step=args.step, mode=args.mode, use_quality=not args.no_quality)
