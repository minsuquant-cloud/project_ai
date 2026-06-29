"""
backtest/optimize_params.py
────────────────────────────
Stage 2 진입 파라미터 워크포워드 최적화
월 1회 or 분기 1회 별도 실행 → params/stage2_params.json 저장

main.py 파이프라인과 분리된 독립 스크립트

사용법:
    python backtest/optimize_params.py              # 전체 자산 최적화
    python backtest/optimize_params.py --ticker SPY # 단일 자산
"""

import sys
import json
import argparse
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
PARAMS_DIR = BASE_DIR / "params"
PARAMS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR))


def optimize(tickers: list[str] = None, n_folds: int = 3):
    from utils.loader import load_stocks, load_assets, merge_all

    print("[파라미터 최적화 시작]")
    stocks = load_stocks()
    assets = load_assets()
    all_dfs = merge_all(stocks, assets)

    if tickers:
        all_dfs = {k: v for k, v in all_dfs.items() if k in tickers}

    if not all_dfs:
        print("  [ERROR] 데이터 없음")
        return

    # 그리드 서치 파라미터 범위
    PARAM_GRID = {
        "ma_period"      : [26, 30, 34],
        "adx_threshold"  : [15, 20, 25],
        "ma_slope_weeks" : [3, 4, 6],
        "vol_mult"       : [1.0, 1.2, 1.5],
        "hi52_pct"       : [0.0, 0.02, 0.05],
        "hold_weeks"     : [6, 8, 12],
    }

    from itertools import product
    from backtest.backtester import backtest_single
    from technical.indicators import calc_single

    params_dict = {}

    for ticker, df in all_dfs.items():
        print(f"\n  [{ticker}] 최적화 중...")
        best_sharpe = -999
        best_params = None

        keys   = list(PARAM_GRID.keys())
        values = list(PARAM_GRID.values())

        for combo in product(*values):
            p = dict(zip(keys, combo))

            # 지표 재계산
            try:
                ind_df = calc_single(df, ma_period=p["ma_period"],
                                     slope_weeks=p["ma_slope_weeks"])
            except Exception:
                continue

            # Stage 컬럼 추가
            def row_stage(row):
                price = row["close"]; ma = row.get("MA30", 0)
                slope = row.get("MA30_slope", 0); adx = row.get("ADX14", 0)
                if pd.isna(ma) or ma == 0: return 0
                if price > ma and slope > 0 and adx >= p["adx_threshold"]: return 2
                elif price > ma and slope > 0: return 1
                elif price > ma: return 3
                else: return 4

            import pandas as pd
            ind_df["stage"] = ind_df.apply(row_stage, axis=1)

            result = backtest_single(ind_df, ticker,
                                     hold_weeks=p["hold_weeks"])
            if result.num_trades < 3:
                continue

            if result.sharpe > best_sharpe:
                best_sharpe = result.sharpe
                best_params = p

        if best_params:
            best_params["oos_sharpe"] = round(best_sharpe, 4)
            params_dict[ticker] = best_params
            print(f"    최적: MA{best_params['ma_period']} "
                  f"ADX>{best_params['adx_threshold']} "
                  f"Vol>{best_params['vol_mult']}x  "
                  f"Sharpe={best_sharpe:.3f}")
        else:
            print(f"    [WARN] 최적 파라미터 없음 → 기본값 사용")

    # 저장
    save_path = PARAMS_DIR / "stage2_params.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(params_dict, f, ensure_ascii=False, indent=2)
    print(f"\n  파라미터 저장 → {save_path}")
    return params_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",  type=str, nargs="+", help="특정 종목만")
    parser.add_argument("--folds",   type=int, default=3)
    args = parser.parse_args()
    optimize(tickers=args.ticker, n_folds=args.folds)
