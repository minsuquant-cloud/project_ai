"""
backtest/backtester.py
───────────────────────
Stage 2 기반 워크포워드 백테스트
입력: {ticker: weekly_df with stage + indicators}
출력: 수익률, 샤프, MDD, 승률 등

전략:
    - Stage 2 진입 시 매수
    - Stage 2 이탈(Stage 3/4) 시 매도
    - 비중은 balanced 방식 적용
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

RESULT_DIR = Path(__file__).parent.parent / "data" / "processed"


@dataclass
class BacktestResult:
    ticker        : str
    total_return  : float = 0.0
    sharpe        : float = 0.0
    max_drawdown  : float = 0.0
    win_rate      : float = 0.0
    num_trades    : int   = 0
    avg_hold_weeks: float = 0.0
    start_date    : str   = ""
    end_date      : str   = ""

    def __repr__(self):
        return (
            f"{self.ticker:12s} | "
            f"수익률={self.total_return:+.1%} | "
            f"샤프={self.sharpe:.2f} | "
            f"MDD={self.max_drawdown:.1%} | "
            f"승률={self.win_rate:.1%} | "
            f"거래={self.num_trades}회"
        )


def backtest_single(
    df:         pd.DataFrame,
    ticker:     str,
    start:      Optional[str] = None,
    end:        Optional[str] = None,
    hold_weeks: int = 8,
) -> BacktestResult:
    """
    단일 자산 백테스트
    Stage 2 진입 → 보유 → Stage 이탈 or hold_weeks 만료 시 청산
    """
    d = df.copy()
    if start:
        d = d[d.index >= pd.Timestamp(start)]
    if end:
        d = d[d.index <= pd.Timestamp(end)]
    if len(d) < 30 or "stage" not in d.columns:
        return BacktestResult(ticker=ticker)

    closes  = d["close"].values
    stages  = d["stage"].values
    equity  = [1.0]
    trades  = []
    in_trade     = False
    entry_price  = 0.0
    entry_bar    = 0

    for i in range(1, len(closes)):
        if in_trade:
            held = i - entry_bar
            exit_signal = (stages[i] != 2) or (held >= hold_weeks)
            if exit_signal:
                ret = closes[i] / entry_price - 1
                trades.append(ret)
                equity.append(equity[-1] * (1 + ret))
                in_trade = False
            else:
                equity.append(equity[-1])
        else:
            equity.append(equity[-1])

        if not in_trade and stages[i] == 2 and stages[i-1] != 2:
            in_trade    = True
            entry_price = closes[i]
            entry_bar   = i

    if not trades:
        return BacktestResult(ticker=ticker)

    arr = np.array(equity)
    rets = np.diff(arr) / arr[:-1]
    sharpe  = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(52)
    peak    = np.maximum.accumulate(arr)
    mdd     = float(((arr - peak) / peak).min())

    return BacktestResult(
        ticker         = ticker,
        total_return   = float(arr[-1] - 1),
        sharpe         = float(sharpe),
        max_drawdown   = mdd,
        win_rate       = float((np.array(trades) > 0).mean()),
        num_trades     = len(trades),
        avg_hold_weeks = float(hold_weeks),
        start_date     = str(d.index[0].date()),
        end_date       = str(d.index[-1].date()),
    )


def run_backtest_all(
    all_dfs:   dict[str, pd.DataFrame],
    start:     Optional[str] = None,
    end:       Optional[str] = None,
    hold_weeks: int = 8,
    save:      bool = True,
) -> pd.DataFrame:
    """
    전체 자산 백테스트 실행

    Returns
    -------
    DataFrame: 자산별 백테스트 결과
    """
    print(f"\n[백테스트] {len(all_dfs)}개 자산")
    print(f"  기간: {start or '전체'} ~ {end or '현재'}")

    results = []
    for ticker, df in all_dfs.items():
        r = backtest_single(df, ticker, start=start, end=end, hold_weeks=hold_weeks)
        results.append(r)
        print(f"  {r}")

    result_df = pd.DataFrame([{
        "ticker"       : r.ticker,
        "total_return" : r.total_return,
        "sharpe"       : r.sharpe,
        "max_drawdown" : r.max_drawdown,
        "win_rate"     : r.win_rate,
        "num_trades"   : r.num_trades,
        "avg_hold_weeks": r.avg_hold_weeks,
        "start_date"   : r.start_date,
        "end_date"     : r.end_date,
    } for r in results]).sort_values("sharpe", ascending=False)

    if save:
        save_path = RESULT_DIR / "backtest_results.csv"
        result_df.to_csv(save_path, index=False)
        print(f"\n  결과 저장 → {save_path}")

    return result_df
