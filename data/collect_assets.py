"""
data/collect_assets.py
───────────────────────
비트코인 / 금 / 원유 / S&P500 ETF / 채권 ETF 자동 수집
yfinance 사용 — 일봉 다운로드 후 data/raw/assets/ 에 저장
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RAW_DIR = Path(__file__).parent / "raw" / "assets"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── 수집 대상 정의 ─────────────────────────────────
ASSETS = {
    "SPY"    : "S&P500 ETF",
    "GLD"    : "금 ETF",
    "USO"    : "원유 ETF",
    "BTC-USD": "비트코인",
    "TLT"    : "미국 장기채 ETF",
    "IEF"    : "미국 중기채 ETF",
}

START_DATE = "2015-01-01"   # 충분한 백테스트 기간 확보


def collect_asset(ticker: str, name: str) -> bool:
    """단일 자산 수집 후 CSV 저장"""
    try:
        df = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)
        if df.empty:
            print(f"  [WARN] {ticker} ({name}): 데이터 없음")
            return False

        # 컬럼 정리
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index.name = "date"
        df = df.dropna(subset=["close"])

        # 저장
        save_path = RAW_DIR / f"{ticker}.csv"
        df.to_csv(save_path)
        print(f"  [OK] {ticker} ({name}): {len(df)}행 → {save_path.name}")
        return True

    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return False


def collect_all_assets():
    """전체 멀티에셋 수집"""
    print(f"\n[멀티에셋 수집] {datetime.now():%Y-%m-%d %H:%M}")
    print(f"  저장 경로: {RAW_DIR}")
    print(f"  수집 기간: {START_DATE} ~ 오늘\n")

    results = {}
    for ticker, name in ASSETS.items():
        results[ticker] = collect_asset(ticker, name)

    success = sum(results.values())
    print(f"\n  완료: {success}/{len(ASSETS)}개 성공")
    return results


if __name__ == "__main__":
    collect_all_assets()
