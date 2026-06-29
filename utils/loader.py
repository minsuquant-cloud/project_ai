"""
utils/loader.py
───────────────
주식 데이터(팀원 제공) + 멀티에셋(본인 수집) 로딩
일봉 → 주봉 변환 후 통합 DataFrame 반환
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

BASE_DIR    = Path(__file__).parent.parent
STOCKS_DIR  = BASE_DIR / "data" / "raw" / "stocks"
ASSETS_DIR  = BASE_DIR / "data" / "raw" / "assets"


# ══════════════════════════════════════════════
#  일봉 → 주봉 변환
# ══════════════════════════════════════════════

def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    일봉 DataFrame → 주봉 (월요일 기준)
    입력 컬럼: date(index), open, high, low, close, volume
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    weekly = df.resample("W-MON").agg({
        "open"  : "first",
        "high"  : "max",
        "low"   : "min",
        "close" : "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    # 거래 없는 주 제거
    weekly = weekly[weekly["volume"] > 0]
    return weekly


# ══════════════════════════════════════════════
#  단일 CSV 로딩
# ══════════════════════════════════════════════

def _load_csv(path: Path, ticker: str) -> Optional[pd.DataFrame]:
    """CSV 로딩 공통 함수"""
    if not path.exists():
        print(f"  [WARN] 파일 없음: {path.name}")
        return None
    try:
        df = pd.read_csv(path)

        # 날짜 컬럼 자동 탐지 (Date / date / 날짜 등)
        date_cols = [c for c in df.columns if c.lower() in ("date", "날짜", "일자")]
        if not date_cols:
            print(f"  [WARN] {ticker}: 날짜 컬럼 없음")
            return None

        df = df.rename(columns={date_cols[0]: "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # 컬럼명 소문자 통일
        df.columns = [c.lower() for c in df.columns]

        # 필수 컬럼 확인
        required = {"open", "high", "low", "close", "volume"}
        # 대소문자 변형 자동 매핑
        col_map = {}
        for req in required:
            for col in df.columns:
                if col.lower() == req:
                    col_map[col] = req
        df = df.rename(columns=col_map)

        missing = required - set(df.columns)
        if missing:
            print(f"  [WARN] {ticker}: 컬럼 누락 {missing}")
            return None

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close", "volume"])
        df["ticker"] = ticker
        return df[["open", "high", "low", "close", "volume", "ticker"]]

    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return None


# ══════════════════════════════════════════════
#  팀원 제공 주식 데이터 로딩
# ══════════════════════════════════════════════

def load_stocks(
    min_weeks: int  = 52,
    weekly:    bool = True,
) -> dict[str, pd.DataFrame]:
    """
    data/raw/stocks/*.csv 로딩
    팀원이 준 파일 — 파일명이 ticker코드

    Returns
    -------
    {ticker: weekly_df}
    """
    csv_files = sorted(STOCKS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"  [INFO] 주식 데이터 없음: {STOCKS_DIR}")
        return {}

    print(f"\n[주식 로딩] {len(csv_files)}개 파일")
    result = {}

    for path in csv_files:
        ticker = path.stem
        daily  = _load_csv(path, ticker)
        if daily is None:
            continue

        df = to_weekly(daily) if weekly else daily
        if len(df) < min_weeks:
            print(f"  [SKIP] {ticker}: {len(df)}주 < {min_weeks}주")
            continue

        result[ticker] = df

    print(f"  로딩 성공: {len(result)}개")
    return result


# ══════════════════════════════════════════════
#  멀티에셋 로딩
# ══════════════════════════════════════════════

def load_assets(
    min_weeks: int  = 52,
    weekly:    bool = True,
) -> dict[str, pd.DataFrame]:
    """
    data/raw/assets/*.csv 로딩
    본인이 수집한 ETF / 비트코인 / 금 / 원유

    Returns
    -------
    {ticker: weekly_df}
    """
    csv_files = sorted(ASSETS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"  [INFO] 멀티에셋 데이터 없음 — collect_assets.py 먼저 실행")
        return {}

    print(f"\n[멀티에셋 로딩] {len(csv_files)}개 파일")
    result = {}

    for path in csv_files:
        ticker = path.stem
        daily  = _load_csv(path, ticker)
        if daily is None:
            continue

        df = to_weekly(daily) if weekly else daily
        if len(df) < min_weeks:
            continue

        result[ticker] = df
        print(f"  [OK] {ticker}: {len(df)}주")

    return result


# ══════════════════════════════════════════════
#  전체 통합
# ══════════════════════════════════════════════

def merge_all(
    stocks: dict[str, pd.DataFrame],
    assets: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    주식 + 멀티에셋 합치기
    같은 ticker가 있으면 assets 우선
    """
    merged = {**stocks, **assets}   # assets가 stocks를 덮어씀
    print(f"\n[통합] 전체 {len(merged)}개 자산 "
          f"(주식 {len(stocks)}개 + 멀티에셋 {len(assets)}개)")
    return merged
