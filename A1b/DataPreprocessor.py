from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd

IndexJoin = Literal["inner", "outer", "left", "right"]


@dataclass
class DataPreprocessor:
    """
    Preprocess OHLCV crypto time-series for downstream modeling.

    Assumptions:
      - Each dataframe is indexed by a DatetimeIndex (prefer tz-aware, UTC).
      - Columns include: ['open','high','low','close','volume'] (subset allowed for some methods).
    """
    data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    tz: str = "UTC"       # normalize to this timezone
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("DataPreprocessor"))

    # ---------- utilities ----------

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex")
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        if str(df.index.tz) != self.tz:
            df = df.tz_convert(self.tz)
        return df.sort_index()

    @staticmethod
    def _infer_or_use_freq(index: pd.DatetimeIndex, fallback: Optional[str] = None) -> Optional[str]:
        freq = pd.infer_freq(index)
        return freq or fallback

    # ---------- selection / alignment ----------

    def get_dataframe_in_date_range(
        self,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        freq: str,
        pairs: Optional[Iterable[str]] = None,
        missing_margin_pct: float = 50.0,
    ) -> Dict[str, pd.DataFrame]:
        """
        Reindex each chosen pair to the expected range/freq and keep those with <= missing_margin_pct missing candles.
        """
        expected = pd.date_range(start, end, freq=freq, tz=self.tz)
        selected: Dict[str, pd.DataFrame] = {}

        for pair, raw in self.data.items():
            if pairs is not None and pair not in pairs:
                continue

            df = self._ensure_datetime_index(raw)
            # align to expected index at given freq using last known value per period
            actual = df.reindex(expected).resample(freq).last()
            # % of fully-missing rows
            missing_pct = round((actual.isna().all(axis=1).sum() / len(actual)) * 100, 2)

            if missing_pct > missing_margin_pct:
                self.logger.info("Skipping %s: missing %.2f%% > %.2f%%", pair, missing_pct, missing_margin_pct)
                continue

            self.logger.info(
                "Keeping %s | expected=%d, non-missing=%d, missing=%.2f%%",
                pair, len(actual), len(actual.dropna(how="all")), missing_pct
            )
            selected[pair] = actual.dropna(how="all")

        return selected

    def align_all(
        self,
        data: Dict[str, pd.DataFrame],
        freq: Optional[str] = None,
        join: IndexJoin = "inner",
        target_index: Optional[pd.DatetimeIndex] = None,
        forward_fill: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Align multiple dataframes to a common index. If target_index is None,
        it is built by union/intersection according to `join` and optionally
        resampled to `freq`.
        """
        prepped = {k: self._ensure_datetime_index(v) for k, v in data.items()}

        if target_index is None:
            indices = [df.index for df in prepped.values()]
            if join == "outer":
                idx = indices[0].union_many(indices[1:])
            elif join == "inner":
                idx = indices[0].intersection_many(indices[1:])
            elif join == "left":
                # use first dataframe's index
                first_key = next(iter(prepped))
                idx = prepped[first_key].index
            elif join == "right":
                last_key = list(prepped.keys())[-1]
                idx = prepped[last_key].index
            else:
                raise ValueError("Invalid join")
            if freq:
                idx = pd.date_range(idx.min(), idx.max(), freq=freq, tz=self.tz)
        else:
            idx = target_index

        out: Dict[str, pd.DataFrame] = {}
        for k, df in prepped.items():
            aligned = df.reindex(idx)
            if forward_fill:
                aligned = aligned.ffill()
            out[k] = aligned
        return out

    # ---------- cleaning ----------

    def validate_and_clean(
        self,
        df: pd.DataFrame,
        freq: Optional[str] = None,
        max_gap_periods: int = 10,
        interpolate_prices: bool = True,
    ) -> pd.DataFrame:
        """
        - De-duplicate index
        - Reindex to full range at `freq` (infer if not provided)
        - Identify/flag long missing runs
        - Interpolate prices (open/high/low/close) up to `max_gap_periods`; volume is NOT interpolated
        """
        df = self._ensure_datetime_index(df)
        df = df[~df.index.duplicated(keep="first")].copy()

        use_freq = freq or self._infer_or_use_freq(df.index)
        if use_freq is None:
            raise ValueError("Could not infer frequency; please provide `freq`.")

        full_index = pd.date_range(df.index.min(), df.index.max(), freq=use_freq, tz=self.tz)
        df = df.reindex(full_index)

        # Track missing segments before filling
        missing_mask = df[['open','high','low','close']].isna().all(axis=1)
        # find contiguous runs of True
        run_id = (missing_mask != missing_mask.shift(1)).cumsum()
        run_lengths = missing_mask.groupby(run_id).transform('sum')
        long_gap_mask = missing_mask & (run_lengths > max_gap_periods)

        # Interpolation
        prices = ['open','high','low','close']
        for col in prices:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if interpolate_prices:
            df[prices] = df[prices].interpolate(method="time", limit=max_gap_periods)
            df[prices] = df[prices].ffill()

        # Volume handling: do not interpolate; fill missing with 0 if you prefer
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors="coerce").fillna(0)

        # Flag long gaps *after* filling so downstream can ignore those periods if needed
        df['long_gap_flag'] = long_gap_mask

        # Basic price sanity (conservative)
        for col in prices:
            prev = df[col].shift(1)
            df[f'{col}_outlier'] = (df[col] <= 0) | ((prev > 0) & (df[col] > prev * 5))

        return df

    # ---------- resampling ----------

    def resample_ohlcv(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Resample to OHLCV candles. Assumes prices have been cleaned/filled beforehand.
        """
        df = self._ensure_datetime_index(df)
        agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }
        if 'volume' in df.columns:
            agg['volume'] = 'sum'
        resampled = df.resample(freq).agg(agg)

        # drop periods where we still have no prices at all
        return resampled.dropna(subset=['open','high','low','close'], how='any')

    # ---------- synchronization (pairwise kept for convenience) ----------

    def synchronize_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame, how: IndexJoin = 'inner') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align two dataframes to a common index according to `how` and forward-fill.
        """
        df1 = self._ensure_datetime_index(df1)
        df2 = self._ensure_datetime_index(df2)

        if how == 'outer':
            idx = df1.index.union(df2.index)
        elif how == 'inner':
            idx = df1.index.intersection(df2.index)
        elif how == 'left':
            idx = df1.index
        elif how == 'right':
            idx = df2.index
        else:
            raise ValueError("Invalid how")
        return df1.reindex(idx).ffill(), df2.reindex(idx).ffill()

    # ---------- splitting ----------

    def split_data(
        self,
        df: pd.DataFrame,
        train_frac: Optional[float] = None,
        val_frac: Optional[float] = None,
        date_splits: Optional[Tuple[pd.Timestamp | str, pd.Timestamp | str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Either fraction-based split or date-based split.
        - Fractions: keep chronological order.
        - Dates: date_splits = (train_end, val_end) in the index timezone.
        """
        df = self._ensure_datetime_index(df)

        if date_splits is not None:
            train_end, val_end = pd.to_datetime(date_splits[0]), pd.to_datetime(date_splits[1])
            train = df.loc[:train_end]
            val = df.loc[train_end:val_end].iloc[1:]  # avoid duplicate boundary if exists
            test = df.loc[val_end:].iloc[1:]
            return train, val, test

        assert train_frac is not None and val_frac is not None, "Provide fractions or date_splits."
        assert train_frac + val_frac < 1.0, "Train + Val fractions must be < 1.0"

        n = len(df)
        train_end_i = int(n * train_frac)
        val_end_i = train_end_i + int(n * val_frac)
        return df.iloc[:train_end_i], df.iloc[train_end_i:val_end_i], df.iloc[val_end_i:]

    # ---------- features / model prep ----------

    def prepare_for_model(
        self,
        df: pd.DataFrame,
        add_features: bool = True,
        window: Optional[int] = None,
        horizon: int = 1,
        target_col: str = "close",
        log_returns: bool = True,
        clip_outliers_z: Optional[float] = 6.0,
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Basic feature set and optional windowing for seq models.
        Returns (features_df, X, y). If window is None, only features_df is meaningful.
        """
        df = self._ensure_datetime_index(df).copy()
        assert target_col in df.columns, f"{target_col} missing"

        # returns
        if log_returns:
            df['ret_1'] = np.log(df[target_col]).diff()
        else:
            df['ret_1'] = df[target_col].pct_change()

        # rolling stats
        df['vol_10'] = df['ret_1'].rolling(10).std()
        df['vol_50'] = df['ret_1'].rolling(50).std()
        df['ma_10'] = df[target_col].rolling(10).mean()
        df['ma_50'] = df[target_col].rolling(50).mean()

        # target: next-step return (classification/regression-ready)
        if log_returns:
            df['y'] = np.log(df[target_col]).shift(-horizon) - np.log(df[target_col])
        else:
            df['y'] = df[target_col].shift(-horizon) / df[target_col] - 1

        # optional outlier clipping on returns/features
        if clip_outliers_z is not None:
            for col in ['ret_1','vol_10','vol_50']:
                z = (df[col] - df[col].mean()) / (df[col].std(ddof=0) + 1e-12)
                df[col] = df[col].mask(z.abs() > clip_outliers_z, np.nan)

        df = df.dropna()

        if window is None:
            return df, None, None

        # window to 3D arrays (samples, window, features); y aligned with horizon
        feature_cols = ['ret_1','vol_10','vol_50','ma_10','ma_50']
        values = df[feature_cols].values
        y = df['y'].values

        X_list: List[np.ndarray] = []
        y_list: List[float] = []
        for i in range(len(df) - window - horizon + 1):
            X_list.append(values[i:i+window])
            y_list.append(y[i+window-1])  # predict after last window point

        X = np.stack(X_list) if X_list else np.empty((0, window, len(feature_cols)))
        y = np.array(y_list) if y_list else np.empty((0,))
        return df, X, y

    # ---------- convenience ----------

    def create_close_data(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Return wide DF of close prices per pair.
        """
        data = self.data if data is None else data
        closes = {}
        for pair, df in data.items():
            dfc = self._ensure_datetime_index(df)
            if 'close' in dfc.columns:
                closes[pair] = pd.to_numeric(dfc['close'], errors='coerce')
            else:
                self.logger.warning("`close` missing for %s; skipping.", pair)
        return pd.DataFrame(closes).sort_index()
