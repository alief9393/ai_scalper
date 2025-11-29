# walkforward_backtest.py

import argparse
from datetime import timedelta

import numpy as np
import pandas as pd
import pandas_ta as ta  # 🔥 tambahin ini

# Import fungsi & config dari backtest utama
from backtest_dl_signal_atr import (
    run_backtest,
    HORIZON,
    INPUT_FILE,
    ATR_COL,
    ATR_MA_PERIOD,
    ADX_COL,
    USE_REGIME_FILTERS,
    USE_MTF_FILTER,
    MTF_TREND_COL,
    H1_EMA_FAST,
    H1_EMA_SLOW,
)

def compute_stats(trades_df: pd.DataFrame, start_balance: float):
    """
    Hitung stats basic untuk satu set trade.
    Mirip calc_stats() tapi dalam bentuk dict (biar bisa dipakai per window & total).
    """
    if trades_df.empty:
        return {
            "n_trades": 0,
            "winrate": np.nan,
            "avg_r": np.nan,
            "profit_factor": np.nan,
            "max_dd": np.nan,
            "end_balance": start_balance,
        }

    equity_series = pd.Series(
        [start_balance] + trades_df["balance_after"].tolist()
    )

    wins = (trades_df["net_pnl_usd"] > 0).sum()
    losses = (trades_df["net_pnl_usd"] <= 0).sum()
    n_trades = len(trades_df)
    winrate = wins / n_trades if n_trades > 0 else np.nan

    sum_pos = trades_df.loc[trades_df["net_pnl_usd"] > 0, "net_pnl_usd"].sum()
    sum_neg = trades_df.loc[trades_df["net_pnl_usd"] < 0, "net_pnl_usd"].sum()
    profit_factor = sum_pos / abs(sum_neg) if sum_neg < 0 else np.nan

    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    max_dd = dd.min()

    end_balance = equity_series.iloc[-1]

    avg_r = trades_df["ret_real"].mean()

    return {
        "n_trades": int(n_trades),
        "winrate": winrate,
        "avg_r": avg_r,
        "profit_factor": profit_factor,
        "max_dd": max_dd,
        "end_balance": float(end_balance),
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward DL ATR backtest (super realistis).")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYY-MM-DD). Default: min time in chosen set")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (YYYY-MM-DD). Default: max time in chosen set")
    parser.add_argument("--balance", type=float, default=100.0,
                        help="Starting balance")
    parser.add_argument("--set", type=str, default="test",
                        choices=["train", "val", "test", "all"],
                        help="Subset berdasarkan kolom 'set'. Default: test (pure OOS).")
    parser.add_argument("--window_days", type=int, default=14,
                        help="Panjang 1 window walk-forward dalam hari (default: 14)")
    args = parser.parse_args()

    # ===== Load data =====
    df = pd.read_csv(INPUT_FILE, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    if "set" not in df.columns:
        raise ValueError("Kolom 'set' tidak ditemukan di CSV. Pastikan INPUT_FILE benar.")

    # 🔥 Hitung ADX & ATR_MA kalau regime filter aktif
    if USE_REGIME_FILTERS:
        # ADX
        if ADX_COL not in df.columns:
            print(f"[INFO] Kolom ADX '{ADX_COL}' belum ada di walkforward DF, menghitung dengan pandas_ta...")
            adx = ta.adx(df["high"], df["low"], df["close"], length=14)
            # ambil kolom nama sama persis ADX_COL (harusnya "ADX_14")
            if ADX_COL in adx.columns:
                df[ADX_COL] = adx[ADX_COL]
            else:
                # fallback: cari kolom yang mengandung "ADX"
                adx_cols = [c for c in adx.columns if "ADX" in c.upper()]
                if not adx_cols:
                    raise ValueError("Gagal menemukan kolom ADX dari pandas_ta.adx()")
                df[ADX_COL] = adx[adx_cols[0]]

        # ATR MA utk regime
        if ATR_COL not in df.columns:
            raise ValueError(f"Kolom ATR '{ATR_COL}' tidak ditemukan di DF.")
        df["atr_ma_50"] = df[ATR_COL].rolling(ATR_MA_PERIOD).mean()

    if USE_MTF_FILTER:
        df_h1 = df.resample("1H", on="time").agg({"close": "last"}).dropna()

        df_h1["ema_fast"] = df_h1["close"].ewm(span=H1_EMA_FAST, adjust=False).mean()
        df_h1["ema_slow"] = df_h1["close"].ewm(span=H1_EMA_SLOW, adjust=False).mean()
        diff = df_h1["ema_fast"] - df_h1["ema_slow"]

        df_h1[MTF_TREND_COL] = 0
        df_h1.loc[diff > 0, MTF_TREND_COL] = 1
        df_h1.loc[diff < 0, MTF_TREND_COL] = -1

        df["time_h1"] = df["time"].dt.floor("H")
        df = df.merge(
            df_h1[[MTF_TREND_COL]],
            left_on="time_h1",
            right_index=True,
            how="left",
        )

    # Setelah indikator beres, baru filter set
    if args.set != "all":
        df = df[df["set"] == args.set].copy().reset_index(drop=True)

        if df.empty:
            print("[ERR] Tidak ada data setelah filter set.")
            return

    # Filter by date if provided
    if args.start is not None:
        start_dt = pd.to_datetime(args.start)
        df = df[df["time"] >= start_dt]
    if args.end is not None:
        end_dt = pd.to_datetime(args.end)
        df = df[df["time"] <= end_dt]

    if df.empty:
        print("[ERR] Tidak ada data setelah filter tanggal.")
        return

    overall_start = df["time"].min()
    overall_end = df["time"].max()

    print("===== WALK-FORWARD CONFIG =====")
    print(f"Set          : {args.set}")
    print(f"Data range   : {overall_start}  ->  {overall_end}")
    print(f"Window (hari): {args.window_days}")
    print(f"Start balance: {args.balance}")
    print("================================\n")

    # ===== Walk-forward loop =====
    window_start = overall_start if args.start is None else pd.to_datetime(args.start)
    window_end_global = overall_end if args.end is None else pd.to_datetime(args.end)

    balance = args.balance
    all_trades = []
    window_summaries = []

    wf_idx = 0

    while window_start < window_end_global:
        wf_idx += 1
        window_end = window_start + timedelta(days=args.window_days)
        if window_end > window_end_global:
            window_end = window_end_global

        df_win = df[(df["time"] >= window_start) & (df["time"] <= window_end)].copy()

        if len(df_win) < (HORIZON + 5):
            # Terlalu sedikit bar untuk jalanin trading horizon
            print(f"[WF-{wf_idx}] {window_start} -> {window_end} | SKIP (bars: {len(df_win)})")
            window_start = window_end
            continue

        print(f"[WF-{wf_idx}] {window_start} -> {window_end} | bars: {len(df_win)} | balance_in: {balance:.2f}")

        trades_df, mc = run_backtest(df_win, horizon=HORIZON, start_balance=balance)

        if trades_df.empty:
            print(f"  [WF-{wf_idx}] No trades in this window.")
            window_summaries.append({
                "wf_id": wf_idx,
                "start": window_start,
                "end": window_end,
                "n_trades": 0,
                "winrate": np.nan,
                "avg_r": np.nan,
                "profit_factor": np.nan,
                "max_dd": np.nan,
                "balance_in": balance,
                "balance_out": balance,
                "margin_call": mc,
            })
            window_start = window_end
            continue

        # Tambah meta kolom
        trades_df["wf_id"] = wf_idx
        trades_df["wf_start"] = window_start
        trades_df["wf_end"] = window_end

        # Update balance ke akhir window
        balance_out = float(trades_df["balance_after"].iloc[-1])

        stats = compute_stats(trades_df, start_balance=trades_df["balance_before"].iloc[0])

        window_summaries.append({
            "wf_id": wf_idx,
            "start": window_start,
            "end": window_end,
            "n_trades": stats["n_trades"],
            "winrate": stats["winrate"],
            "avg_r": stats["avg_r"],
            "profit_factor": stats["profit_factor"],
            "max_dd": stats["max_dd"],
            "balance_in": trades_df["balance_before"].iloc[0],
            "balance_out": balance_out,
            "margin_call": mc or trades_df["is_margin_call"].any(),
        })

        print(
            f"  Trades: {stats['n_trades']}, "
            f"WR: {stats['winrate']*100:.2f}%, "
            f"PF: {stats['profit_factor']:.2f}, "
            f"DD: {stats['max_dd']*100:.2f}%, "
            f"Bal: {trades_df['balance_before'].iloc[0]:.2f} -> {balance_out:.2f}"
        )

        all_trades.append(trades_df)
        balance = balance_out
        window_start = window_end

    if not all_trades:
        print("[WARN] Tidak ada trade di semua window.")
        return

    all_trades_df = pd.concat(all_trades, ignore_index=True).sort_values("decision_time")
    all_trades_df.to_csv("dl_signal_trades_walkforward.csv", index=False)
    print("\n[OK] Saved walk-forward trades -> dl_signal_trades_walkforward.csv")

    # ===== Summary per window =====
    ws_df = pd.DataFrame(window_summaries)
    print("\n===== WALK-FORWARD WINDOW SUMMARY =====")
    for _, row in ws_df.iterrows():
        print(
            f"WF-{int(row['wf_id'])}: {row['start']} -> {row['end']} | "
            f"Trades: {int(row['n_trades'])} | "
            f"WR: {0 if np.isnan(row['winrate']) else row['winrate']*100:.2f}% | "
            f"PF: {row['profit_factor'] if not np.isnan(row['profit_factor']) else float('nan'):.2f} | "
            f"DD: {row['max_dd']*100 if not np.isnan(row['max_dd']) else float('nan'):.2f}% | "
            f"Bal: {row['balance_in']:.2f} -> {row['balance_out']:.2f}"
        )
    print("========================================\n")

    # ===== Global stats =====
    global_stats = compute_stats(all_trades_df, start_balance=args.balance)
    print("===== GLOBAL WALK-FORWARD STATS =====")
    print(f"Total trades       : {global_stats['n_trades']}")
    print(f"Winrate overall    : {global_stats['winrate']*100:.2f}%")
    print(f"Avg R per trade    : {global_stats['avg_r']*100:.3f}%")
    print(f"Profit factor      : {global_stats['profit_factor']:.3f}")
    print(f"Max drawdown       : {global_stats['max_dd']*100:.2f}%")
    print(f"Start balance      : {args.balance:.2f}")
    print(f"End balance        : {global_stats['end_balance']:.2f}")
    print("======================================")


if __name__ == "__main__":
    main()
