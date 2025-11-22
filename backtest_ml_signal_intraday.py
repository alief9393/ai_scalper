# backtest_ml_signal_intraday.py

import pandas as pd
import numpy as np

INPUT_FILE = "XAUUSD_M15_WITH_SIGNALS.csv"

# Harus sama dengan HORIZON di labeling (add_label_direction.py)
HORIZON = 3  # 3 candle M15 ≈ 45 menit

# Model SL/TP (dalam USD per XAU)
SL_USD = 3.0   # stop loss distance dari entry (contoh 3 USD)
TP_USD = 3.0   # take profit distance dari entry (contoh 3 USD, RR 1:1)

# Biaya trading (per 0.01 lot)
COMMISSION_PER_001 = 0.06   # USD per 0.01 lot
SPREAD_HIDDEN_USD  = 0.037  # USD spread efektif per trade (round-trip)

START_BALANCE = 1000.0  # equity awal (bebas)


def simulate_trade_path(df, i, horizon):
    """
    Simulasikan 1 trade:
    - Keputusan di bar i (pakai signal di bar i)
    - Entry di OPEN bar (i+1)
    - Exit:
        * kalau kena SL/TP intrabar (cek high/low setiap bar)
        * kalau tidak, exit time-based di CLOSE bar (i + horizon)
    - Return sudah dikurangi biaya (spread + komisi).

    Return:
        dict (trade info) atau None kalau tidak bisa enter (misalnya i+1/horizon out of range)
    """
    n = len(df)
    if i + 1 >= n:
        return None

    row_decision = df.iloc[i]
    sig = int(row_decision["signal"])
    if sig == 0:
        return None

    direction = 1 if sig == 1 else -1

    entry_idx = i + 1
    exit_idx_time = i + horizon
    if exit_idx_time >= n:
        return None

    entry_row = df.iloc[entry_idx]
    entry_price = entry_row["open"]
    entry_time = entry_row["time"]

    # Level SL/TP
    if direction == 1:
        sl_level = entry_price - SL_USD
        tp_level = entry_price + TP_USD
    else:  # SHORT
        sl_level = entry_price + SL_USD
        tp_level = entry_price - TP_USD

    exit_price = None
    exit_time = None
    exit_reason = None

    # Loop bar demi bar dari entry_idx sampai exit_idx_time
    for j in range(entry_idx, exit_idx_time + 1):
        bar = df.iloc[j]
        high = bar["high"]
        low = bar["low"]

        # Conservative assumption:
        # Untuk LONG: cek SL dulu (low) → baru TP (high)
        # Untuk SHORT: cek SL dulu (high) → baru TP (low)

        if direction == 1:  # LONG
            # Kena SL?
            if low <= sl_level:
                exit_price = sl_level
                exit_time = bar["time"]
                exit_reason = "SL"
                break
            # Kena TP?
            if high >= tp_level:
                exit_price = tp_level
                exit_time = bar["time"]
                exit_reason = "TP"
                break
        else:  # SHORT
            # Kena SL?
            if high >= sl_level:
                exit_price = sl_level
                exit_time = bar["time"]
                exit_reason = "SL"
                break
            # Kena TP?
            if low <= tp_level:
                exit_price = tp_level
                exit_time = bar["time"]
                exit_reason = "TP"
                break

    # Kalau tidak kena SL/TP sampai horizon -> exit time-based di CLOSE exit_idx_time
    if exit_price is None:
        exit_row = df.iloc[exit_idx_time]
        exit_price = exit_row["close"]
        exit_time = exit_row["time"]
        exit_reason = "TIME"

    # Raw price move dalam USD (per 0.01 lot, asumsi PnL ≈ price_change)
    raw_move_usd = direction * (exit_price - entry_price)

    # Biaya: komisi + spread hidden
    total_cost_usd = COMMISSION_PER_001 + SPREAD_HIDDEN_USD

    net_move_usd = raw_move_usd - total_cost_usd

    # Return sebagai persentase relatif ke entry price
    ret_real = net_move_usd / entry_price

    # "Ideal" return (tanpa biaya, pure labeling) – optional
    future_ret = row_decision.get("future_ret", np.nan)
    ret_ideal = direction * future_ret if not np.isnan(future_ret) else np.nan

    trade = {
        "decision_time": row_decision["time"],
        "entry_time": entry_time,
        "exit_time": exit_time,
        "signal": sig,
        "direction": "LONG" if direction == 1 else "SHORT",
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "raw_move_usd": raw_move_usd,
        "total_cost_usd": total_cost_usd,
        "ret_real": ret_real,
        "ret_ideal": ret_ideal,
    }
    return trade


def run_backtest(df_test, horizon=3):
    df = df_test.reset_index(drop=True)
    n = len(df)
    trades = []

    i = 0
    while i < n - horizon - 1:
        row = df.iloc[i]
        sig = int(row["signal"])

        if sig == 0:
            i += 1
            continue

        trade = simulate_trade_path(df, i, horizon=horizon)
        if trade is not None:
            trades.append(trade)
            # Loncat ke setelah exit untuk menghindari overlapping trade
            # Cari index exit_time dalam df
            exit_time = trade["exit_time"]
            # posisi index exit di df
            exit_idx = df.index[df["time"] == exit_time][0]
            i = exit_idx + 1
        else:
            i += 1

    trades_df = pd.DataFrame(trades)
    return trades_df


def calc_stats(trades_df, start_balance=1000.0):
    if trades_df.empty:
        print("[WARN] No trades generated.")
        return

    # Equity curve (sequential, 1x per trade)
    eq = [start_balance]
    for r in trades_df["ret_real"]:
        eq.append(eq[-1] * (1.0 + r))
    equity_series = pd.Series(eq[1:])  # drop initial

    # Winrate global
    wins = (trades_df["ret_real"] > 0).sum()
    losses = (trades_df["ret_real"] <= 0).sum()
    n_trades = len(trades_df)
    winrate = wins / n_trades if n_trades > 0 else np.nan

    # Long / Short stats
    longs = trades_df[trades_df["signal"] == 1]
    shorts = trades_df[trades_df["signal"] == -1]

    def side_stats(sub):
        if len(sub) == 0:
            return np.nan, np.nan
        wr = (sub["ret_real"] > 0).mean()
        avg_r = sub["ret_real"].mean()
        return wr, avg_r

    wr_long, avg_long = side_stats(longs)
    wr_short, avg_short = side_stats(shorts)

    # Profit factor: sum(win) / abs(sum(loss))
    sum_pos = trades_df.loc[trades_df["ret_real"] > 0, "ret_real"].sum()
    sum_neg = trades_df.loc[trades_df["ret_real"] <= 0, "ret_real"].sum()
    profit_factor = sum_pos / abs(sum_neg) if sum_neg < 0 else np.nan

    # Max drawdown
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    max_dd = dd.min()  # negatif

    print("===== ML SIGNAL BACKTEST (TEST SET, SL/TP + COST) =====")
    print(f"Trades           : {n_trades}")
    print(f"Winrate overall  : {winrate * 100:.2f}%")
    print(f"Avg R per trade  : {trades_df['ret_real'].mean() * 100:.3f}%")
    print(f"Profit factor    : {profit_factor:.3f}")
    print(f"Max drawdown     : {max_dd * 100:.2f}%")
    print()
    print(f"Long trades      : {len(longs)}")
    print(f"  Winrate long   : {wr_long * 100:.2f}%")
    print(f"  Avg R long     : {avg_long * 100:.3f}%")
    print()
    print(f"Short trades     : {len(shorts)}")
    print(f"  Winrate short  : {wr_short * 100:.2f}%")
    print(f"  Avg R short    : {avg_short * 100:.3f}%")
    print()
    print(f"Start balance    : {start_balance:.2f}")
    print(f"End balance      : {equity_series.iloc[-1]:.2f}")
    print("=========================================================")


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Pakai hanya TEST set (out-of-sample)
    df_test = df[df["set"] == "test"].copy().reset_index(drop=True)

    print("[INFO] Rows in TEST set:", len(df_test))

    trades_df = run_backtest(df_test, horizon=HORIZON)
    print("[INFO] Generated trades:", len(trades_df))

    # Simpan detail trade untuk analisa lebih lanjut
    trades_df.to_csv("ml_signal_trades_test_sl_tp_cost.csv", index=False)
    print("[OK] Saved trades -> ml_signal_trades_test_sl_tp_cost.csv")

    # Hitung statistik
    calc_stats(trades_df, start_balance=START_BALANCE)


if __name__ == "__main__":
    main()
