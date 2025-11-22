import pandas as pd
import numpy as np
import argparse

# ================== CONFIG ==================
INPUT_FILE = "XAUUSD_M15_WITH_DL_SIGNALS.csv"

HORIZON = 3  # sama seperti labeling

ATR_COL = "atr_14_m15"
ATR_MULT_SL = 1.2      # SL = ATR_MULT_SL * ATR
RR_TP = 2.0            # TP = RR_TP * SL

# Biaya per 0.01 lot (round-trip)
COMMISSION_PER_001 = 0.06   # USD per 0.01 lot
SPREAD_HIDDEN_USD  = 0.037  # USD per 0.01 lot

# Akun & risk
START_BALANCE = 100.0
RISK_PER_TRADE = 0.01       # 1% dari balance per trade

# Market model XAUUSD
CONTRACT_SIZE = 100.0       # 1 lot = 100 oz (umum di broker XAUUSD)
LEVERAGE = 500              # leverage akun
STOP_OUT_LEVEL = 0.2        # margin call jika Equity <= 20% dari margin

# Batas lot
MIN_LOT = 0.01
MAX_LOT = 0.10


def simulate_trade_path(df, i, horizon, balance):
    """
    Simulasi 1 trade:
    - Hitung lot berdasar risk % & SL (ATR)
    - Hitung margin
    - Simulasi bar-by-bar sampai SL/TP/TIME/MARGIN_CALL
    - Update balance
    """
    n = len(df)
    row_decision = df.iloc[i]

    sig = int(row_decision.get("dl_signal", 0))
    proba_up = row_decision.get("dl_proba_up", np.nan)
    atr_val = row_decision.get(ATR_COL, np.nan)

    # Skip kalau:
    # - tidak ada sinyal
    # - proba belum ada
    # - ATR NaN
    if (
        sig == 0
        or np.isnan(proba_up)
        or np.isnan(atr_val)
        or i + 1 >= n
    ):
        return None, balance, i + 1, False

    direction = 1 if sig == 1 else -1  # 1 = LONG, -1 = SHORT

    entry_idx = i + 1
    exit_idx_time = i + horizon
    if exit_idx_time >= n:
        return None, balance, i + 1, False

    entry_row = df.iloc[entry_idx]
    entry_price = float(entry_row["open"])
    entry_time = entry_row["time"]

    atr_val = float(atr_val)
    sl_dist = ATR_MULT_SL * atr_val
    tp_dist = RR_TP * sl_dist

    if sl_dist <= 0:
        return None, balance, i + 1, False

    # ========= LOT SIZING (risk % dari balance) =========
    # Risk dalam USD
    risk_amount = balance * RISK_PER_TRADE
    # PnL per 1 lot jika kena SL:
    # (sl_dist * CONTRACT_SIZE * lot)
    # => lot = risk_amount / (sl_dist * CONTRACT_SIZE)
    lot = risk_amount / (sl_dist * CONTRACT_SIZE)

    # Clamp lot
    lot = max(lot, MIN_LOT)
    lot = min(lot, MAX_LOT)

    if lot <= 0:
        return None, balance, i + 1, False

    # ========= Margin requirement =========
    margin_required = entry_price * CONTRACT_SIZE * lot / LEVERAGE

    # Kalau margin nggak cukup, skip sinyal ini
    if margin_required > balance:
        return None, balance, i + 1, False

    # ========= Hitung level SL / TP =========
    if direction == 1:  # LONG
        sl_level = entry_price - sl_dist
        tp_level = entry_price + tp_dist
    else:  # SHORT
        sl_level = entry_price + sl_dist
        tp_level = entry_price - tp_dist

    exit_price = None
    exit_time = None
    exit_reason = None
    margin_call = False
    exit_idx = entry_idx

    # ========= Simulasi bar-by-bar =========
    for j in range(entry_idx, exit_idx_time + 1):
        bar = df.iloc[j]
        high = float(bar["high"])
        low = float(bar["low"])

        # --- Cek kemungkinan margin call (floating loss terburuk di bar ini) ---
        if direction == 1:
            worst_price = low
        else:
            worst_price = high

        float_pnl_worst = (worst_price - entry_price) * direction * CONTRACT_SIZE * lot
        equity_worst = balance + float_pnl_worst  # diasumsikan hanya 1 posisi open

        # Margin call: equity <= STOP_OUT_LEVEL * margin_required
        if equity_worst <= STOP_OUT_LEVEL * margin_required:
            exit_price = worst_price
            exit_time = bar["time"]
            exit_reason = "MARGIN_CALL"
            margin_call = True
            exit_idx = j
            break

        # --- Kalau belum margin call, cek SL/TP ---
        if direction == 1:  # LONG
            if low <= sl_level:
                exit_price = sl_level
                exit_time = bar["time"]
                exit_reason = "SL"
                exit_idx = j
                break
            if high >= tp_level:
                exit_price = tp_level
                exit_time = bar["time"]
                exit_reason = "TP"
                exit_idx = j
                break
        else:  # SHORT
            if high >= sl_level:
                exit_price = sl_level
                exit_time = bar["time"]
                exit_reason = "SL"
                exit_idx = j
                break
            if low <= tp_level:
                exit_price = tp_level
                exit_time = bar["time"]
                exit_reason = "TP"
                exit_idx = j
                break

    # Kalau tidak SL/TP/MARGIN_CALL sampai horizon habis -> keluar di close horizon
    if exit_price is None:
        exit_row = df.iloc[exit_idx_time]
        exit_price = float(exit_row["close"])
        exit_time = exit_row["time"]
        exit_reason = "TIME"
        exit_idx = exit_idx_time

    # ========= Hitung PnL (dalam USD) + biaya =========
    price_move = (exit_price - entry_price) * direction
    gross_pnl_usd = price_move * CONTRACT_SIZE * lot

    # Biaya total per 0.01 lot (komisi + spread)
    cost_per_001 = COMMISSION_PER_001 + SPREAD_HIDDEN_USD
    total_cost_usd = cost_per_001 * (lot / 0.01)

    net_pnl_usd = gross_pnl_usd - total_cost_usd

    balance_before = balance
    balance_after = balance_before + net_pnl_usd
    if balance_after < 0:
        balance_after = 0.0

    ret_real = net_pnl_usd / balance_before if balance_before > 0 else 0.0

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
        "atr_entry": atr_val,
        "sl_dist_usd": sl_dist,
        "tp_dist_usd": tp_dist,
        "lot": lot,
        "margin_required": margin_required,
        "gross_pnl_usd": gross_pnl_usd,
        "trade_cost_usd": total_cost_usd,
        "net_pnl_usd": net_pnl_usd,
        "ret_real": ret_real,        # return % vs balance_before
        "ret_ideal": ret_ideal,
        "balance_before": balance_before,
        "balance_after": balance_after,
        "is_margin_call": margin_call,
    }

    return trade, balance_after, exit_idx + 1, margin_call


def run_backtest(df_test, horizon, start_balance):
    df = df_test.reset_index(drop=True)
    n = len(df)
    trades = []

    balance = start_balance
    i = 0
    margin_call_happened = False

    while i < n - horizon - 1 and balance > 0:
        trade, balance, next_i, mc = simulate_trade_path(df, i, horizon, balance)

        if trade is not None:
            trades.append(trade)

        if mc:
            margin_call_happened = True
            i = next_i
            break

        i = next_i

    trades_df = pd.DataFrame(trades)
    return trades_df, margin_call_happened


def calc_stats(trades_df, start_balance=100.0, margin_call_happened=False):
    if trades_df.empty:
        print("[WARN] No trades generated.")
        return

    # Equity curve dari balance_after
    equity_series = pd.Series(
        [start_balance] + trades_df["balance_after"].tolist()
    )

    wins = (trades_df["net_pnl_usd"] > 0).sum()
    losses = (trades_df["net_pnl_usd"] <= 0).sum()
    n_trades = len(trades_df)
    winrate = wins / n_trades if n_trades > 0 else np.nan

    longs = trades_df[trades_df["signal"] == 1]
    shorts = trades_df[trades_df["signal"] == -1]

    def side_stats(sub):
        if len(sub) == 0:
            return np.nan, np.nan
        wr = (sub["net_pnl_usd"] > 0).mean()
        avg_r = sub["ret_real"].mean()
        return wr, avg_r

    wr_long, avg_long = side_stats(longs)
    wr_short, avg_short = side_stats(shorts)

    sum_pos = trades_df.loc[trades_df["net_pnl_usd"] > 0, "net_pnl_usd"].sum()
    sum_neg = trades_df.loc[trades_df["net_pnl_usd"] < 0, "net_pnl_usd"].sum()
    profit_factor = sum_pos / abs(sum_neg) if sum_neg < 0 else np.nan

    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    max_dd = dd.min()

    final_balance = equity_series.iloc[-1]

    avg_lot = trades_df["lot"].mean()
    max_lot = trades_df["lot"].max()

    print("===== DL (LSTM) ATR SL/TP BACKTEST (SUPER REAL) =====")
    print(f"Trades             : {n_trades}")
    print(f"Winrate overall    : {winrate * 100:.2f}%")
    print(f"Avg R per trade    : {trades_df['ret_real'].mean() * 100:.3f}%")
    print(f"Profit factor      : {profit_factor:.3f}")
    print(f"Max drawdown       : {max_dd * 100:.2f}%")
    print()
    print(f"Long trades        : {len(longs)}")
    print(f"  Winrate long     : {wr_long * 100:.2f}%")
    print(f"  Avg R long       : {avg_long * 100:.3f}%")
    print()
    print(f"Short trades       : {len(shorts)}")
    print(f"  Winrate short    : {wr_short * 100:.2f}%")
    print(f"  Avg R short      : {avg_short * 100:.3f}%")
    print()
    print(f"Avg lot            : {avg_lot:.3f}")
    print(f"Max lot            : {max_lot:.3f}")
    print()
    print(f"Start balance      : {start_balance:.2f}")
    print(f"End balance        : {final_balance:.2f}")
    if margin_call_happened or trades_df["is_margin_call"].any():
        last_mc = trades_df[trades_df["is_margin_call"]].iloc[-1]
        print(">>> MARGIN CALL terjadi pada:", last_mc["exit_time"])
        print("    Balance setelah MC :", last_mc['balance_after'])
    print("=====================================================")


def main():
    parser = argparse.ArgumentParser(description="Super-real DL ATR backtest with lot & margin.")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--balance", type=float, default=START_BALANCE, help="Starting balance")
    args = parser.parse_args()

    df = pd.read_csv(INPUT_FILE, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Pakai hanya TEST set (sesuai pipeline sebelumnya)
    df_test = df.copy().reset_index(drop=True)

    # Filter date range kalau ada
    if args.start is not None:
        start_dt = pd.to_datetime(args.start)
        df_test = df_test[df_test["time"] >= start_dt]
    if args.end is not None:
        end_dt = pd.to_datetime(args.end)
        df_test = df_test[df_test["time"] <= end_dt]

    print("[INFO] Rows in TEST set (after date filter):", len(df_test))
    
    if df_test.empty:
        print("[WARN] Tidak ada data di range tanggal ini.")
        return

    trades_df, margin_call_happened = run_backtest(df_test, horizon=HORIZON, start_balance=args.balance)
    print("[INFO] Generated trades:", len(trades_df))

    trades_df.to_csv("dl_signal_trades_superreal_atr_sl_tp_cost.csv", index=False)
    print("[OK] Saved trades -> dl_signal_trades_superreal_atr_sl_tp_cost.csv")

    calc_stats(trades_df, start_balance=args.balance, margin_call_happened=margin_call_happened)


if __name__ == "__main__":
    main()
