# grid_atr_params_dl.py

import pandas as pd
import numpy as np

INPUT_FILE = "XAUUSD_M15_WITH_DL_SIGNALS.csv"

HORIZON = 3
ATR_COL = "atr_14_m15"

# Grid ATR & RR yang mau diuji
ATR_MULT_SL_LIST = [0.8, 1.0, 1.2]
RR_TP_LIST       = [1.2, 1.5, 2.0]

COMMISSION_PER_001 = 0.06
SPREAD_HIDDEN_USD  = 0.037

START_BALANCE = 1000.0


def simulate_trade_path(df, i, horizon, atr_mult_sl, rr_tp):
    n = len(df)
    if i + 1 >= n:
        return None

    row_decision = df.iloc[i]
    sig = int(row_decision["dl_signal"])

    if sig == 0 or np.isnan(row_decision.get(ATR_COL, np.nan)) or np.isnan(row_decision.get("dl_proba_up", np.nan)):
        return None

    direction = 1 if sig == 1 else -1

    entry_idx = i + 1
    exit_idx_time = i + horizon
    if exit_idx_time >= n:
        return None

    entry_row = df.iloc[entry_idx]
    entry_price = entry_row["open"]
    entry_time = entry_row["time"]

    atr_val = float(row_decision[ATR_COL])
    sl_dist = atr_mult_sl * atr_val
    tp_dist = rr_tp * sl_dist

    if direction == 1:
        sl_level = entry_price - sl_dist
        tp_level = entry_price + tp_dist
    else:
        sl_level = entry_price + sl_dist
        tp_level = entry_price - tp_dist

    exit_price = None
    exit_time = None
    exit_reason = None

    # cek intrabar SL/TP
    for j in range(entry_idx, exit_idx_time + 1):
        bar = df.iloc[j]
        high = bar["high"]
        low = bar["low"]

        if direction == 1:  # LONG
            if low <= sl_level:
                exit_price = sl_level
                exit_time = bar["time"]
                exit_reason = "SL"
                break
            if high >= tp_level:
                exit_price = tp_level
                exit_time = bar["time"]
                exit_reason = "TP"
                break
        else:  # SHORT
            if high >= sl_level:
                exit_price = sl_level
                exit_time = bar["time"]
                exit_reason = "SL"
                break
            if low <= tp_level:
                exit_price = tp_level
                exit_time = bar["time"]
                exit_reason = "TP"
                break

    # kalau nggak kena SL/TP, exit time-based
    if exit_price is None:
        exit_row = df.iloc[exit_idx_time]
        exit_price = exit_row["close"]
        exit_time = exit_row["time"]
        exit_reason = "TIME"

    raw_move_usd = direction * (exit_price - entry_price)
    total_cost_usd = COMMISSION_PER_001 + SPREAD_HIDDEN_USD
    net_move_usd = raw_move_usd - total_cost_usd

    ret_real = net_move_usd / entry_price

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
        "ret_real": ret_real,
    }
    return trade


def run_backtest(df_test, horizon, atr_mult_sl, rr_tp):
    df = df_test.reset_index(drop=True)
    n = len(df)
    trades = []

    i = 0
    while i < n - horizon - 1:
        row = df.iloc[i]
        sig = int(row["dl_signal"])

        if sig == 0 or np.isnan(row.get("dl_proba_up", np.nan)):
            i += 1
            continue

        trade = simulate_trade_path(df, i, horizon, atr_mult_sl, rr_tp)
        if trade is not None:
            trades.append(trade)
            exit_time = trade["exit_time"]
            exit_idx = df.index[df["time"] == exit_time][0]
            i = exit_idx + 1
        else:
            i += 1

    return pd.DataFrame(trades)


def calc_stats(trades_df, start_balance=1000.0):
    if trades_df.empty:
        return {
            "n_trades": 0,
            "winrate": np.nan,
            "avg_r": np.nan,
            "profit_factor": np.nan,
            "max_dd": np.nan,
        }

    eq = [start_balance]
    for r in trades_df["ret_real"]:
        eq.append(eq[-1] * (1.0 + r))
    equity_series = pd.Series(eq[1:])

    wins = (trades_df["ret_real"] > 0).sum()
    losses = (trades_df["ret_real"] <= 0).sum()
    n_trades = len(trades_df)
    winrate = wins / n_trades if n_trades > 0 else np.nan

    sum_pos = trades_df.loc[trades_df["ret_real"] > 0, "ret_real"].sum()
    sum_neg = trades_df.loc[trades_df["ret_real"] <= 0, "ret_real"].sum()
    profit_factor = sum_pos / abs(sum_neg) if sum_neg < 0 else np.nan

    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    max_dd = dd.min()

    avg_r = trades_df["ret_real"].mean()

    return {
        "n_trades": n_trades,
        "winrate": winrate,
        "avg_r": avg_r,
        "profit_factor": profit_factor,
        "max_dd": max_dd,
        "end_balance": float(equity_series.iloc[-1]),
    }


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df_test = df[df["set"] == "test"].copy().reset_index(drop=True)

    print("[INFO] Rows in TEST set:", len(df_test))

    rows = []
    for sl_mult in ATR_MULT_SL_LIST:
        for rr_tp in RR_TP_LIST:
            print(f"\n[INFO] Testing SL_MULT={sl_mult}, RR_TP={rr_tp} ...")
            trades_df = run_backtest(df_test, horizon=HORIZON, atr_mult_sl=sl_mult, rr_tp=rr_tp)
            stats = calc_stats(trades_df, start_balance=START_BALANCE)

            print(f"  Trades      : {stats['n_trades']}")
            print(f"  Winrate     : {stats['winrate'] * 100 if not np.isnan(stats['winrate']) else np.nan:.2f}%")
            print(f"  Avg R/trade : {stats['avg_r'] * 100 if not np.isnan(stats['avg_r']) else np.nan:.3f}%")
            print(f"  ProfitFactor: {stats['profit_factor']:.3f}")
            print(f"  Max DD      : {stats['max_dd'] * 100 if not np.isnan(stats['max_dd']) else np.nan:.2f}%")
            print(f"  End balance : {stats['end_balance']:.2f}")

            rows.append({
                "atr_mult_sl": sl_mult,
                "rr_tp": rr_tp,
                "n_trades": stats["n_trades"],
                "winrate": stats["winrate"],
                "avg_r": stats["avg_r"],
                "profit_factor": stats["profit_factor"],
                "max_dd": stats["max_dd"],
                "end_balance": stats["end_balance"],
            })

    res_df = pd.DataFrame(rows)

    # format buat print
    res_print = res_df.copy()
    res_print["winrate_pct"] = res_print["winrate"] * 100
    res_print["avg_r_pct"] = res_print["avg_r"] * 100
    res_print["max_dd_pct"] = res_print["max_dd"] * 100

    print("\n===== GRID ATR DL RESULT (TEST SET) =====")
    print(
        res_print[
            ["atr_mult_sl", "rr_tp", "n_trades",
             "winrate_pct", "avg_r_pct", "profit_factor",
             "max_dd_pct", "end_balance"]
        ].to_string(index=False,
                    formatters={
                        "winrate_pct": lambda x: f"{x:.2f}",
                        "avg_r_pct":  lambda x: f"{x:.3f}",
                        "profit_factor": lambda x: f"{x:.3f}",
                        "max_dd_pct": lambda x: f"{x:.2f}",
                        "end_balance": lambda x: f"{x:.2f}",
                    })
    )

    res_df.to_csv("dl_atr_grid_results.csv", index=False)
    print("\n[OK] Saved grid results -> dl_atr_grid_results.csv")


if __name__ == "__main__":
    main()
