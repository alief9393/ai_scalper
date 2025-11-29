import pandas as pd
import numpy as np
import argparse
import pandas_ta as ta

# ================== CONFIG ==================
INPUT_FILE = "XAUUSD_M15_WITH_HYBRID_SIGNALS_FULL.csv"

HORIZON = 3  # sama seperti labeling

ATR_COL = "atr_14_m15"
ATR_MULT_SL = 1.2
RR_TP = 2.0  # fallback kalau dynamic TP dimatikan

# (Confidence filter per-model TIDAK dipakai lagi, kita pakai Hybrid Confidence)
USE_PROBA_CONF_FILTER = False
PROBA_CONF_MARGIN = 0.25  # kalau mau dipakai lagi

# Dynamic TP (RR tergantung ADX + MTF trend, lalu dimodif oleh Hybrid Strength)
USE_DYNAMIC_TP = True
RR_TP_BASE = 2.0
ADX_TP_MED = 20.0
ADX_TP_STRONG = 25.0
RR_TP_MED_TREND = 2.5
RR_TP_STRONG_TREND = 3.0

# Biaya per 0.01 lot (round-trip)
COMMISSION_PER_001 = 0.06
SPREAD_HIDDEN_USD = 0.037

# Akun & risk
START_BALANCE = 100.0
RISK_PER_TRADE = 0.01  # fallback jika dynamic risk dimatikan

# Dynamic risk (persentase risk per trade adaptif terhadap ADX + drawdown)
USE_DYNAMIC_RISK = True
BASE_RISK_PCT = 0.0200  # 2.0% base (sebelum throttle)
MIN_RISK_PCT = 0.05     # 10%
MAX_RISK_PCT = 0.10     # 15%

# Market model XAUUSD
CONTRACT_SIZE = 100.0
LEVERAGE = 500
STOP_OUT_LEVEL = 0.2

# Batas lot
MIN_LOT = 0.02
MAX_LOT = 0.60

# Regime filters
USE_REGIME_FILTERS = True
ATR_MA_PERIOD = 50
ATR_MIN_MULT = 1.0

ADX_COL = "ADX_14"
ADX_MIN = 15.0

# Session filter
USE_SESSION_FILTER = True
SESSION_START_HOUR = 7
SESSION_END_HOUR = 22

# MTF trend filter
USE_MTF_FILTER = True
MTF_TREND_COL = "trend_h1_dir"
H1_EMA_FAST = 50
H1_EMA_SLOW = 200

# ======== PIP & SLIPPAGE MODEL (EXNESS RULE) ========
PIP_SIZE = 0.01  # 1 pip XAUUSD = 0.01 harga

# nilai pip per 0.01 lot
PIP_VALUE_PER_001 = PIP_SIZE * CONTRACT_SIZE * 0.01  # = 0.01 USD/pip (0.01 lot)

# spread dalam pip, berdasarkan cost spread broker
SPREAD_PIPS = SPREAD_HIDDEN_USD / PIP_VALUE_PER_001  # ~3.7 pip

# rule exness: slippage-free 0 – 3x spread, kita ambil expected ~0.5 dari max
MAX_SLIPPAGE_PIPS = 3.0 * SPREAD_PIPS         # ~11 pip
AVG_SLIPPAGE_PIPS = 0.5 * MAX_SLIPPAGE_PIPS   # ~5.5 pip (dipakai buat backtest)


# ============ SESSION & REGIME ============

def in_liquid_session(ts: pd.Timestamp) -> bool:
    if pd.isna(ts):
        return False
    h = ts.hour
    return SESSION_START_HOUR <= h < SESSION_END_HOUR


def passes_regime_filters(row: pd.Series) -> bool:
    if not USE_REGIME_FILTERS:
        return True

    atr = row.get(ATR_COL, np.nan)
    atr_ma = row.get("atr_ma_50", np.nan)
    if np.isnan(atr) or np.isnan(atr_ma):
        return False
    if atr < ATR_MIN_MULT * atr_ma:
        return False

    if ADX_COL not in row.index:
        raise KeyError(f"Kolom ADX '{ADX_COL}' tidak ada di row/index DF.")
    adx_val = row.get(ADX_COL, np.nan)
    if np.isnan(adx_val) or adx_val < ADX_MIN:
        return False

    if USE_SESSION_FILTER:
        ts = row.get("time", pd.NaT)
        if not in_liquid_session(ts):
            return False

    return True


# ============ HYBRID CONFIDENCE ============

def get_hybrid_conf_group(row: pd.Series, direction: int):
    """
    Hitung strength & confidence group berdasarkan dl_proba_up + xgb_proba_up.
    direction: 1 (LONG) atau -1 (SHORT)
    Return: (group, risk_mult, rr_mult)
    """
    dl_p = row.get("dl_proba_up", np.nan)
    xgb_p = row.get("xgb_proba_up", np.nan)

    if np.isnan(dl_p) or np.isnan(xgb_p):
        return "UNKNOWN", 1.0, 1.0

    if direction == 1:
        dl_edge = dl_p - 0.5
        xgb_edge = xgb_p - 0.5
    else:
        dl_edge = 0.5 - dl_p
        xgb_edge = 0.5 - xgb_p

    # Kalau salah satu model sebenarnya kontra arah -> DISAGREE
    if dl_edge < 0 or xgb_edge < 0:
        return "DISAGREE", 0.0, 0.0

    # Strength = rata-rata edge dua model
    strength = 0.5 * dl_edge + 0.5 * xgb_edge

    if strength >= 0.25:
        # Keduanya cukup jauh dari 0.5 (~>= 0.75)
        return "A_STRONG", 1.5, 1.15
    elif strength >= 0.15:
        # Confidence moderat (~>= 0.65)
        return "B_MODERATE", 1.0, 1.00
    else:
        return "C_WEAK", 0.0, 0.0


# ============ RR & RISK CORE (ADX + DD) ============

def get_rr_tp_for_row(row: pd.Series, direction: int) -> float:
    """
    RR base berdasarkan ADX + MTF trend.
    (Nanti masih dikali lagi oleh hybrid RR multiplier.)
    """
    if not USE_DYNAMIC_TP:
        return RR_TP

    rr = RR_TP_BASE
    adx_val = row.get(ADX_COL, np.nan)
    trend_dir = row.get(MTF_TREND_COL, 0) if USE_MTF_FILTER else 0

    if not np.isnan(adx_val):
        same_trend = (
            not pd.isna(trend_dir)
            and trend_dir != 0
            and int(trend_dir) == direction
        )
        if same_trend:
            if adx_val >= ADX_TP_STRONG:
                rr = RR_TP_STRONG_TREND
            elif adx_val >= ADX_TP_MED:
                rr = RR_TP_MED_TREND

    return rr


def get_risk_pct_core(row: pd.Series, dd_frac: float | None) -> float:
    """
    Risk core berdasarkan ADX + drawdown (tanpa hybrid multiplier).
    """
    if not USE_DYNAMIC_RISK:
        return RISK_PER_TRADE

    adx_val = row.get(ADX_COL, np.nan)
    if np.isnan(adx_val):
        risk_core = RISK_PER_TRADE
    else:
        if adx_val < ADX_MIN:
            risk_core = MIN_RISK_PCT
        elif adx_val < 25:
            mult = 0.8
            risk_core = BASE_RISK_PCT * mult
        elif adx_val < 35:
            mult = 1.0
            risk_core = BASE_RISK_PCT * mult
        else:
            mult = 1.2
            risk_core = BASE_RISK_PCT * mult

    # Clamp awal ke MIN/MAX
    risk_core = max(risk_core, MIN_RISK_PCT)
    risk_core = min(risk_core, MAX_RISK_PCT)

    # Drawdown-based throttle
    if dd_frac is None:
        dd_factor = 1.0
    else:
        if dd_frac >= -0.03:        # DD < 3% → equity sehat, gas dikit
            dd_factor = 1.1
        elif dd_frac >= -0.07:      # 3–7% → defensive ringan
            dd_factor = 0.9
        elif dd_frac >= -0.15:      # 7–15%
            dd_factor = 0.6
        elif dd_frac >= -0.25:      # 15–25%
            dd_factor = 0.3
        else:                       # >25%
            dd_factor = 0.0

    risk = risk_core * dd_factor
    risk = max(risk, 0.0)
    risk = min(risk, MAX_RISK_PCT)
    return risk


# ============ TRADE SIMULATION ============

def simulate_trade_path(
    df: pd.DataFrame,
    i: int,
    horizon: int,
    balance: float,
    peak_balance: float,
):
    n = len(df)
    row_decision = df.iloc[i]

    # Regime filter dulu
    if not passes_regime_filters(row_decision):
        return None, balance, i + 1, False

    # Gunakan hybrid_signal sebagai arah final
    sig = int(row_decision.get("hybrid_signal", 0))
    if sig == 0 or i + 1 >= n:
        return None, balance, i + 1, False

    atr_val = row_decision.get(ATR_COL, np.nan)
    if np.isnan(atr_val):
        return None, balance, i + 1, False

    # Arah trade
    direction = 1 if sig == 1 else -1

    # 🔥 HYBRID CONFIDENCE FILTER + RISK/RR MULT
    conf_group, risk_mult, rr_mult = get_hybrid_conf_group(row_decision, direction)

    if conf_group in ("DISAGREE", "C_WEAK"):
        # lemah / beda arah -> skip
        return None, balance, i + 1, False

    # Entry index = NEXT BAR -> pure out-of-sample entry
    entry_idx = i + 1
    exit_idx_time = i + horizon
    if exit_idx_time >= n:
        return None, balance, i + 1, False

    entry_row = df.iloc[entry_idx]
    entry_price_raw = float(entry_row["open"])
    entry_time = entry_row["time"]

    # === APPLY SLIPPAGE DI ENTRY (market execution) ===
    if direction == 1:
        # BUY --> harga sedikit lebih buruk (lebih tinggi)
        entry_price = entry_price_raw + AVG_SLIPPAGE_PIPS * PIP_SIZE
    else:
        # SELL --> harga sedikit lebih buruk (lebih rendah)
        entry_price = entry_price_raw - AVG_SLIPPAGE_PIPS * PIP_SIZE

    atr_val = float(atr_val)
    sl_dist = ATR_MULT_SL * atr_val
    if sl_dist <= 0:
        return None, balance, i + 1, False

    # RR base dari ADX + MTF trend
    rr_tp_base = get_rr_tp_for_row(row_decision, direction)
    # Modif oleh Hybrid Strength
    rr_tp = rr_tp_base * rr_mult
    # Clamp RR biar nggak absurd
    rr_tp = max(rr_tp, 1.0)
    rr_tp = min(rr_tp, 4.0)

    tp_dist = rr_tp * sl_dist

    # ==== Drawdown sekarang (sebelum trade) ====
    if peak_balance > 0:
        dd_frac = (balance - peak_balance) / peak_balance
    else:
        dd_frac = 0.0

    # ==== Risk% dinamis (ADX + DD) lalu dikali Hybrid risk_mult ====
    risk_core = get_risk_pct_core(row_decision, dd_frac)
    risk_pct = risk_core * risk_mult
    risk_pct = min(risk_pct, MAX_RISK_PCT)
    risk_pct = max(risk_pct, 0.0)

    if risk_pct <= 0:
        return None, balance, i + 1, False

    risk_amount = balance * risk_pct

    # Lot sizing (risk / SL)
    lot = risk_amount / (sl_dist * CONTRACT_SIZE)
    lot = max(lot, MIN_LOT)
    lot = min(lot, MAX_LOT)

    if lot <= 0:
        return None, balance, i + 1, False

    margin_required = entry_price * CONTRACT_SIZE * lot / LEVERAGE
    if margin_required > balance:
        return None, balance, i + 1, False

    # Level SL/TP
    if direction == 1:
        sl_level = entry_price - sl_dist
        tp_level = entry_price + tp_dist
    else:
        sl_level = entry_price + sl_dist
        tp_level = entry_price - tp_dist

    exit_price = None
    exit_time = None
    exit_reason = None
    margin_call = False
    exit_idx = entry_idx

    # Simulasi bar-by-bar sampai horizon
    for j in range(entry_idx, exit_idx_time + 1):
        bar = df.iloc[j]
        high = float(bar["high"])
        low = float(bar["low"])

        # Margin call simulasi (sama seperti DL/XGB backtest)
        if direction == 1:
            worst_price = low
        else:
            worst_price = high

        float_pnl_worst = (worst_price - entry_price) * direction * CONTRACT_SIZE * lot
        equity_worst = balance + float_pnl_worst

        if equity_worst <= STOP_OUT_LEVEL * margin_required:
            exit_price = worst_price
            exit_time = bar["time"]
            exit_reason = "MARGIN_CALL"
            margin_call = True
            exit_idx = j
            break

        # Cek SL/TP
        if direction == 1:
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
        else:
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

    # Kalau sampai horizon nggak kena SL/TP/MARGIN_CALL -> exit di close horizon
    if exit_price is None:
        exit_row = df.iloc[exit_idx_time]
        exit_price = float(exit_row["close"])
        exit_time = exit_row["time"]
        exit_reason = "TIME"
        exit_idx = exit_idx_time
    
    exit_price_raw = exit_price

    if exit_reason in ("SL", "TP", "MARGIN_CALL"):
        if direction == 1:
            # long: exit slippage selalu bikin harga lebih jelek (lebih rendah)
            exit_price = exit_price_raw - AVG_SLIPPAGE_PIPS * PIP_SIZE
        else:
            # short: exit slippage bikin harga lebih jelek (lebih tinggi)
            exit_price = exit_price_raw + AVG_SLIPPAGE_PIPS * PIP_SIZE

    price_move = (exit_price - entry_price) * direction
    gross_pnl_usd = price_move * CONTRACT_SIZE * lot

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
        "ret_real": ret_real,
        "ret_ideal": ret_ideal,
        "balance_before": balance_before,
        "balance_after": balance_after,
        "is_margin_call": margin_call,
        "hybrid_conf_group": conf_group,
        "risk_mult": risk_mult,
        "rr_mult": rr_mult,
        "rr_final": rr_tp,
    }

    return trade, balance_after, exit_idx + 1, margin_call


def run_backtest(df_test: pd.DataFrame, horizon: int, start_balance: float):
    df = df_test.reset_index(drop=True)
    n = len(df)
    trades: list[dict] = []

    balance = start_balance
    peak_balance = start_balance
    i = 0
    margin_call_happened = False

    while i < n - horizon - 1 and balance > 0:
        trade, new_balance, next_i, mc = simulate_trade_path(
            df, i, horizon, balance, peak_balance
        )

        balance = new_balance
        peak_balance = max(peak_balance, balance)

        if trade is not None:
            trades.append(trade)

        if mc:
            margin_call_happened = True
            i = next_i
            break

        i = next_i

    trades_df = pd.DataFrame(trades)
    return trades_df, margin_call_happened


def calc_stats(
    trades_df: pd.DataFrame,
    start_balance: float = 100.0,
    margin_call_happened: bool = False,
):
    if trades_df.empty:
        print("[WARN] No trades generated.")
        return

    equity_series = pd.Series([start_balance] + trades_df["balance_after"].tolist())

    wins = (trades_df["net_pnl_usd"] > 0).sum()
    losses = (trades_df["net_pnl_usd"] <= 0).sum()
    n_trades = len(trades_df)
    winrate = wins / n_trades if n_trades > 0 else np.nan

    longs = trades_df[trades_df["signal"] == 1]
    shorts = trades_df[trades_df["signal"] == -1]

    def side_stats(sub: pd.DataFrame):
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

    print("===== HYBRID (LSTM + XGB) ATR SL/TP BACKTEST (SUPER REAL) =====")
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
        print("    Balance setelah MC :", last_mc["balance_after"])
    print("=====================================================")

    # Optional: breakdown per hybrid confidence
    if "hybrid_conf_group" in trades_df.columns:
        print("\n===== HYBRID CONFIDENCE BREAKDOWN =====")
        print(trades_df["hybrid_conf_group"].value_counts(normalize=True))


def main():
    parser = argparse.ArgumentParser(description="HYBRID ATR backtest with lot & margin.")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--balance", type=float, default=START_BALANCE, help="Starting balance")
    args = parser.parse_args()

    df = pd.read_csv(INPUT_FILE, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # ADX kalau belum ada
    if ADX_COL not in df.columns:
        print(f"[INFO] Kolom ADX '{ADX_COL}' belum ada, menghitung dengan pandas_ta...")
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        df = pd.concat([df, adx[["ADX_14"]]], axis=1)

    # MTF trend H1
    if USE_MTF_FILTER:
        df_h1 = df.resample("1h", on="time").agg({"close": "last"}).dropna()
        df_h1["ema_fast"] = df_h1["close"].ewm(span=H1_EMA_FAST, adjust=False).mean()
        df_h1["ema_slow"] = df_h1["close"].ewm(span=H1_EMA_SLOW, adjust=False).mean()
        diff = df_h1["ema_fast"] - df_h1["ema_slow"]

        df_h1[MTF_TREND_COL] = 0
        df_h1.loc[diff > 0, MTF_TREND_COL] = 1
        df_h1.loc[diff < 0, MTF_TREND_COL] = -1

        df["time_h1"] = df["time"].dt.floor("h")
        df = df.merge(
            df_h1[[MTF_TREND_COL]],
            left_on="time_h1",
            right_index=True,
            how="left",
        )

    if "set" not in df.columns:
        raise ValueError("Kolom 'set' tidak ditemukan di CSV. Pastikan pipeline sebelumnya benar.")

    df_test = df[df["set"] == "test"].copy().reset_index(drop=True)

    if USE_REGIME_FILTERS:
        if ATR_COL not in df_test.columns:
            raise ValueError(f"Kolom ATR '{ATR_COL}' tidak ditemukan di DF.")
        df_test["atr_ma_50"] = df_test[ATR_COL].rolling(ATR_MA_PERIOD).mean()

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

    trades_df, margin_call_happened = run_backtest(
        df_test, horizon=HORIZON, start_balance=args.balance
    )
    print("[INFO] Generated trades:", len(trades_df))

    trades_df.to_csv("hybrid_signal_trades_superreal_atr_sl_tp_cost.csv", index=False)
    print("[OK] Saved trades -> hybrid_signal_trades_superreal_atr_sl_tp_cost.csv")

    calc_stats(trades_df, start_balance=args.balance, margin_call_happened=margin_call_happened)


if __name__ == "__main__":
    main()
