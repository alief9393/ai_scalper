# live_engine_hybrid_intraday.py
# (updated: DRY_RUN now simulates opening positions & closing on SL/TP/TIME per M15 bars)

import time
from datetime import datetime
import configparser

import numpy as np
import pandas as pd
import pandas_ta as ta
import MetaTrader5 as mt5
import torch
import torch.nn as nn
import xgboost as xgb

from features_intraday import add_m15_features, add_h1_context, add_daily_context


# =============== CONFIG LOADER =====================

def load_config():
    cfg = configparser.ConfigParser()
    cfg.read("dl_live_config.ini")
    return cfg


cfg = load_config()

# --- MT5 / umum ---
SYMBOL       = cfg["MT5"].get("SYMBOL", "XAUUSD")
MAGIC_NUMBER = cfg["MT5"].getint("MAGIC_NUMBER", 987654)
DRY_RUN      = cfg["MT5"].getboolean("DRY_RUN", True)

# --- LSTM model (pakai section [MODEL], sama seperti engine sebelumnya) ---
MODEL_PATH_LSTM   = cfg["MODEL"].get("MODEL_PATH", "lstm_intraday_best.pt")
DATA_META_FILE_DL = cfg["MODEL"].get("DATA_META_FILE", "dl_intraday_dataset_seq64.npz")
THR_LONG_DL       = cfg["MODEL"].getfloat("THR_LONG", 0.70)
THR_SHORT_DL      = cfg["MODEL"].getfloat("THR_SHORT", 0.30)

# --- XGB model (section optional [MODEL_XGB]) ---
try:
    MODEL_PATH_XGB = cfg["MODEL_XGB"].get("MODEL_PATH", "xgb_intraday.model")
    META_FILE_XGB  = cfg["MODEL_XGB"].get("META_FILE", "xgb_intraday_meta.npz")
except KeyError:
    # fallback kalau section belum ada
    MODEL_PATH_XGB = "xgb_intraday.model"
    META_FILE_XGB  = "xgb_intraday_meta.npz"

# --- ATR / TP-SL (fallback sama seperti backtest) ---
ATR_COL      = cfg["ATR_SLTP"].get("ATR_COL", "atr_14_m15")
ATR_MULT_SL  = cfg["ATR_SLTP"].getfloat("ATR_MULT_SL", 1.2)
RR_TP_GLOBAL = cfg["ATR_SLTP"].getfloat("RR_TP", 2.0)  # kalau dynamic TP OFF

# --- LOT dari config (fallback kalau dynamic risk OFF) ---
RISK_PER_TRADE = cfg["LOTS"].getfloat("RISK_PER_TRADE", 0.01)
MIN_LOT        = cfg["LOTS"].getfloat("MIN_LOT", 0.01)
MAX_LOT        = cfg["LOTS"].getfloat("MAX_LOT", 0.10)

# --- Biaya trading (buat estimasi / logging) ---
COMMISSION_PER_001     = cfg["COST_SIMULATION"].getfloat("COMMISSION_PER_001", 0.06)
SPREAD_HIDDEN_USD      = cfg["COST_SIMULATION"].getfloat("SPREAD_HIDDEN_USD", 0.037)
EST_COST_PER_TRADE_001 = COMMISSION_PER_001 + SPREAD_HIDDEN_USD

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "D1": mt5.TIMEFRAME_D1,
}
TIMEFRAME = TIMEFRAME_MAP.get(cfg["MT5"].get("TIMEFRAME", "M15"), mt5.TIMEFRAME_M15)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Akan diisi setelah connect_mt5
CONTRACT_SIZE = 100.0
SYMBOL_INFO   = None

# Simulasi DD throttle pakai equity akun
HORIZON = 3  # sama dengan labeling / backtest

# ===== HYBRID CONFIG (samakan dengan backtest hybrid lu) =====

# hybrid probability:
#   hybrid_proba_up = W_LSTM * dl_proba_up + W_XGB * xgb_proba_up
HYB_LSTM_WEIGHT = 0.6
HYB_XGB_WEIGHT  = 0.4

# hybrid confidence:
#   conf = |hybrid_proba_up - 0.5|
HYB_CONF_STRONG   = 0.30   # >= 0.30  -> A_STRONG
HYB_CONF_MODERATE = 0.20   # >= 0.20  -> B_MODERATE
# di bawah itu -> WEAK (no trade)

# Dynamic TP (RR tergantung ADX + MTF trend + hybrid strength)
USE_DYNAMIC_TP   = True
RR_TP_BASE       = 2.0        # baseline
ADX_TP_MED       = 20.0
ADX_TP_STRONG    = 25.0
RR_TP_MED_TREND  = 2.5
RR_TP_STRONG_TREND = 3.0
# extra boost by hybrid strength:
RR_BOOST_STRONG   = 0.5      # tambahan R untuk A_STRONG
RR_BOOST_MODERATE = 0.0

# Dynamic risk (persentase risk per trade adaptif terhadap ADX + drawdown + hybrid strength)
USE_DYNAMIC_RISK = True
BASE_RISK_PCT    = 0.0200  # 2.0% base
MIN_RISK_PCT     = 0.10    # 10%  (sesuai backtest agresif)
MAX_RISK_PCT     = 0.15    # 15%
RISK_MULT_STRONG   = 1.5   # A_STRONG
RISK_MULT_MODERATE = 1.0   # B_MODERATE

# Market model XAUUSD (hanya buat size lot & margin logika internal, real margin by broker)
LEVERAGE = 500

# Regime filters
USE_REGIME_FILTERS = True
ATR_MA_PERIOD = 50
ATR_MIN_MULT  = 1.0

ADX_COL = "ADX_14"
ADX_MIN = 15.0

# Session filter
USE_SESSION_FILTER = True
SESSION_START_HOUR = 7
SESSION_END_HOUR   = 22

# MTF trend filter
USE_MTF_FILTER = True
MTF_TREND_COL = "trend_h1_dir"
H1_EMA_FAST   = 50
H1_EMA_SLOW   = 200

# Proba filter di level HYBRID (bukan per model)
USE_HYB_CONF_FILTER = True

# ====== SIMULATION STATE (DRY_RUN) ======
SIM_START_BALANCE = cfg["SIM"].getfloat("SIM_START_BALANCE", 100.0) if "SIM" in cfg else 100.0
sim_balance = SIM_START_BALANCE
sim_peak_balance = SIM_START_BALANCE
sim_position = None       # dict posisi virtual open
sim_trades: list[dict] = []

# ====== MODEL DEF ======

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]
        logits = self.fc(h_last)
        return logits


def load_lstm_meta_and_model():
    data = np.load(DATA_META_FILE_DL, allow_pickle=True)
    feature_names = list(data["feature_names"])
    seq_len       = int(data["seq_len"][0])
    feat_mean     = data["feat_mean"].astype(np.float32)
    feat_std      = data["feat_std"].astype(np.float32)

    input_dim = len(feature_names)
    model = LSTMClassifier(input_dim=input_dim)
    state = torch.load(MODEL_PATH_LSTM, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    print("[INFO] Loaded LSTM meta/model:")
    print("  seq_len    :", seq_len)
    print("  n_features :", input_dim)
    print("  device     :", DEVICE)

    return feature_names, seq_len, feat_mean, feat_std, model


def load_xgb_model_and_meta():
    bst = xgb.Booster()
    bst.load_model(MODEL_PATH_XGB)

    meta = np.load(META_FILE_XGB, allow_pickle=True)
    xgb_feature_names = list(meta["feature_names"])

    print("[INFO] Loaded XGB model/meta:")
    print("  features   :", len(xgb_feature_names))

    return bst, xgb_feature_names


# =============== MT5 UTILS ==================

def connect_mt5():
    global CONTRACT_SIZE, SYMBOL_INFO

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    print("[INFO] Connected to MT5")

    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        raise RuntimeError(f"Symbol {SYMBOL} not found in MT5")

    if not symbol_info.visible:
        mt5.symbol_select(SYMBOL, True)

    SYMBOL_INFO = symbol_info

    if symbol_info.trade_contract_size > 0:
        CONTRACT_SIZE = float(symbol_info.trade_contract_size)
    else:
        CONTRACT_SIZE = 100.0  # fallback

    print("[INFO] Symbol ready:", SYMBOL)
    print("       contract_size:", CONTRACT_SIZE)
    print(
        "       volume_min    :", symbol_info.volume_min,
        "volume_max:", symbol_info.volume_max,
        "volume_step:", symbol_info.volume_step,
    )


def get_latest_m15_df(n_bars=500):
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, n_bars)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos M15 failed: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
    df = df.sort_values("time").reset_index(drop=True)
    return df


def get_latest_h1_df(n_bars=300):
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, n_bars)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos H1 failed: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
    df = df.sort_values("time").reset_index(drop=True)
    return df


def get_latest_d1_df(n_bars=200):
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, n_bars)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos D1 failed: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
    df = df.sort_values("time").reset_index(drop=True)
    return df


def get_symbol_tick():
    info = mt5.symbol_info_tick(SYMBOL)
    if info is None:
        raise RuntimeError(f"symbol_info_tick failed for {SYMBOL}")
    return info.bid, info.ask


def has_open_position():
    positions = mt5.positions_get(symbol=SYMBOL)
    return positions is not None and len(positions) > 0


def get_account_equity():
    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError("account_info() failed")
    return float(acc.equity)


def clamp_lot_to_symbol(lot: float) -> float:
    """Clamp lot ke [MIN_LOT, MAX_LOT] dan batas broker (volume_min, volume_max)."""
    if SYMBOL_INFO is None:
        return lot

    lot = max(lot, MIN_LOT)
    lot = min(lot, MAX_LOT)

    vol_min = SYMBOL_INFO.volume_min
    vol_max = SYMBOL_INFO.volume_max
    vol_step = SYMBOL_INFO.volume_step

    lot = max(lot, vol_min)
    lot = min(lot, vol_max)

    steps = round(lot / vol_step)
    lot = steps * vol_step

    return lot


# =========== RISK / REGIME HELPERS ==========

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


def classify_hybrid_conf(hybrid_proba_up: float):
    """
    Balikkan (conf, group):
      - group: "A_STRONG" / "B_MODERATE" / "WEAK"
    """
    conf = abs(hybrid_proba_up - 0.5)
    if conf >= HYB_CONF_STRONG:
        group = "A_STRONG"
    elif conf >= HYB_CONF_MODERATE:
        group = "B_MODERATE"
    else:
        group = "WEAK"
    return conf, group


def get_rr_tp_for_row(row: pd.Series, direction: int) -> float:
    if not USE_DYNAMIC_TP:
        return RR_TP_GLOBAL

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


def get_risk_pct_for_row(row: pd.Series, dd_frac: float | None, conf_group: str) -> float:
    """
    Risk% final = risk_core(ADX) * dd_factor(drawdown) * risk_mult(hybrid),
    lalu di-clamp MIN/MAX.
    dd_frac: (equity - peak_equity) / peak_equity  (<= 0 saat DD)
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
        if dd_frac >= -0.03:        # DD < 3%
            dd_factor = 1.1
        elif dd_frac >= -0.07:      # 3–7%
            dd_factor = 0.9
        elif dd_frac >= -0.15:      # 7–15%
            dd_factor = 0.6
        elif dd_frac >= -0.25:      # 15–25%
            dd_factor = 0.3
        else:                       # >25%
            dd_factor = 0.0

    # Hybrid confidence multiplier
    if conf_group == "A_STRONG":
        risk_mult = RISK_MULT_STRONG
    elif conf_group == "B_MODERATE":
        risk_mult = RISK_MULT_MODERATE
    else:
        risk_mult = 0.0  # WEAK → jangan trade

    risk = risk_core * dd_factor * risk_mult
    risk = max(risk, 0.0)
    risk = min(risk, MAX_RISK_PCT)
    return risk


def calculate_dynamic_lot(sl_dist: float, last_row: pd.Series, dd_frac: float, conf_group: str) -> float:
    """
    Hitung lot berdasarkan:
      - Dynamic risk (ADX + drawdown + hybrid strength)
      - Equity akun
      - SL distance (sl_dist) dalam harga
      - CONTRACT_SIZE
    """
    if sl_dist <= 0:
        return 0.0

    equity = get_account_equity()
    risk_pct = get_risk_pct_for_row(last_row, dd_frac, conf_group)
    if risk_pct <= 0:
        return 0.0

    risk_amount = equity * risk_pct
    raw_lot = risk_amount / (sl_dist * CONTRACT_SIZE)
    lot = clamp_lot_to_symbol(raw_lot)
    return lot


# ========== SIMULATION HELPERS (DRY_RUN) ==========

def open_sim_position(direction_str: str, entry_price: float,
                      sl_price: float, tp_price: float,
                      sl_dist: float, last_row: pd.Series,
                      conf_group: str):
    """
    Buka posisi virtual (hanya dipakai ketika DRY_RUN=True).
    Risk & lot pakai dynamic risk yang sama dengan backtest,
    tapi basis-nya sim_balance + sim_peak_balance.
    """
    global sim_position, sim_balance, sim_peak_balance

    if sim_position is not None:
        print("[SIM] Sudah ada posisi virtual open, skip open baru.")
        return

    if sl_dist <= 0:
        print("[SIM] SL distance <= 0, skip sim trade.")
        return

    # Drawdown simulasi
    if sim_peak_balance > 0:
        dd_frac_sim = (sim_balance - sim_peak_balance) / sim_peak_balance
    else:
        dd_frac_sim = 0.0

    # Risk% dinamis (pakai ADX + DD sim + hybrid conf)
    risk_pct = get_risk_pct_for_row(last_row, dd_frac_sim, conf_group)
    if risk_pct <= 0:
        print("[SIM] Risk pct <= 0 (WEAK or throttled), skip.")
        return

    risk_amount = sim_balance * risk_pct
    raw_lot = risk_amount / (sl_dist * CONTRACT_SIZE)
    lot = clamp_lot_to_symbol(raw_lot)

    if lot <= 0:
        print("[SIM] Lot <= 0, skip sim trade.")
        return

    sim_position = {
        "direction": direction_str,   # "LONG" / "SHORT"
        "entry_price": entry_price,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "lot": lot,
        "open_time": datetime.utcnow(),  # bisa diganti last_row["time"] kalau mau candle time
        "balance_before": sim_balance,
        "conf_group": conf_group,
        "sl_dist": sl_dist,
    }

    print(f"[SIM] Open {direction_str} {lot:.3f} @ {entry_price:.2f} SL={sl_price:.2f} TP={tp_price:.2f}")


def update_sim_position_with_bar(last_bar: pd.Series):
    """
    Dipanggil setiap M15 bar close (hanya kalau DRY_RUN=True).
    Cek apakah bar ini menyentuh SL/TP posisi virtual.
    """
    global sim_position, sim_balance, sim_peak_balance, sim_trades

    if sim_position is None:
        return

    high = float(last_bar["high"])
    low = float(last_bar["low"])
    time_bar = last_bar["time"]

    direction = sim_position["direction"]
    entry = sim_position["entry_price"]
    sl = sim_position["sl_price"]
    tp = sim_position["tp_price"]
    lot = sim_position["lot"]

    exit_price = None
    exit_reason = None

    # check SL/TP touched within bar (worst-case price movement)
    if direction == "LONG":
        if low <= sl:
            exit_price = sl
            exit_reason = "SL"
        elif high >= tp:
            exit_price = tp
            exit_reason = "TP"
    else:  # SHORT
        if high >= sl:
            exit_price = sl
            exit_reason = "SL"
        elif low <= tp:
            exit_price = tp
            exit_reason = "TP"

    # if not touched -> keep position open (we exit only when SL/TP hit in sim)
    if exit_price is None:
        return

    # hitung PnL
    direction_sign = 1 if direction == "LONG" else -1
    price_move = (exit_price - entry) * direction_sign
    gross_pnl = price_move * CONTRACT_SIZE * lot

    cost_per_001 = EST_COST_PER_TRADE_001
    total_cost = cost_per_001 * (lot / 0.01)

    net_pnl = gross_pnl - total_cost

    balance_before = sim_balance
    sim_balance = max(0.0, sim_balance + net_pnl)
    sim_peak_balance = max(sim_peak_balance, sim_balance)

    trade = {
        "direction": direction,
        "entry_price": entry,
        "exit_price": exit_price,
        "sl_price": sim_position["sl_price"],
        "tp_price": sim_position["tp_price"],
        "lot": lot,
        "open_time": sim_position["open_time"],
        "close_time": time_bar,
        "gross_pnl": gross_pnl,
        "cost": total_cost,
        "net_pnl": net_pnl,
        "balance_before": balance_before,
        "balance_after": sim_balance,
        "exit_reason": exit_reason,
        "conf_group": sim_position.get("conf_group"),
        "sl_dist": sim_position.get("sl_dist"),
    }
    sim_trades.append(trade)

    print(f"[SIM] Close {direction} @ {exit_price:.2f} ({exit_reason}), PnL={net_pnl:.2f}, Bal={sim_balance:.2f}")
    sim_position = None


# =============== INFERENCE =================

def build_lstm_sequence(
    df_feat: pd.DataFrame,
    feature_names: list[str],
    seq_len: int,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
):
    if len(df_feat) < seq_len:
        raise ValueError(f"Data kurang, butuh {seq_len} bar, cuma ada {len(df_feat)}")

    df_seq = df_feat.sort_values("time").iloc[-seq_len:]
    missing = [f for f in feature_names if f not in df_seq.columns]
    if missing:
        raise ValueError(f"Missing LSTM feature columns in live DF: {missing}")

    X_raw = df_seq[feature_names].to_numpy(dtype=np.float32)
    X_norm = (X_raw - feat_mean) / feat_std
    X = np.expand_dims(X_norm, axis=0)

    last_row = df_seq.iloc[-1]
    return X, last_row


def infer_lstm_proba(model, X) -> float:
    X_t = torch.from_numpy(X).float().to(DEVICE)
    with torch.no_grad():
        logits = model(X_t)
        prob = torch.softmax(logits, dim=1)
        proba_up = float(prob[0, 1].cpu().item())
    return proba_up


def infer_xgb_proba(bst: xgb.Booster, last_row: pd.Series, xgb_feature_names: list[str]) -> float:
    missing = [f for f in xgb_feature_names if f not in last_row.index]
    if missing:
        raise ValueError(f"Missing XGB feature columns in live DF: {missing}")

    x_vec = last_row[xgb_feature_names].to_numpy(dtype=np.float32).reshape(1, -1)
    dmatrix = xgb.DMatrix(x_vec, feature_names=xgb_feature_names)
    proba = float(bst.predict(dmatrix)[0])  # proba kelas 1 (UP)
    return proba


def build_hybrid_signal(dl_proba_up: float, xgb_proba_up: float, bid: float, ask: float, last_row: pd.Series):
    # Weighted hybrid proba
    hybrid_proba_up = HYB_LSTM_WEIGHT * dl_proba_up + HYB_XGB_WEIGHT * xgb_proba_up
    conf, conf_group = classify_hybrid_conf(hybrid_proba_up)

    # Tentukan arah dari hybrid_proba_up
    if hybrid_proba_up > 0.5:
        raw_signal = 1
    elif hybrid_proba_up < 0.5:
        raw_signal = -1
    else:
        raw_signal = 0

    # Hybrid confidence filter
    if USE_HYB_CONF_FILTER and conf_group == "WEAK":
        filtered_signal = 0
    else:
        filtered_signal = raw_signal

    # Regime filter terakhir
    if not passes_regime_filters(last_row):
        filtered_signal = 0

    # MTF trend filter
    if filtered_signal != 0 and USE_MTF_FILTER:
        trend_dir = last_row.get(MTF_TREND_COL, 0)
        if not (pd.isna(trend_dir) or trend_dir == 0):
            if int(trend_dir) != filtered_signal:
                filtered_signal = 0

    # Siapkan info entry
    if filtered_signal == 1:
        direction_str = "LONG"
        entry_price = float(ask)
    elif filtered_signal == -1:
        direction_str = "SHORT"
        entry_price = float(bid)
    else:
        direction_str = "FLAT"
        entry_price = None

    atr_val = None
    sl_price = None
    tp_price = None
    sl_dist = None
    tp_dist = None

    if filtered_signal != 0:
        if ATR_COL not in last_row.index:
            raise ValueError(f"{ATR_COL} tidak ada di live features")

        atr_val = float(last_row[ATR_COL])
        sl_dist = ATR_MULT_SL * atr_val

        direction = 1 if filtered_signal == 1 else -1
        rr_tp = get_rr_tp_for_row(last_row, direction)
        # apply hybrid RR boost for strong conf
        if conf_group == "A_STRONG":
            rr_tp = min(rr_tp + RR_BOOST_STRONG, 4.0)
        elif conf_group == "B_MODERATE":
            rr_tp = min(rr_tp + RR_BOOST_MODERATE, 4.0)
        tp_dist = rr_tp * sl_dist

        if filtered_signal == 1:
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

    return {
        "dl_proba_up": dl_proba_up,
        "xgb_proba_up": xgb_proba_up,
        "hybrid_proba_up": hybrid_proba_up,
        "hybrid_conf": conf,
        "hybrid_conf_group": conf_group,
        "signal": filtered_signal,
        "direction": direction_str,
        "entry_price": entry_price,
        "atr_m15": atr_val,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "sl_dist": sl_dist,
        "tp_dist": tp_dist,
        "decision_time": last_row["time"],
    }


# =============== MAIN LOOP =================

def main_loop():
    # Load model & meta
    dl_feature_names, seq_len, feat_mean, feat_std, lstm_model = load_lstm_meta_and_model()
    xgb_model, xgb_feature_names = load_xgb_model_and_meta()

    connect_mt5()

    print("[INFO] Starting HYBRID main loop...")
    print(f"[INFO] Dynamic risk: BASE={BASE_RISK_PCT*100:.2f}% MIN={MIN_RISK_PCT*100:.2f}% MAX={MAX_RISK_PCT*100:.2f}%")
    print(f"[INFO] Hybrid weights: LSTM={HYB_LSTM_WEIGHT:.2f}, XGB={HYB_XGB_WEIGHT:.2f}")
    print(f"[INFO] Hybrid conf: STRONG>={HYB_CONF_STRONG}, MOD>={HYB_CONF_MODERATE}")
    print(f"[INFO] Lot bounds: MIN_LOT={MIN_LOT}, MAX_LOT={MAX_LOT}")
    print(f"[INFO] Dynamic TP: BASE={RR_TP_BASE}, MED={RR_TP_MED_TREND}, STRONG={RR_TP_STRONG_TREND}, BOOST_STRONG={RR_BOOST_STRONG}")
    print(f"[INFO] Regime: ATR_MA_PERIOD={ATR_MA_PERIOD}, ADX_MIN={ADX_MIN}, Session={SESSION_START_HOUR}-{SESSION_END_HOUR}")
    print(f"[SIM] Starting sim balance: {sim_balance:.2f} (DRY_RUN={DRY_RUN})")

    last_bar_time = None

    # Track peak equity untuk DD throttle
    equity_now = get_account_equity()
    peak_equity = equity_now

    while True:
        try:
            df_m15 = get_latest_m15_df(n_bars=500)
            current_bar_time = df_m15["time"].iloc[-1]

            # Deteksi bar baru (close M15)
            if last_bar_time is not None and current_bar_time == last_bar_time:
                # still same bar -> update sim position with latest tick? we update on bar close only
                time.sleep(5)
                continue

            last_bar_time = current_bar_time
            print(f"\n[INFO] New M15 bar detected: {current_bar_time}")

            # Update equity & peak equity (REAL account)
            equity_now = get_account_equity()
            peak_equity = max(peak_equity, equity_now)
            dd_frac = (equity_now - peak_equity) / peak_equity if peak_equity > 0 else 0.0

            print(f"  Equity: {equity_now:.2f}, Peak: {peak_equity:.2f}, DD: {dd_frac*100:.2f}%")

            # Update posisi SIMULASI (kalau DRY_RUN)
            if DRY_RUN:
                # feed the bar that just closed to sim updater
                update_sim_position_with_bar(df_m15.iloc[-1])
                print(f"  [SIM] Bal={sim_balance:.2f}, Peak={sim_peak_balance:.2f}")

            df_feat = build_features_for_m15(df_m15)
            if len(df_feat) < seq_len:
                print("[WARN] Feature DF belum cukup panjang untuk seq_len, skip bar ini.")
                time.sleep(5)
                continue

            # Build seq untuk LSTM
            X_lstm, last_row = build_lstm_sequence(
                df_feat, dl_feature_names, seq_len, feat_mean, feat_std
            )
            bid, ask = get_symbol_tick()

            # LSTM proba
            dl_proba_up = infer_lstm_proba(lstm_model, X_lstm)

            # XGB proba (pakai last_row)
            xgb_proba_up = infer_xgb_proba(xgb_model, last_row, xgb_feature_names)

            # Hybrid combine
            result = build_hybrid_signal(dl_proba_up, xgb_proba_up, bid, ask, last_row)

            print("  Decision time     :", result["decision_time"])
            print(f"  dl_proba_up       : {result['dl_proba_up']:.4f}")
            print(f"  xgb_proba_up      : {result['xgb_proba_up']:.4f}")
            print(f"  hybrid_proba_up   : {result['hybrid_proba_up']:.4f}")
            print(f"  hybrid_conf       : {result['hybrid_conf']:.4f}")
            print(f"  hybrid_conf_group : {result['hybrid_conf_group']}")
            print(f"  hybrid_signal     : {result['signal']} ({result['direction']})")

            if result["signal"] == 0:
                print("  -> FLAT / filtered out, no trade.")
            else:
                print(f"  entry_price       : {result['entry_price']:.2f}")
                print(f"  ATR(M15)          : {result['atr_m15']:.2f}")
                print(f"  SL price          : {result['sl_price']:.2f}")
                print(f"  TP price          : {result['tp_price']:.2f}")
                rr_approx = result["tp_dist"] / result["sl_dist"] if result["sl_dist"] else 0.0
                print(f"  RR (approx)       : {rr_approx:.2f} R")

                if result["sl_dist"] is None or result["sl_dist"] <= 0:
                    print("  -> SL distance invalid, skip trade.")
                    time.sleep(5)
                    continue

                # calculate lot using REAL equity when real; SIM equity when DRY_RUN
                if DRY_RUN:
                    # use sim_balance for lot sizing in paper mode
                    # but calculate_dynamic_lot uses get_account_equity, so we compute manually similar logic
                    dd_frac_sim = (sim_balance - sim_peak_balance) / sim_peak_balance if sim_peak_balance > 0 else 0.0
                    lot = calculate_dynamic_lot(result["sl_dist"], last_row, dd_frac_sim, result["hybrid_conf_group"])
                else:
                    lot = calculate_dynamic_lot(result["sl_dist"], last_row, dd_frac, result["hybrid_conf_group"])

                if lot <= 0:
                    print("  -> Calculated lot <= 0, skip trade.")
                    time.sleep(5)
                    continue

                est_cost = EST_COST_PER_TRADE_001 * (lot / 0.01)
                print(f"  Dynamic lot       : {lot:.3f}")
                print(f"  Est. cost/trade   : ~{est_cost:.3f} USD")

                if has_open_position():
                    print("  -> Existing position detected, skip new order.")
                else:
                    if DRY_RUN:
                        print("  -> DRY_RUN: open simulated position now.")
                        open_sim_position(
                            result["direction"],
                            result["entry_price"],
                            result["sl_price"],
                            result["tp_price"],
                            result["sl_dist"],
                            last_row,
                            result["hybrid_conf_group"],
                        )
                    else:
                        print("  -> Sending HYBRID order...")
                        send_order(
                            result["direction"],
                            lot,
                            result["entry_price"],
                            result["sl_price"],
                            result["tp_price"],
                        )

            time.sleep(5)

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")
            break
        except Exception as e:
            print("[ERROR] Exception in main loop:", e)
            time.sleep(10)

    # Shutdown: save sim trades if any
    if DRY_RUN and len(sim_trades) > 0:
        df_sim = pd.DataFrame(sim_trades)
        df_sim.to_csv("hybrid_live_sim_trades.csv", index=False)
        print("[SIM] Saved sim trades -> hybrid_live_sim_trades.csv")
        print(f"[SIM] Final sim balance: {sim_balance:.2f}")

    mt5.shutdown()
    print("[INFO] MT5 shutdown.")


if __name__ == "__main__":
    main_loop()
