#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binance USDT‑M Perpetual Scanner (Telegram notify)
- 每次運行：掃描全市場 USDT PERPETUAL，依量價關係 +（可選）MACD/RSI 篩選，產出交易建議
- 設計給 GitHub Actions 使用（單輪執行），也可本機執行
Author: ChatGPT
"""
import os, time, json, math, statistics
from datetime import datetime, timezone
from typing import List, Dict, Optional
import requests

BINANCE_FAPI = "https://fapi.binance.com"
STATE_FILE = os.environ.get("STATE_FILE", "signals_state.json")

# ----------------------------- Config -----------------------------
CFG = {
    "quote": os.environ.get("QUOTE", "USDT"),
    "timeframes": [x.strip() for x in os.environ.get("TIMEFRAMES", "1h").split(",")],
    "klines_limit": int(os.environ.get("KLINES_LIMIT", "240")),
    # 量能判定（相對近20根SMA）
    "vol_ratio_up": float(os.environ.get("VOL_RATIO_UP", "1.2")),
    "vol_ratio_down": float(os.environ.get("VOL_RATIO_DOWN", "0.8")),
    # 價格漲跌 vs 前一收盤；介於 ±此值視為「價平」
    "px_flat_threshold": float(os.environ.get("PX_FLAT_THRESHOLD", "0.001")),  # 0.1%
    # 高/低位分位數
    "low_quantile": float(os.environ.get("LOW_Q", "0.2")),
    "high_quantile": float(os.environ.get("HIGH_Q", "0.8")),
    # 技術濾網
    "use_macd_filter": os.environ.get("USE_MACD_FILTER", "true").lower() == "true",
    "use_rsi_filter": os.environ.get("USE_RSI_FILTER", "false").lower() == "true",
    # 訊號最低信心
    "min_conf": float(os.environ.get("MIN_CONF", "0.7")),
    # 風險管理（用於建議的SL/TP計算）
    "atr_mult": float(os.environ.get("ATR_MULT", "1.2")),
    "tp1_r_mult": float(os.environ.get("TP1_R_MULT", "1.5")),
    "tp2_r_mult": float(os.environ.get("TP2_R_MULT", "2.5")),
    # 交易所步進緩存
    "price_step": {},
    # Telegram
    "tg_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
    "tg_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
    # 只掃 USDT 永續
    "only_perp": os.environ.get("ONLY_PERP", "true").lower() == "true",
}

# ----------------------------- Utils -----------------------------
def now_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def http_get(url, params=None):
    for _ in range(3):
        try:
            r = requests.get(url, params=params, timeout=12)
            if r.status_code == 200:
                return r.json()
            time.sleep(0.5)
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"GET failed: {url}")

def load_state() -> Dict:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(st: Dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

def round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    precision = 0
    if step < 1:
        s = f"{step:.10f}".rstrip("0")
        if "." in s:
            precision = len(s.split(".")[1])
    return round(round(value / step) * step, precision)

# ----------------------------- Indicators -----------------------------
def ema(values: List[float], period: int) -> List[float]:
    k = 2/(period+1)
    out = []
    ema_prev = None
    for v in values:
        if ema_prev is None:
            ema_prev = v
        else:
            ema_prev = v*k + ema_prev*(1-k)
        out.append(ema_prev)
    return out

def macd(values: List[float], fast=12, slow=26, signal=9):
    if len(values) < slow + signal + 5:
        return None
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    dif_series = [a-b for a, b in zip(ema_fast[-len(ema_slow):], ema_slow)]
    dea_series = ema(dif_series, signal)
    hist_series = [d - s for d, s in zip(dif_series[-len(dea_series):], dea_series)]
    return {"dif": dif_series[-1], "dea": dea_series[-1], "hist": hist_series[-1]}

def rsi(values: List[float], period=14):
    if len(values) < period + 2:
        return None
    gains, losses = [], []
    for i in range(1, len(values)):
        chg = values[i] - values[i-1]
        gains.append(max(chg, 0))
        losses.append(abs(min(chg, 0)))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
    rs = float('inf') if avg_loss == 0 else avg_gain / avg_loss
    val = 100 - (100/(1+rs))
    return val

def atr(h: List[float], l: List[float], c: List[float], period=14):
    if len(c) < period + 2:
        return None
    trs = []
    for i in range(1, len(c)):
        tr = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        trs.append(tr)
    atr_val = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr_val = (atr_val*(period-1) + trs[i]) / period
    return atr_val

# ----------------------------- Binance helpers -----------------------------
def fetch_exchange_info():
    data = http_get(f"{BINANCE_FAPI}/fapi/v1/exchangeInfo")
    symbols = []
    for s in data.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if CFG["only_perp"] and s.get("contractType") != "PERPETUAL":
            continue
        if s.get("quoteAsset") != CFG["quote"]:
            continue
        sym = s["symbol"]
        symbols.append(sym)
        for filt in s.get("filters", []):
            if filt.get("filterType") == "PRICE_FILTER":
                tick = float(filt.get("tickSize", "0.001"))
                CFG["price_step"][sym] = tick
    return symbols

def fetch_klines(symbol: str, interval: str, limit: int):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = http_get(f"{BINANCE_FAPI}/fapi/v1/klines", params=params)
    out = []
    for k in data:
        out.append({
            "open_time": int(k[0]),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "close_time": int(k[6]),
        })
    return out

# ----------------------------- Classification -----------------------------
def quantile_level(closes: List[float], q_low: float, q_high: float) -> str:
    sorted_c = sorted(closes)
    n = len(sorted_c)
    if n < 5:
        return "中位"
    low_idx = max(0, int(q_low * (n - 1)))
    high_idx = max(0, int(q_high * (n - 1)))
    low_val = sorted_c[low_idx]
    high_val = sorted_c[high_idx]
    cur = closes[-1]
    if cur <= low_val:
        return "低位"
    if cur >= high_val:
        return "高位"
    return "中位"

def price_movement(closes: List[float], flat_th: float) -> str:
    chg = (closes[-1] - closes[-2]) / closes[-2]
    if chg > flat_th:
        return "價漲"
    if chg < -flat_th:
        return "價跌"
    return "價平"

def volume_movement(vols: List[float], up_ratio: float, down_ratio: float) -> str:
    sma = statistics.fmean(vols[-20:]) if len(vols) >= 20 else statistics.fmean(vols)
    ratio = vols[-1] / sma if sma > 0 else 1.0
    if ratio >= up_ratio:
        return "量增"
    if ratio <= down_ratio:
        return "量減"
    return "量平"

RULE_MAP = {
    ("低位","價漲","量增"): ("空頭趨勢反轉","LONG"),
    ("高位","價漲","量增"): ("多頭趨勢反轉","SHORT"),
    ("低位","價漲","量平"): ("空頭反轉","LONG"),
    ("高位","價漲","量平"): ("盤整後多頭趨勢反轉","SHORT"),
    ("低位","價漲","量減"): ("底部形成","LONG"),
    ("高位","價漲","量減"): ("頂部形成","SHORT"),
    ("低位","價跌","量增"): ("空頭反轉多頭","LONG"),
    ("高位","價跌","量增"): ("多頭反轉空頭","SHORT"),
    ("低位","價跌","量平"): ("底部已現","LONG"),
    ("高位","價跌","量平"): ("頂部已現","SHORT"),
    ("低位","價跌","量減"): ("底部即將形成","LONG"),
    ("高位","價跌","量減"): ("頂部即將反轉","SHORT"),
    ("低位","價平","量增"): ("空頭即將結束","LONG"),
    ("高位","價平","量增"): ("多頭即將結束","SHORT"),
    ("低位","價平","量平"): ("底部將結束","LONG"),
    ("高位","價平","量平"): ("頂部將結束","SHORT"),
    ("低位","價平","量減"): ("底部已現","LONG"),
    ("高位","價平","量減"): ("頂部已現","SHORT"),
}

def evaluate_signal(symbol: str, tf: str, kl: List[Dict]) -> Optional[Dict]:
    closes = [x["close"] for x in kl]
    highs = [x["high"] for x in kl]
    lows  = [x["low"]  for x in kl]
    vols  = [x["volume"] for x in kl]
    if len(closes) < 50:
        return None

    level = quantile_level(closes[-120:] if len(closes)>=120 else closes, CFG["low_quantile"], CFG["high_quantile"])
    p_move = price_movement(closes, CFG["px_flat_threshold"])
    v_move = volume_movement(vols, CFG["vol_ratio_up"], CFG["vol_ratio_down"])

    key = (level, p_move, v_move)
    if key not in RULE_MAP:
        return None
    meaning, direction = RULE_MAP[key]

    conf = 0.6
    m = macd(closes)
    if CFG["use_macd_filter"] and m is not None:
        if direction == "LONG" and m["dif"] > 0 and m["dif"] >= m["dea"]:
            conf += 0.2
        elif direction == "SHORT" and m["dif"] < 0 and m["dif"] <= m["dea"]:
            conf += 0.2
        else:
            conf -= 0.2
    if CFG["use_rsi_filter"]:
        r = rsi(closes)
        if r is not None:
            if direction == "LONG" and r < 60:
                conf += 0.1
            if direction == "SHORT" and r > 40:
                conf += 0.1

    a = atr(highs, lows, closes, period=14)
    last = closes[-1]
    if a is None or a <= 0:
        return None
    risk = CFG["atr_mult"] * a
    if direction == "LONG":
        sl = last - risk
        tp1 = last + CFG["tp1_r_mult"]*risk
        tp2 = last + CFG["tp2_r_mult"]*risk
    else:
        sl = last + risk
        tp1 = last - CFG["tp1_r_mult"]*risk
        tp2 = last - CFG["tp2_r_mult"]*risk

    step = CFG["price_step"].get(symbol, 0.01)
    entry = round_step(last, step)
    sl = round_step(sl, step)
    tp1 = round_step(tp1, step)
    tp2 = round_step(tp2, step)

    conf = max(0.0, min(1.0, conf))
    signal_time = kl[-1]["close_time"]

    return {
        "symbol": symbol,
        "timeframe": tf,
        "level": level,
        "price_move": p_move,
        "vol_move": v_move,
        "meaning": meaning,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "confidence": round(conf, 2),
        "signal_time": signal_time,
    }

# ----------------------------- Telegram -----------------------------
def tg_send(msg: str):
    token = CFG["tg_bot_token"]
    chat_id = CFG["tg_chat_id"]
    if not token or not chat_id:
        print("[WARN] Telegram 未設定，僅列印訊息：\n", msg)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown", "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print("[WARN] Telegram 發送失敗：", r.text)
    except Exception as e:
        print("[WARN] Telegram 發送例外：", e)

def format_signal(sig: Dict) -> str:
    arrow = "🟢多" if sig["direction"] == "LONG" else "🔴空"
    return (
        f"{arrow} *{sig['symbol']}* `{sig['timeframe']}`\n"
        f"情境：{sig['level']}{sig['price_move']}{sig['vol_move']} → {sig['meaning']}\n"
        f"建議：方向 *{sig['direction']}*｜進場 `{sig['entry']}`｜SL `{sig['sl']}`｜TP1 `{sig['tp1']}`｜TP2 `{sig['tp2']}`\n"
        f"信心：`{sig['confidence']}`｜更新：{now_ts()}"
    )

# ----------------------------- Main -----------------------------
def scan_once():
    symbols = fetch_exchange_info()
    st = load_state()
    new_st = st.copy()
    out_signals = []

    for tf in CFG["timeframes"]:
        for sym in symbols:
            try:
                kl = fetch_klines(sym, tf, CFG["klines_limit"])
                sig = evaluate_signal(sym, tf, kl)
                if not sig:
                    continue
                key = f"{sym}:{tf}"
                last_rec = st.get(key)
                if last_rec and last_rec.get("signal_time") == sig["signal_time"] and last_rec.get("direction") == sig["direction"]:
                    continue
                if sig["confidence"] >= CFG["min_conf"]:
                    out_signals.append(sig)
                    new_st[key] = {"signal_time": sig["signal_time"], "direction": sig["direction"]}
            except Exception as e:
                print(f"[ERR] {sym} {tf} -> {e}")

    if out_signals:
        out_signals.sort(key=lambda x: (-x["confidence"], x["symbol"]))
        batch, cur = [], 0
        for s in out_signals:
            text = format_signal(s)
            batch.append(text)
            cur += len(text)
            if cur > 3500:
                tg_send("\n\n".join(batch))
                batch, cur = [], 0
        if batch:
            tg_send("\n\n".join(batch))
    else:
        print("[INFO] 本次無訊號")

    save_state(new_st)

def main():
    print(f"[INFO] 開始掃描 {now_ts()} | TF={CFG['timeframes']}")
    scan_once()

if __name__ == "__main__":
    main()
