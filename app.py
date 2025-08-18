from flask import Flask, render_template, jsonify, request
import requests, time, traceback, os
from datetime import datetime
import pytz

app = Flask(__name__)

# =================== CONFIG ===================
API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "b7ea33d435964da0b0a65b1c6a029891")
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "EUR/JPY", "AUD/USD"]
INTERVAL = "5min"
OUTPUTSIZE = 220
LOCAL_TZ = pytz.timezone("Asia/Kolkata")

# ZigZag / ATR
ATR_PERIOD = 14
ZZ_ATR_MULT = 1.0
MAX_SWINGS = 80

# CWRV 1-2-3
FIB_MIN = 0.236
FIB_MAX = 0.786
MIN_BODY_PERCENT = 0.25     # breakout candle body vs range (0..1)
RECENT_P3_MAX_AGE = 50       # candles

# Sideways
SIDEWAYS_BARS = 12
SIDEWAYS_OVERLAP_COUNT = 9

# Backtest
BACKTEST_SIGNALS = 30

# Cache
CACHE_TTL = 20  # seconds
_cache = {"candles": {}, "ts": {}}


# =================== UTILS ===================
def log(msg):
    try:
        print(f"[{datetime.utcnow().isoformat()}] {msg}")
    except Exception:
        pass

def http_get(url, params=None, timeout=12):
    try:
        return requests.get(url, params=params, timeout=timeout)
    except Exception as e:
        log(f"http_get error: {e}")
        return None


# =================== DATA ===================
def fetch_candles(symbol, interval=INTERVAL, outputsize=OUTPUTSIZE):
    """
    Twelve Data fetch. Returns oldest->newest candle list with keys t,o,h,l,c.
    None on failure (never throws).
    """
    try:
        now = time.time()
        if symbol in _cache["candles"] and (now - _cache["ts"].get(symbol, 0)) < CACHE_TTL:
            return _cache["candles"][symbol]

        base = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": API_KEY,
            "format": "JSON"
        }
        r = http_get(base, params=params, timeout=12)
        if r is None:
            return None
        data = r.json()

        if "values" not in data:
            # Try without slash (EURUSD etc.)
            alt = symbol.replace("/", "")
            if alt != symbol:
                params["symbol"] = alt
                r = http_get(base, params=params, timeout=12)
                if r is None:
                    return None
                data = r.json()

        if "values" not in data:
            log(f"fetch_candles bad response for {symbol}: {data}")
            return None

        raw = list(reversed(data["values"]))  # oldest->newest
        candles = []
        for v in raw:
            try:
                candles.append({
                    "t": v.get("datetime"),
                    "o": float(v["open"]),
                    "h": float(v["high"]),
                    "l": float(v["low"]),
                    "c": float(v["close"]),
                })
            except Exception:
                continue

        if not candles:
            return None

        _cache["candles"][symbol] = candles
        _cache["ts"][symbol] = now
        return candles
    except Exception as e:
        log(f"fetch_candles exception {symbol}: {e}")
        return None


# =================== INDICATORS ===================
def compute_atr(highs, lows, closes, period=ATR_PERIOD):
    try:
        if len(closes) < 5:
            return 0.0
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        if not trs:
            return 0.0
        p = min(period, len(trs))
        return sum(trs[-p:]) / p
    except Exception:
        return 0.0

def zigzag_swings(candles, atr_mult=ZZ_ATR_MULT, max_keep=MAX_SWINGS):
    """
    Simple ATR-based swing finder.
    Returns list of dicts: {'idx','type'('H'/'L'),'price','time'}
    """
    try:
        n = len(candles)
        if n < ATR_PERIOD + 3:
            return []

        highs = [c["h"] for c in candles]
        lows  = [c["l"] for c in candles]
        closes= [c["c"] for c in candles]
        a = compute_atr(highs, lows, closes, ATR_PERIOD)
        if a <= 0:
            return []

        threshold = a * atr_mult
        swings = []
        direction = None
        cur_peak = highs[0]; cur_peak_idx = 0
        cur_trough = lows[0]; cur_trough_idx = 0

        for i in range(1, n):
            h = highs[i]; l = lows[i]
            if h >= cur_peak:
                cur_peak = h; cur_peak_idx = i
            if l <= cur_trough:
                cur_trough = l; cur_trough_idx = i

            if direction is None:
                if cur_peak - cur_trough >= threshold:
                    direction = "up" if closes[-1] >= closes[0] else "down"
                    if direction == "up":
                        swings.append({"idx": cur_trough_idx, "type": "L", "price": cur_trough, "time": candles[cur_trough_idx]["t"]})
                    else:
                        swings.append({"idx": cur_peak_idx, "type": "H", "price": cur_peak, "time": candles[cur_peak_idx]["t"]})
                continue

            if direction == "up":
                if cur_peak - l >= threshold:
                    swings.append({"idx": cur_peak_idx, "type": "H", "price": cur_peak, "time": candles[cur_peak_idx]["t"]})
                    direction = "down"
                    cur_trough = l; cur_trough_idx = i
            else:
                if h - cur_trough >= threshold:
                    swings.append({"idx": cur_trough_idx, "type": "L", "price": cur_trough, "time": candles[cur_trough_idx]["t"]})
                    direction = "up"
                    cur_peak = h; cur_peak_idx = i

        if direction == "up":
            swings.append({"idx": cur_peak_idx, "type": "H", "price": cur_peak, "time": candles[cur_peak_idx]["t"]})
        elif direction == "down":
            swings.append({"idx": cur_trough_idx, "type": "L", "price": cur_trough, "time": candles[cur_trough_idx]["t"]})

        # dedupe & sort
        unique = {(s["idx"], s["type"]): s for s in swings}
        swings = sorted(unique.values(), key=lambda x: x["idx"])
        return swings[-max_keep:]
    except Exception as e:
        log(f"zigzag error: {e}")
        return []


"""def market_structure(swings):
    try:
        if len(swings) < 5:
            return "sideways", "few swings"
        highs = [s for s in swings if s["type"] == "H"]
        lows  = [s for s in swings if s["type"] == "L"]
        if len(highs) >= 3 and len(lows) >= 3:
            if highs[-3]["price"] < highs[-2]["price"] < highs[-1]["price"] and lows[-3]["price"] < lows[-2]["price"] < lows[-1]["price"]:
                return "up", "HH & HL"
            if highs[-3]["price"] > highs[-2]["price"] > highs[-1]["price"] and lows[-3]["price"] > lows[-2]["price"] > lows[-1]["price"]:
                return "down", "LH & LL"
        return "sideways", "mixed"
    except Exception as e:
        log(f"market_structure error: {e}")
        return "sideways", "error"""
        
        def market_structure(swings):
    try:
        if len(swings) < 3:  # need at least 2 swings
            return "sideways", "few swings"

        highs = [s for s in swings if s["type"] == "H"]
        lows  = [s for s in swings if s["type"] == "L"]

        # Uptrend check (less strict) → only 1 HH + 1 HL required
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-2]["price"] < highs[-1]["price"] and lows[-2]["price"] < lows[-1]["price"]:
                return "up", "HH & HL"

        # Downtrend check (less strict) → only 1 LH + 1 LL required
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-2]["price"] > highs[-1]["price"] and lows[-2]["price"] > lows[-1]["price"]:
                return "down", "LH & LL"

        # If not trending, then sideways
        return "sideways", "mixed"
    except Exception as e:
        log(f"market_structure error: {e}")
        return "sideways", "error"
        
'''def market_structure(swings, tolerance=0.0001):
    try:
        if len(swings) < 5:
            return "sideways", "few swings"

        highs = [s for s in swings if s["type"] == "H"]
        lows  = [s for s in swings if s["type"] == "L"]

        def higher(a, b):
            return (b - a) > tolerance

        def lower(a, b):
            return (a - b) > tolerance

        if len(highs) >= 3 and len(lows) >= 3:
            if higher(highs[-3]["price"], highs[-2]["price"]) and higher(highs[-2]["price"], highs[-1]["price"]) \
               and higher(lows[-3]["price"], lows[-2]["price"]) and higher(lows[-2]["price"], lows[-1]["price"]):
                return "up", "HH & HL"

            if lower(highs[-3]["price"], highs[-2]["price"]) and lower(highs[-2]["price"], highs[-1]["price"]) \
               and lower(lows[-3]["price"], lows[-2]["price"]) and lower(lows[-2]["price"], lows[-1]["price"]):
                return "down", "LH & LL"

        return "sideways", "mixed"
    except Exception as e:
        log(f"market_structure error: {e}")
        return "sideways", "error"'''
        
'''def market_structure(swings, candles, overlap_bars=5):
    try:
        if len(swings) < 2:
            return "sideways", "few swings"

        # ---- 1. Uptrend check ----
        for i in range(1, len(swings)):
            if swings[i]["type"] == "H" and swings[i]["price"] > swings[i-1]["price"]:
                # found HH
                for j in range(i+1, len(swings)):
                    if swings[j]["type"] == "L" and swings[j]["price"] > swings[i-1]["price"]:
                        return "up", "min 1HH + 1HL"

        # ---- 2. Downtrend check ----
        for i in range(1, len(swings)):
            if swings[i]["type"] == "L" and swings[i]["price"] < swings[i-1]["price"]:
                # found LL
                for j in range(i+1, len(swings)):
                    if swings[j]["type"] == "H" and swings[j]["price"] < swings[i-1]["price"]:
                        return "down", "min 1LL + 1LH"

        # ---- 3. Sideways check by candle overlap ----
        overlap = 0
        for i in range(1, len(candles)):
            prev = candles[i-1]
            curr = candles[i]
            prev_dir = "green" if prev["c"] > prev["o"] else "red"
            curr_dir = "green" if curr["c"] > curr["o"] else "red"
            if prev_dir != curr_dir:
                overlap += 1
                if overlap >= overlap_bars:
                    return "sideways", f"{overlap_bars}+ alternating candles"
            else:
                overlap = 0  # reset streak

        return "sideways", "no clear trend"
    except Exception as e:
        log(f"market_structure error: {e}")
        return "sideways", "error"'''

def detect_pattern(candles):
    """Simple candle helpers for confluence."""
    try:
        if len(candles) < 2: return None
        a, b = candles[-2], candles[-1]
        if a["c"] < a["o"] and b["c"] > b["o"] and b["c"] > a["o"]:
            return "bullish_engulf"
        if a["c"] > a["o"] and b["c"] < b["o"] and b["c"] < a["o"]:
            return "bearish_engulf"
        body = abs(b["c"] - b["o"])
        up_w = b["h"] - max(b["c"], b["o"])
        lo_w = min(b["c"], b["o"]) - b["l"]
        rng = b["h"] - b["l"] if (b["h"] - b["l"]) != 0 else 1e-9
        if (lo_w / rng) > 0.66:
            return "pin_bottom"
        if (up_w / rng) > 0.66:
            return "pin_top"
        return None
    except Exception:
        return None


def sideways_filter(candles):
    """Detect boxy overlap in last N bars."""
    try:
        if len(candles) < SIDEWAYS_BARS: return False
        last = candles[-SIDEWAYS_BARS:]
        overlaps = 0
        for i in range(1, len(last)):
            a, b = last[i-1], last[i]
            a_lo, a_hi = min(a["o"], a["c"]), max(a["o"], a["c"])
            b_lo, b_hi = min(b["o"], b["c"]), max(b["o"], b["c"])
            if (b_lo >= a_lo and b_hi <= a_hi) or (a_lo >= b_lo and a_hi <= b_hi):
                overlaps += 1
        return overlaps >= SIDEWAYS_OVERLAP_COUNT
    except Exception:
        return 


# =================== ELLIOTT WAVE (HEURISTICS) ===================
# We approximate using recent swings. These are rule-driven checks, not perfect labeling.

def _pct_move(a, b):
    if a == 0: return 0.0
    return abs((b - a) / a)

def _len(a, b):
    return b - a

def detect_impulse(swings):
    """
    Try to match a 5-point L-H-L-H-L (bull) or H-L-H-L-H (bear) with basic rules:
    - Wave2 does not retrace 100% of Wave1
    - Wave3 not shortest (approx)
    - Wave4 does not overlap Wave1 territory (approx)
    Returns (label, direction) or (None, None)
    """
    if len(swings) < 5:
        return None, None
    seq = swings[-5:]
    t = [s["type"] for s in seq]
    if t == ["L", "H", "L", "H", "L"]:  # bull structure
        p = [s["price"] for s in seq]
        w1 = _len(p[0], p[1]); w2 = _len(p[1], p[2]); w3 = _len(p[2], p[3]); w4 = _len(p[3], p[4])
        # rules
        if p[2] <= p[0]:  # Wave2 retrace > 100%
            return None, None
        if abs(w3) <= min(abs(w1), abs(w4))*0.8:  # 3 too small vs others
            return None, None
        if p[4] <= p[1] - (p[1]-p[0])*0.05:  # Wave4 overlaps Wave1 territory (approx)
            return None, None
        return "Impulse (bull)", "up"

    if t == ["H", "L", "H", "L", "H"]:  # bear structure
        p = [s["price"] for s in seq]
        w1 = _len(p[1], p[0]); w2 = _len(p[0], p[2]); w3 = _len(p[2], p[1]); w4 = _len(p[1], p[3])
        if p[2] >= p[0]:  # Wave2 retrace > 100%
            return None, None
        if abs(w3) <= min(abs(w1), abs(w4))*0.8:
            return None, None
        if p[4] >= p[1] + (p[0]-p[1])*0.05:
            return None, None
        return "Impulse (bear)", "down"

    return None, None

def detect_diagonal(swings):
    """
    Very rough diagonal check: overlapping structure with contracting ranges.
    """
    if len(swings) < 5:
        return None
    seq = swings[-5:]
    rngs = [abs(seq[i]["price"] - seq[i-1]["price"]) for i in range(1, 5)]
    contracting = all(rngs[i] <= rngs[i-1] for i in range(1, len(rngs)))
    if not contracting:
        return None
    # if last type echoes start type and ranges contract, call diagonal
    return "Diagonal (ending/leading)"

def detect_zigzag(swings):
    """
    A-B-C = 5-3-5 feel using swings (H/L alternation with decent depth B).
    """
    if len(swings) < 3:
        return None
    seq = swings[-3:]
    t = [s["type"] for s in seq]
    p = [s["price"] for s in seq]
    # Bullish zigzag ends on L: H-L-H sequence (down A, up B, down C) in bearish context; inverse for bullish
    # We'll infer by depth of B vs A and C relation
    if t == ["H", "L", "H"]:
        # B is shallow retrace, C reaches near A
        a_len = _pct_move(p[0], p[1])
        c_len = _pct_move(p[2], p[1])
        if a_len > 0.003 and c_len > a_len*0.6:
            return "Zigzag"
    if t == ["L", "H", "L"]:
        a_len = _pct_move(p[0], p[1])
        c_len = _pct_move(p[2], p[1])
        if a_len > 0.003 and c_len > a_len*0.6:
            return "Zigzag"
    return None

def detect_flat(swings):
    """
    Flat: A ≈ B near same level, then C breaks slightly beyond A.
    """
    if len(swings) < 3:
        return None
    seq = swings[-3:]
    p = [s["price"] for s in seq]
    t = [s["type"] for s in seq]
    if t == ["L", "H", "L"]:
        # B near A (top), C slightly under A
        if abs(_pct_move(p[0], p[1])) < 0.004 and p[2] < p[0]:
            return "Flat"
    if t == ["H", "L", "H"]:
        if abs(_pct_move(p[0], p[1])) < 0.004 and p[2] > p[0]:
            return "Flat"
    return None

def detect_triangle(swings):
    """
    Triangle: series of contracting H/L over ~5 points.
    """
    if len(swings) < 5:
        return None
    seq = swings[-5:]
    highs = [s["price"] for s in seq if s["type"] == "H"]
    lows  = [s["price"] for s in seq if s["type"] == "L"]
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[0] > highs[-1] and lows[0] < lows[-1]:
            return "Triangle (contracting)"
    return None

def elliott_type_and_wave(swings):
    """
    Combine detectors into a single label like:
    - "Wave 3 (Impulse, up)"
    - "Wave C (Zigzag)"
    - "Triangle (contracting)"
    """
    try:
        itype, idir = detect_impulse(swings)
        if itype:
            # Very rough current wave position guess: if last is L in bull or H in bear, we might be in Wave 4/5.
            seq = swings[-5:]
            t = [s["type"] for s in seq]
            if idir == "up":
                if t == ["L","H","L","H","L"]:
                    # if last L is shallow vs previous L -> we expect Wave 5 next
                    return "Wave 5 (Impulse, up)"
                return "Impulse (up)"
            else:
                if t == ["H","L","H","L","H"]:
                    return "Wave 5 (Impulse, down)"
                return "Impulse (down)"

        diag = detect_diagonal(swings)
        if diag:
            return diag

        zz = detect_zigzag(swings)
        if zz:
            # If last swing is L -> probably end of C down in bull; if H -> end of C up in bear
            last = swings[-1]
            if last["type"] == "L":
                return "Wave C (Zigzag, down)"
            else:
                return "Wave C (Zigzag, up)"

        flat = detect_flat(swings)
        if flat:
            last = swings[-1]
            if last["type"] == "L":
                return "Wave C (Flat, down)"
            else:
                return "Wave C (Flat, up)"

        tri = detect_triangle(swings)
        if tri:
            return tri

        return "unknown"
    except Exception as e:
        log(f"elliott_type_and_wave error: {e}")
        return "unknown"


# =================== CWRV 1-2-3 ===================
def find_123_points_from_swings(swings, lookback=10):
    try:
        if len(swings) < 3: return None, None, None, "not enough swings"
        start = max(0, len(swings) - lookback)
        for i in range(start, len(swings)-2):
            s1, s2, s3 = swings[i], swings[i+1], swings[i+2]
            # Bullish 1-2-3: higher low
            if s1["type"] == "L" and s2["type"] == "H" and s3["type"] == "L" and s3["price"] > s1["price"]:
                return s1, s2, s3, "L-H-L"
            # Bearish 1-2-3: lower high
            if s1["type"] == "H" and s2["type"] == "L" and s3["type"] == "H" and s3["price"] < s1["price"]:
                return s1, s2, s3, "H-L-H"
        return None, None, None, "no 1-2-3"
    except Exception as e:
        return None, None, None, f"error {e}"

def fib_retrace(p1, p2, p3):
    try:
        denom = (p2 - p1)
        if denom == 0: return 0.0
        return abs((p2 - p3) / denom)
    except Exception:
        return 0.0

def validate_123(p1, p2, p3, candles, trend):
    try:
        if not (p1 and p2 and p3): return False, "missing points"
        if not (p1["idx"] < p2["idx"] < p3["idx"]): return False, "bad chronology"
        fib = fib_retrace(p1["price"], p2["price"], p3["price"])
        if not (FIB_MIN <= fib <= FIB_MAX): return False, f"fib {fib:.3f} outside"
        last = candles[-1]
        # breakout beyond p2 in trend direction
        if trend == "up":
            if last["c"] <= p2["price"]:
                return False, "no breakout above p2"
        elif trend == "down":
            if last["c"] >= p2["price"]:
                return False, "no breakout below p2"
        # breakout candle has decent body
        body = abs(last["c"] - last["o"])
        rng = last["h"] - last["l"] if last["h"] - last["l"] > 0 else 1e-9
        if (body / rng) < MIN_BODY_PERCENT:
            return False, f"breakout body too small ({body/rng:.2f})"
        # point 3 must be reasonably recent
        if (len(candles) - p3["idx"]) > RECENT_P3_MAX_AGE:
            return False, "p3 too old"
        return True, f"fib={fib:.3f}; body={body/rng:.2f}"
    except Exception as e:
        return False, f"validate error {e}"

def backtest_accuracy(candles, swings, trend):
    try:
        if len(candles) < 60 or len(swings) < 6:
            return 100
        wins = []
        for i in range(25, len(candles)-1):
            hist = candles[:i+1]
            sw = zigzag_swings(hist)
            st, _ = market_structure(sw)
            if st != trend: 
                continue
            p1,p2,p3,_ = find_123_points_from_swings(sw)
            ok, _ = validate_123(p1,p2,p3,hist,st)
            if not ok:
                continue
            pred = "CALL" if st == "up" else "PUT"
            nxt = candles[i+1]
            win = (nxt["c"] > nxt["o"] and pred == "CALL") or (nxt["c"] < nxt["o"] and pred == "PUT")
            wins.append(win)
        if not wins:
            return 100
        recent = wins[-BACKTEST_SIGNALS:]
        return int(round(sum(1 for w in recent if w) / len(recent) * 100))
    except Exception as e:
        log(f"backtest error: {e}")
        return 100


# =================== ANALYZE ===================
def analyze_pair(symbol):
    out = {
        "pair": symbol,
        "signal": "SIDEWAYS",
        "status": "NO TRADE",
        "accuracy": 100,
        "cwrv": "No",
        "cwrv_conf": 0,
        "wave": "unknown",
        "candles": [],
        "why": ""
    }
    try:
        candles = fetch_candles(symbol)
        if not candles or len(candles) < 40:
            out["why"] = "insufficient data"
            return out

        out["candles"] = candles[-60:]
        swings = zigzag_swings(candles)
        trend, tr = market_structure(swings)
        is_sideways = sideways_filter(candles)
        p1,p2,p3,findmsg = find_123_points_from_swings(swings)
        valid, valmsg = validate_123(p1,p2,p3,candles,trend) if (p1 and p2 and p3) else (False, findmsg)
        pat = detect_pattern(candles)
        wave_label = elliott_type_and_wave(swings)
        accuracy = backtest_accuracy(candles, swings, trend)

        # confidence blending: CWRV + candle + wave
        conf = 0
        if valid: conf += 60
        if pat in ("bullish_engulf", "pin_bottom") and trend == "up": conf += 10
        if pat in ("bearish_engulf", "pin_top") and trend == "down": conf += 10
        if "Wave 5 (Impulse" in wave_label:
            conf -= 10   # late impulse slightly riskier
        if "Triangle" in wave_label:
            conf -= 10   # consolidation
        conf += int((accuracy - 70) * 0.3)
        conf = max(0, min(100, int(round(conf))))

        # final decision
        if trend == "up" and valid and not is_sideways:
            signal = "CALL"
        elif trend == "down" and valid and not is_sideways:
            signal = "PUT"
        else:
            signal = "SIDEWAYS"

        if signal == "SIDEWAYS" or is_sideways:
            status = "NO TRADE"
        else:
            if conf >= 65 and accuracy >= 75:
                status = "TRADE"
            elif conf >= 45 and accuracy >= 55:
                status = "RISKY"
            else:
                status = "NO TRADE"

        out.update({
            "signal": signal,
            "status": status,
            "accuracy": accuracy,
            "cwrv": "Yes" if valid else "No",
            "cwrv_conf": conf,
            "wave": wave_label,
            "why": f"trend={trend} ({tr}); find={findmsg}; validate={valmsg}; pat={pat}; swings={len(swings)}; sideways={is_sideways}"
        })
        return out
    except Exception as e:
        log(f"analyze_pair error {symbol}: {e}\n{traceback.format_exc()}")
        out["why"] = f"analysis error: {e}"
        return out


# =================== ROUTES ===================
@app.route("/")
def index():
    try:
        return render_template("index.html", pairs=PAIRS)
    except Exception:
        return "<h1>CWRV App</h1><p>Use /analyze</p>"

@app.route("/health")
def health():
    return jsonify({"ok": True, "time_utc": datetime.utcnow().isoformat()})

@app.route("/analyze", methods=["GET"])
def analyze():
    try:
        pairs = request.args.getlist("pair") or PAIRS
        results = []
        for s in pairs:
            res = analyze_pair(s)
            results.append(res if isinstance(res, dict) else {
                "pair": s, "signal": "SIDEWAYS", "status": "NO TRADE",
                "accuracy": 100, "cwrv": "No", "cwrv_conf": 0, "wave": "unknown",
                "why": "invalid"
            })
        return jsonify({"results": results})
    except Exception as e:
        log(f"/analyze error: {e}\n{traceback.format_exc()}")
        return jsonify({"results": [], "error": str(e)}), 500

@app.route("/debug", methods=["GET"])
def debug():
    pair = request.args.get("pair", PAIRS[0])
    try:
        candles = fetch_candles(pair)
        if not candles:
            return jsonify({"error": "no data"})
        swings = zigzag_swings(candles)
        p1,p2,p3,findmsg = find_123_points_from_swings(swings)
        return jsonify({
            "pair": pair,
            "swings": swings,
            "p1": p1, "p2": p2, "p3": p3,
            "find": findmsg,
            "elliott": elliott_type_and_wave(swings)
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log(f"Starting app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
