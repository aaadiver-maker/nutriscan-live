"""
Stock Shares Analysis & Recommendation Agent.

Standalone tool, unrelated to the NutriScan food scanner in app.py. Pulls
price history and fundamentals for a ticker via yfinance (no API key
needed), scores it on technical and fundamental signals, and prints a
buy/hold/sell-style recommendation with the reasoning behind it.

Not financial advice - this is a rule-based heuristic over public data,
not investment guidance.

Run with:  python stock_agent.py AAPL
            python stock_agent.py AAPL MSFT NVDA --period 1y
"""
import argparse
import sys
from dataclasses import dataclass, field

import numpy as np
import yfinance as yf

RECOMMENDATIONS = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]


@dataclass
class Signal:
    name: str
    score: int  # -2..+2, contribution to the overall recommendation
    detail: str


@dataclass
class AnalysisResult:
    ticker: str
    price: float
    technical_signals: list = field(default_factory=list)
    fundamental_signals: list = field(default_factory=list)

    @property
    def signals(self):
        return self.technical_signals + self.fundamental_signals

    @property
    def total_score(self):
        return sum(s.score for s in self.signals)

    @property
    def recommendation(self):
        # Map the summed signal score onto the 5-point scale. Max possible
        # score is roughly +-16 across all signals below, so these cutoffs
        # keep "Hold" for genuinely mixed/neutral cases rather than only
        # for a score of exactly zero.
        score = self.total_score
        if score >= 6:
            return "Strong Buy"
        if score >= 2:
            return "Buy"
        if score > -2:
            return "Hold"
        if score > -6:
            return "Sell"
        return "Strong Sell"


def fetch_history(ticker, period="6mo"):
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        raise ValueError(f"No price history found for '{ticker}'. Check the symbol.")
    return hist


def compute_rsi(closes, window=14):
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # A window with no losing days gives avg_loss == 0, which the divide
    # above turns into NaN rather than the conventional RSI of 100.
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def compute_macd(closes, fast=12, slow=26, signal=9):
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def analyze_technicals(hist):
    closes = hist["Close"]
    signals = []
    price = float(closes.iloc[-1])

    if len(closes) >= 50:
        sma50 = closes.rolling(50).mean().iloc[-1]
        gap_pct = (price - sma50) / sma50 * 100
        score = 1 if price > sma50 else -1
        signals.append(Signal(
            "Price vs SMA50", score,
            f"Price ${price:.2f} is {abs(gap_pct):.1f}% "
            f"{'above' if price > sma50 else 'below'} its 50-day average (${sma50:.2f}).",
        ))

    if len(closes) >= 200:
        sma200 = closes.rolling(200).mean().iloc[-1]
        sma50 = closes.rolling(50).mean().iloc[-1]
        golden_cross = sma50 > sma200
        signals.append(Signal(
            "Golden/Death cross", 1 if golden_cross else -1,
            f"50-day average is {'above' if golden_cross else 'below'} the 200-day average "
            f"({'golden cross - bullish' if golden_cross else 'death cross - bearish'}).",
        ))

    rsi = compute_rsi(closes)
    if not rsi.empty and not np.isnan(rsi.iloc[-1]):
        last_rsi = float(rsi.iloc[-1])
        if last_rsi >= 70:
            score, note = -1, "overbought"
        elif last_rsi <= 30:
            score, note = 1, "oversold, possible rebound"
        else:
            score, note = 0, "neutral range"
        signals.append(Signal("RSI (14)", score, f"RSI is {last_rsi:.1f} ({note})."))

    macd_line, signal_line = compute_macd(closes)
    if len(macd_line) >= 2:
        macd_now, sig_now = macd_line.iloc[-1], signal_line.iloc[-1]
        macd_prev, sig_prev = macd_line.iloc[-2], signal_line.iloc[-2]
        crossed_up = macd_prev <= sig_prev and macd_now > sig_now
        crossed_down = macd_prev >= sig_prev and macd_now < sig_now
        if crossed_up:
            score, note = 2, "just crossed above its signal line - bullish crossover"
        elif crossed_down:
            score, note = -2, "just crossed below its signal line - bearish crossover"
        elif macd_now > sig_now:
            score, note = 1, "above its signal line"
        else:
            score, note = -1, "below its signal line"
        signals.append(Signal("MACD", score, f"MACD is {note}."))

    returns = closes.pct_change().dropna()
    if len(returns) >= 20:
        annualized_vol = float(returns.std() * np.sqrt(252) * 100)
        # Volatility is informational only (score 0) - high volatility isn't
        # inherently bearish, it's a risk-sizing consideration, not a signal.
        signals.append(Signal(
            "Volatility", 0,
            f"Annualized volatility is ~{annualized_vol:.1f}%.",
        ))

    return signals, price


def _pct(x):
    return None if x is None else x * 100


def analyze_fundamentals(ticker):
    info = yf.Ticker(ticker).info
    signals = []

    pe = info.get("trailingPE")
    if pe is not None and pe > 0:
        if pe < 15:
            score, note = 1, "low - potentially undervalued"
        elif pe <= 25:
            score, note = 0, "moderate"
        else:
            score, note = -1, "high - potentially overvalued"
        signals.append(Signal("P/E ratio", score, f"Trailing P/E is {pe:.1f} ({note})."))

    peg = info.get("trailingPegRatio") or info.get("pegRatio")
    if peg is not None and peg > 0:
        score = 1 if peg < 1 else (-1 if peg > 2 else 0)
        signals.append(Signal("PEG ratio", score, f"PEG ratio is {peg:.2f}."))

    rev_growth = _pct(info.get("revenueGrowth"))
    if rev_growth is not None:
        score = 1 if rev_growth > 5 else (-1 if rev_growth < 0 else 0)
        signals.append(Signal("Revenue growth", score, f"YoY revenue growth is {rev_growth:.1f}%."))

    earn_growth = _pct(info.get("earningsGrowth"))
    if earn_growth is not None:
        score = 1 if earn_growth > 5 else (-1 if earn_growth < 0 else 0)
        signals.append(Signal("Earnings growth", score, f"YoY earnings growth is {earn_growth:.1f}%."))

    margins = _pct(info.get("profitMargins"))
    if margins is not None:
        score = 1 if margins > 15 else (-1 if margins < 0 else 0)
        signals.append(Signal("Profit margin", score, f"Net profit margin is {margins:.1f}%."))

    roe = _pct(info.get("returnOnEquity"))
    if roe is not None:
        score = 1 if roe > 15 else (-1 if roe < 0 else 0)
        signals.append(Signal("Return on equity", score, f"ROE is {roe:.1f}%."))

    debt_to_equity = info.get("debtToEquity")
    if debt_to_equity is not None:
        # yfinance reports this as a plain ratio*100 (e.g. 150 == 1.5x)
        ratio = debt_to_equity / 100
        score = 1 if ratio < 1 else (-1 if ratio > 2 else 0)
        signals.append(Signal("Debt/Equity", score, f"Debt-to-equity is {ratio:.2f}x."))

    div_yield = _pct(info.get("dividendYield"))
    if div_yield:
        signals.append(Signal(
            "Dividend yield", 0,
            f"Dividend yield is {div_yield:.2f}% (informational, not scored).",
        ))

    return signals


def analyze(ticker, period="6mo"):
    hist = fetch_history(ticker, period)
    tech_signals, price = analyze_technicals(hist)
    fund_signals = analyze_fundamentals(ticker)
    result = AnalysisResult(ticker=ticker.upper(), price=price)
    result.technical_signals = tech_signals
    result.fundamental_signals = fund_signals
    return result


def print_report(result):
    print(f"\n{'=' * 60}")
    print(f"{result.ticker}  -  ${result.price:.2f}")
    print(f"{'=' * 60}")

    print("\nTechnical signals:")
    for s in result.technical_signals:
        print(f"  [{s.score:+d}] {s.name}: {s.detail}")

    print("\nFundamental signals:")
    if result.fundamental_signals:
        for s in result.fundamental_signals:
            print(f"  [{s.score:+d}] {s.name}: {s.detail}")
    else:
        print("  No fundamental data available for this ticker.")

    print(f"\nOverall score: {result.total_score:+d}")
    print(f"Recommendation: {result.recommendation}")
    print("\nThis is a rule-based heuristic over public market data, not financial advice.")


def main():
    parser = argparse.ArgumentParser(description="Stock shares analysis & recommendation agent.")
    parser.add_argument("tickers", nargs="+", help="One or more ticker symbols, e.g. AAPL MSFT")
    parser.add_argument("--period", default="6mo",
                         help="History window for technicals, e.g. 1mo, 6mo, 1y, 2y (default: 6mo)")
    args = parser.parse_args()

    exit_code = 0
    for ticker in args.tickers:
        try:
            result = analyze(ticker, args.period)
            print_report(result)
        except Exception as exc:
            print(f"\n{ticker.upper()}: failed to analyze - {exc}", file=sys.stderr)
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
