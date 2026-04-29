#%%
import re
import math
import time
import numpy as np
import pandas as pd
import yfinance as yf
import io
from curl_cffi import requests
from scipy.io import savemat

#%%
import pandas as pd
import numpy as np
import re

html_path = "DetailedDaily_en.html"

# read all tables; pick the one that contains numeric symbols
tables = pd.read_html(html_path)

def looks_like_symbols(col):
    # True if many entries are 3-5 digit numbers
    s = col.astype(str).str.strip()
    ok = s.str.match(r"^\d{3,5}$", na=False).mean()
    return ok > 0.5

target = None
for df in tables:
    for c in df.columns:
        if looks_like_symbols(df[c]):
            target = df.copy()
            sym_col = c
            break
    if target is not None:
        break

if target is None:
    raise RuntimeError("Could not find the symbol table in the HTML.")

symbols = target[sym_col].astype(str).str.extract(r"(\d{3,5})", expand=False)
symbols = symbols.dropna().astype(int).astype(str).str.zfill(4)

tickers_yahoo = symbols + ".SR"

# Optional: drop duplicates, keep order
tickers_yahoo = pd.Index(tickers_yahoo).drop_duplicates().to_list()

pd.Series(tickers_yahoo, name="ticker").to_csv("tadawul_tickers_yahoo.csv", index=False)
with open("tadawul_tickers_yahoo.txt", "w") as f:
    f.write("\n".join(tickers_yahoo))

print("Saved", len(tickers_yahoo), "tickers.")
print("Example:", tickers_yahoo[:10])
#%%
def code_to_yahoo_sr(code):
    # code can be int or string; enforce 4-digit padding
    code_str = str(code).strip()
    code_str = code_str.zfill(4) if code_str.isdigit() else code_str
    return f"{code_str}.SR"

def yahoo_name(ticker, session=None):
    """
    Pull a name from Yahoo via yfinance. This can be slow if you call it 200 times.
    Use sparingly (e.g., after you already narrowed to top 50).
    """
    try:
        t = yf.Ticker(ticker, session=session)
        info = t.get_info()  # may be slow
        return info.get("shortName") or info.get("longName") or ""
    except Exception:
        return ""

# ---------- (A) Get a Tadawul universe ----------
# You have 2 realistic options:
#  1) Scrape Saudi Exchange page(s) containing a table of listed companies/symbols (preferred)
#  2) Maintain a cached CSV of Tadawul codes + names, update periodically

# Placeholder: you should replace URL with the Saudi Exchange page that contains the full listed companies table.
# Many such pages are dynamic; if read_html fails, you’ll need a one-time manual download/export from the site and cache it.
URL_LISTED = "https://www.saudiexchange.sa/wps/portal/saudiexchange/ourmarkets/main-market-watch/"
#pd.read_html(URL_LISTED)


try:
    tables = pd.read_html(URL_LISTED)
    # Inspect tables to find the one with "Symbol"/"Company"/"Code"
    # For example:
    df0 = max(tables, key=lambda d: d.shape[0])
    print("Candidate table columns:", df0.columns)
    # Suppose the symbol column is named like "Symbol" or "رمز"
    # You must adapt this selection to what you see:
    sym_col = None
    for c in df0.columns:
        if str(c).lower() in ["symbol", "code"] or "رمز" in str(c):
            sym_col = c
            break
    if sym_col is None:
        raise RuntimeError("Could not find a symbol/code column in scraped table.")
    codes = df0[sym_col].astype(str).tolist()
except Exception as e:
    print("read_html failed on Saudi Exchange page (likely dynamic).")
    print("Action: export the listed companies table once to CSV and load it here.")
    raise

tickers_all = [code_to_yahoo_sr(c) for c in codes]
tickers_all = sorted(set(tickers_all))
print("Universe size (raw):", len(tickers_all))

# ---------- (B) Validate on Yahoo by attempting a tiny download ----------
# (Many invalid tickers will return all-NaN)
probe = yf.download(
    tickers_all[:200],  # probe first 200; later you can probe in batches
    start="2010-01-01",
    end="2010-03-01",
    group_by="ticker",
    auto_adjust=False,
    session=session,
    threads=True
)
print("Probe downloaded.")


#%%
# -----------------------------
# 0) Settings
# -----------------------------
start = "2010-01-01"
end   = None  # or "2025-01-01"

MIN_NONMISSING_RETS = 1000     # like your EuroStoxx filter
MIN_ROW_NONMISSING  = 40       # like your dropna(thresh=40)
MAX_IDENTICAL_RUN   = 10       # like your identical-run filter

TOPK = 50

OUT_MAT_PANEL = "tadawul_top50_liquid_daily_adj_close_and_rets.mat"
OUT_MAT_ADJ   = "tadawul_top50_liquid_daily_adj_close_clean.mat"
OUT_CSV_RETS  = "tadawul_top50_liquid_daily_logrets_clean.csv"
OUT_CSV_ADJ   = "tadawul_top50_liquid_daily_adj_close_clean.csv"

# A simple chunking to reduce Yahoo throttling
CHUNK_SIZE = 40
SLEEP_BETWEEN_CHUNKS = 1.0

# -----------------------------
# 1) Get Tadawul universe (symbols) and convert to Yahoo (.SR)
# -----------------------------
# This page provides a long list of Tadawul symbols (numeric).
# We convert to Yahoo tickers by appending ".SR".
# Source example: 2222.SR exists on Yahoo Finance.  (Saudi Aramco)  [oai_citation:1‡Yahoo Finance](https://finance.yahoo.com/quote/2222.SR/holders/?utm_source=chatgpt.com)
# Symbol list source: stockanalysis.  [oai_citation:2‡StockAnalysis](https://stockanalysis.com/list/saudi-stock-exchange/?utm_source=chatgpt.com)
UNIVERSE_URL = "https://stockanalysis.com/stocks/country/saudi-arabia/"
html = requests.get(UNIVERSE_URL, impersonate="chrome").text
print(html)
# Extract 4-digit symbols from the table
# (Tadawul symbols are typically 4 digits; adjust if you want 5-digit etc.)
symbols = sorted(set(re.findall(r"\b\d{4}\b", html)))

# Convert to Yahoo tickers
tickers_all = [f"{s}.SR" for s in symbols]

print(f"Universe size (from scrape): {len(tickers_all)}")
print(tickers_all)
#%%
# -----------------------------
# 2) Download helper (Adj Close + Volume)
# -----------------------------
def yf_download_chunk(tickers, start, end=None, session=None):
    px = yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        session=session,
        progress=False,
    )
    return px

def extract_field(px, field):
    """
    Returns a DataFrame indexed by date with columns=tickers.
    Supports both MultiIndex (ticker, field) and single-index.
    """
    if isinstance(px.columns, pd.MultiIndex):
        if field in px.columns.get_level_values(1):
            return px.xs(field, axis=1, level=1)
        else:
            raise KeyError(f"Field {field} not in MultiIndex columns.")
    else:
        # single ticker case: px has columns like ["Open","High",...]
        if field in px.columns:
            return px[field].to_frame()
        raise KeyError(f"Field {field} not in columns.")

# -----------------------------
# 3) Download the full universe in chunks (Adj Close + Volume)
# -----------------------------
session = requests.Session(impersonate="chrome")

adj_list = []
vol_list = []

for i in range(0, len(tickers_all), CHUNK_SIZE):
    chunk = tickers_all[i:i+CHUNK_SIZE]
    print(f"Downloading chunk {i//CHUNK_SIZE + 1}/{math.ceil(len(tickers_all)/CHUNK_SIZE)} ...")

    px = yf_download_chunk(chunk, start=start, end=end, session=session)

    # Adj Close and Volume
    try:
        adj = extract_field(px, "Adj Close")
    except KeyError:
        # if no Adj Close at all, skip chunk
        continue

    try:
        vol = extract_field(px, "Volume")
    except KeyError:
        # volume sometimes missing; create empty with same shape
        vol = adj.copy()*np.nan

    adj_list.append(adj)
    vol_list.append(vol)

    time.sleep(SLEEP_BETWEEN_CHUNKS)

adj_all = pd.concat(adj_list, axis=1)
vol_all = pd.concat(vol_list, axis=1)

# Keep unique columns and align indices
adj_all = adj_all.loc[~adj_all.index.duplicated()].sort_index()
vol_all = vol_all.loc[adj_all.index, adj_all.columns]

print("Downloaded shapes:")
print("  adj_all:", adj_all.shape)
print("  vol_all:", vol_all.shape)

# -----------------------------
# 4) Liquidity metric and Top-50 selection
# -----------------------------
# Use turnover proxy: AdjClose * Volume (in local currency units/day).
# Robust aggregator: median daily turnover over the sample.
turnover = adj_all * vol_all

# Require enough data to be meaningful
turnover_ok = turnover.notna().sum() >= MIN_NONMISSING_RETS
turnover = turnover.loc[:, turnover_ok]

liq = turnover.median(axis=0, skipna=True)   # one number per ticker
liq = liq.dropna().sort_values(ascending=False)

top_tickers = liq.index[:TOPK].tolist()

print(f"Top-{TOPK} tickers selected.")
print(top_tickers[:10], "...")

# Restrict to top tickers
adj = adj_all[top_tickers].copy()
vol = vol_all[top_tickers].copy()

# -----------------------------
# 5) Build log-returns and apply your cleaning steps
# -----------------------------
rets = np.log(adj).diff()

# same cleaning style you used
rets = rets.dropna(how="all")
rets = rets.dropna(thresh=MIN_ROW_NONMISSING, axis=0)

# filter out columns with too many missing values
non_missing_counts = rets.notna().sum()
rets = rets.loc[:, non_missing_counts >= MIN_NONMISSING_RETS]

# keep the same column order after filtering
adj = adj.reindex(index=rets.index, columns=rets.columns)
vol = vol.reindex(index=rets.index, columns=rets.columns)

def max_consecutive_identical(series):
    max_count = count = 1
    previous_value = None
    for value in series:
        if value == previous_value:
            count += 1
        else:
            max_count = max(max_count, count)
            count = 1
            previous_value = value
    return max(max_count, count)

cols_to_keep = []
for col in rets.columns:
    if max_consecutive_identical(rets[col].dropna()) <= MAX_IDENTICAL_RUN:
        cols_to_keep.append(col)
    else:
        print("Dropping due to identical-run:", col)

rets = rets[cols_to_keep]
adj  = adj[cols_to_keep]
vol  = vol[cols_to_keep]

print("After cleaning:")
print("  rets:", rets.shape)
print("  adj :", adj.shape)

# -----------------------------
# 6) Export to MATLAB .mat in your preferred layout [N, T]
# -----------------------------
T = rets.shape[0]
N = rets.shape[1]

data_q = rets.to_numpy(dtype=np.float64).T  # [N, T]
adj_q  = adj.to_numpy(dtype=np.float64).T   # [N, T]

time_q = rets.index.strftime("%Y-%m-%d").to_numpy(dtype=object).reshape(1, T)
tickers_q = np.array(rets.columns.tolist(), dtype=object).reshape(1, N)

mat_panel = {
    "time_q": time_q,
    "tickers_q": tickers_q,
    "data_q": data_q,   # log returns
    "adj_q": adj_q,     # adjusted close
}
savemat(OUT_MAT_PANEL, mat_panel, do_compression=True)
print("Saved:", OUT_MAT_PANEL)

mat_adj = {
    "time_q": time_q,
    "tickers_q": tickers_q,
    "adj_q": adj_q,
}
savemat(OUT_MAT_ADJ, mat_adj, do_compression=True)
print("Saved:", OUT_MAT_ADJ)

# Optional CSVs (same ordering)
rets.to_csv(OUT_CSV_RETS, index=True)
adj.to_csv(OUT_CSV_ADJ, index=True)
print("Saved CSVs:", OUT_CSV_RETS, "and", OUT_CSV_ADJ)