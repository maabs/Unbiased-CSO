# %%
import re
import time
import math
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests
from scipy.io import savemat

# %%
# ============================================================
# STEP 1: Parse iShares XLS → yfinance tickers
# ============================================================

xls_path = "iShares-STOXX-Europe-600-UCITS-ETF-DE-EUR-Dist_fund.xls"

def parse_ishares_xls(filepath: str) -> pd.DataFrame:
    """
    Parses the iShares STOXX Europe 600 holdings XLS (XML-based).
    Returns a DataFrame with columns:
        ticker, name, sector, location, exchange, currency, yf_ticker
    """
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()

    ns = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
    root = ET.fromstring(content)
    ws    = root.find(".//ss:Worksheet", ns)
    table = ws.find("ss:Table", ns)
    rows  = table.findall("ss:Row", ns)

    records = []
    for row in rows[8:]:          # row 7 = header, rows 8+ = data
        cells = row.findall("ss:Cell", ns)
        vals  = [c.find("ss:Data", ns).text if c.find("ss:Data", ns) is not None else ""
                 for c in cells]
        if len(vals) >= 12 and vals[3] == "Aktien":   # equity rows only
            records.append({
                "ticker":   vals[0],
                "name":     vals[1],
                "sector":   vals[2],
                "location": vals[9],
                "exchange": vals[10],
                "currency": vals[11],
            })

    df = pd.DataFrame(records)

    EXCHANGE_SUFFIX = {
        "London Stock Exchange":                         ".L",
        "Nyse Euronext - Euronext Paris":                ".PA",
        "Xetra":                                         ".DE",
        "SIX Swiss Exchange":                            ".SW",
        "Nasdaq Omx Nordic":                             ".ST",
        "Borsa Italiana":                                ".MI",
        "Euronext Amsterdam":                            ".AS",
        "Bolsa De Madrid":                               ".MC",
        "Omx Nordic Exchange Copenhagen A/S":            ".CO",
        "Oslo Bors Asa":                                 ".OL",
        "Nyse Euronext - Euronext Brussels":             ".BR",
        "Nasdaq Omx Helsinki Ltd.":                      ".HE",
        "Warsaw Stock Exchange/Equities/Main Market":    ".WA",
        "Wiener Boerse Ag":                              ".VI",
        "Irish Stock Exchange - All Market":             ".IR",
        "Nyse Euronext - Euronext Lisbon":               ".LS",
    }

    df["yf_ticker"] = df["ticker"] + df["exchange"].map(EXCHANGE_SUFFIX).fillna("")

    # Fix 1: spaces in share-class tickers → hyphens  e.g. "NOVO B.CO" → "NOVO-B.CO"
    df["yf_ticker"] = df["yf_ticker"].str.replace(r"\s+", "-", regex=True)

    # Fix 2: double dots from UK tickers ending in "."  e.g. "RR..L" → "RR.L"
    df["yf_ticker"] = df["yf_ticker"].str.replace(r"\.\.", ".", regex=True)

    unmapped = df[df["yf_ticker"] == df["ticker"]]
    if len(unmapped):
        print(f"WARNING: {len(unmapped)} tickers had no exchange suffix:\n",
              unmapped[["ticker", "exchange"]].to_string())

    return df


meta       = parse_ishares_xls(xls_path)
tickers_yf = meta["yf_ticker"].drop_duplicates().tolist()

# Save ticker list (mirrors sp500_tickers_yahoo_top400.csv)
pd.Series(tickers_yf, name="ticker").to_csv("stoxx600_tickers_yahoo.csv", index=False)
with open("stoxx600_tickers_yahoo.txt", "w") as f:
    f.write("\n".join(tickers_yf) + "\n")

print("Saved", len(tickers_yf), "tickers.")
print("Example:", tickers_yf[:15])

# %%
# ============================================================
# STEP 2: User inputs
# ============================================================

start = "2007-10-01"
end   = "2026-01-31"

MIN_NONMISSING_RETS = 1000
MIN_ROW_NONMISSING  = 40
MAX_IDENTICAL_RUN   = 10

CHUNK_SIZE = 80
SLEEP      = 1.0

out_mat = "stoxx600_daily_adj_close_and_rets.mat"

session = requests.Session(impersonate="chrome")

# Reload from CSV (mirrors S&P script)
tickers_yf = pd.read_csv("stoxx600_tickers_yahoo.csv")["ticker"].astype(str).str.strip().tolist()
tickers_yf = pd.Index(tickers_yf).drop_duplicates().tolist()
print("Universe size:", len(tickers_yf))
print("Example:", tickers_yf[:10])

# ============================================================
# STEP 3: Download helpers
# ============================================================

def download_chunk(tickers):
    px = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        session=session,
        threads=True,
        progress=False,
    )
    return px


def extract_field(px, field):
    """
    Handles both common yfinance MultiIndex layouts:
      - group_by="ticker": columns (TICKER, FIELD)
      - sometimes:         columns (FIELD, TICKER)
    """
    if not isinstance(px.columns, pd.MultiIndex):
        if field not in px.columns:
            raise KeyError(f"Field '{field}' not found. Available: {list(px.columns)}")
        return px[[field]]

    lvl0 = px.columns.get_level_values(0)
    lvl1 = px.columns.get_level_values(1)

    if field in set(lvl1):
        return px.xs(field, axis=1, level=1)
    if field in set(lvl0):
        return px.xs(field, axis=1, level=0)

    raise KeyError(
        f"Field '{field}' not found in MultiIndex. Levels: {set(lvl0)} / {set(lvl1)}"
    )


# ============================================================
# STEP 4: Download in batches
# ============================================================

adj_list = []
vol_list = []

n_batches = math.ceil(len(tickers_yf) / CHUNK_SIZE)

for i in range(0, len(tickers_yf), CHUNK_SIZE):
    b     = i // CHUNK_SIZE + 1
    batch = tickers_yf[i : i + CHUNK_SIZE]
    print(f"Downloading batch {b}/{n_batches} ({len(batch)} tickers)")

    try:
        px = download_chunk(batch)
        adj_list.append(extract_field(px, "Adj Close"))
        vol_list.append(extract_field(px, "Volume"))
    except Exception as e:
        print("Batch failed:", e)
        continue

    time.sleep(SLEEP)

adj_all = pd.concat(adj_list, axis=1)
vol_all = pd.concat(vol_list, axis=1)

# Remove duplicate index rows + align
adj_all = adj_all.loc[~adj_all.index.duplicated()].sort_index()
vol_all = vol_all.reindex(adj_all.index)

# Drop duplicate columns
adj_all = adj_all.loc[:, ~adj_all.columns.duplicated()]
vol_all = vol_all.reindex(columns=adj_all.columns)

print("Downloaded shape (adj):", adj_all.shape)

# ============================================================
# STEP 5: Returns + cleaning
# ============================================================

rets = np.log(adj_all).diff()
rets = rets.dropna(how="all")
rets = rets.dropna(thresh=MIN_ROW_NONMISSING, axis=0)

# Remove stocks with too few observations
non_missing_counts = rets.notna().sum()
rets = rets.loc[:, non_missing_counts >= MIN_NONMISSING_RETS]

# Align prices to cleaned returns
adj_clean = adj_all.reindex(index=rets.index, columns=rets.columns)

print("After cleaning:")
print("Returns shape:", rets.shape)

# ============================================================
# STEP 6: Drop stale / flat series
# ============================================================

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
        print("Dropping (stale returns):", col)

rets      = rets[cols_to_keep]
adj_clean = adj_clean[cols_to_keep]

print("Final shape:", rets.shape)

# %%
# ============================================================
# STEP 7: Export to .mat
# ============================================================

T = rets.shape[0]
N = rets.shape[1]

data_d    = rets.to_numpy(dtype=np.float64).T          # [N × T]
adj_d     = adj_clean.to_numpy(dtype=np.float64).T     # [N × T]
time_d    = rets.index.strftime("%Y-%m-%d").to_numpy(dtype=object).reshape(1, T)
tickers_d = np.array(rets.columns.tolist(), dtype=object).reshape(1, N)

mat_struct = {
    "time_d":     time_d,
    "tickers_d":  tickers_d,
    "data_d":     data_d,   # log returns  [N × T]
    "adj_d":      adj_d,    # adj close    [N × T]
}

savemat(out_mat, mat_struct, do_compression=True)
print("MAT file saved:", out_mat)

# %%
print(data_d.shape)