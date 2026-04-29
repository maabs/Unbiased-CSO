#%%
import pandas as pd
import numpy as np
import yfinance as yf
from curl_cffi import requests
from scipy.io import savemat

# 1) Put your tickers here (Yahoo-style)
#%%
# Date range, same as in DElaportas et al. (2023)

start = "2007-10-01"
end   = "2014-11-05"
ric_to_yahoo = {
    # Switzerland: Yahoo uses .SW (not .S)
    "ROG.S":  "ROG.SW",
    "NOVN.S": "NOVN.SW",
    "NESN.S": "NESN.SW",
    
    "ABBN.S": "ABBN.SW",
    "ZURN.S": "ZURN.SW",
    "CFR.S":  "CFR.SW",

    # France / Eurozone name conventions
    "TTEF.PA": "TTE.PA",    # TotalEnergies
    "ESLX.PA": "EL.PA",     # EssilorLuxottica
    "HRMS.PA": "RMS.PA",    # Hermès
    "LVMH.PA": "MC.PA",     # LVMH
    "BNPP.PA": "BNP.PA",    # BNP Paribas
    "OREP.PA": "OR.PA",     # L'Oréal
    "SCHN.PA": "SU.PA",     # Schneider Electric
    "AXAF.PA": "CS.PA",     # AXA
    "SGEF.PA": "SGO.PA",    # Saint-Gobain (common Yahoo: SGO.PA)
    "SASY.PA": "SAN.PA",    # Sanofi

    # Germany
    "SAPG.DE":  "SAP.DE",
    "SIEGn.DE": "SIE.DE",
    "DTEGn.DE": "DTE.DE",
    "ALVG.DE":  "ALV.DE",
    "MUVGN.DE": "MUV2.DE",  # Munich Re
    "RHMG.DE":  "RHM.DE",   # Rheinmetall

    # Italy
    "CRDI.MI": "UCG.MI",    # UniCredit
    "ENEI.MI": "ENEL.MI",   # Enel
    # Novo Nordisk (Copenhagen): RIC uses NOVOb.CO / NOVOB.CO style, Yahoo uses NOVO-B.CO
    "NOVOB.CO":  "NOVO-B.CO",
    "NOVOb.CO":  "NOVO-B.CO",

    # Munich Re / Siemens / Deutsche Telekom: RIC adds extra letters, Yahoo uses the note symbol
    "MUVGn.DE":  "MUV2.DE",   # Münchener Rück (Munich Re)  [oai_citation:0‡Yahoo Finance](https://finance.yahoo.com/quote/MUV2.DE/?utm_source=chatgpt.com)
    
       # Siemens  [oai_citation:2‡Yahoo Finance](https://finance.yahoo.com/quote/SIE.DE/?utm_source=chatgpt.com)

    # UBS: use UBSG.SW (Yahoo), not UBS.SW (your earlier mapping)
   

    # Shell: Yahoo uses SHELL.AS (Amsterdam), not SHEL.AS
    "SHEL.AS":   "SHELL.AS",  # Shell Amsterdam listing  [oai_citation:4‡Yahoo Finance](https://finance.yahoo.com/quote/SHELL.AS/?utm_source=chatgpt.com)

    # Air Liquide: RIC is AIRP.PA but Yahoo ticker is AI.PA
    "AIRP.PA":   "AI.PA",     # L’Air Liquide  [oai_citation:5‡Yahoo Finance](https://finance.yahoo.com/quote/AI.PA/?utm_source=chatgpt.com)
    
    
    
    "UBSG.S":   "UBSG.SW",
}


session = requests.Session(impersonate="chrome")

#ticker = yf.Ticker('...', session=session)
# Get Euro Stoxx 50 components as of 2014-11-05
tickers = [
    "ASML.AS","ROG.S","HSBA.L","AZN.L","NOVN.S","NESN.S","SIEGn.DE","SHEL.AS","SAPG.DE","SAN.MC",
    "NOVOb.CO","ALVG.DE","SCHN.PA","LVMH.PA","TTEF.PA","ULVR.L","BBVA.MC","UBSG.S","RR.L","IBE.MC",
    "ABBN.S","CRDI.MI","AIR.PA","SAF.PA","BATS.L","DTEGn.DE","BNPP.PA","OREP.PA","AIRP.PA","ISP.MI",
    "SASY.PA","GSK.L","CFR.S","ZURN.S","BP.L","RIO.L","RHMG.DE","ESLX.PA","INGA.AS","NG.L",
    "ENEI.MI","HRMS.PA","SGEF.PA","AXAF.PA","MUVGn.DE","ABI.BR","PRX.AS","REL.L","LSEG.L","DGE.L"
]
tickers = [ric_to_yahoo.get(t, t) for t in tickers]

session = requests.Session(impersonate="chrome")
px = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=False,session=session)

#%%
# Download adjusted close (best for returns due to splits/dividends)
#import os, sqlite3, traceback


#session = requests.Session(impersonate="chrome")
#px = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=False,session=session)
#%%
if isinstance(px.columns, pd.MultiIndex):
    # ticker is level 0, field is level 1
    if "Adj Close" in px.columns.get_level_values(1):
        adj = px.xs("Adj Close", axis=1, level=1)
    
else:
    adj = px["Adj Close"] 

rets = np.log(adj).diff()
rets = rets.dropna(how="all")
rets = rets.dropna(thresh=40, axis=0)
shape = rets.shape
# %%
## Filter out columns with too many missing values (e.g., less than 1000 non-missing entries)
non_missing_counts = rets.notna().sum()
non_missing_counts=non_missing_counts>=1000
rets=rets.loc[:,non_missing_counts]
shape = rets.shape

print(shape)
## Filter out the columns that have more than 10 consecutive days without change in 
## the return value.

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

    max_count = max(max_count, count)
    return max_count

cols_to_keep = []
for col in rets.columns:
    if max_consecutive_identical(rets[col].dropna()) <= 10:
        
        cols_to_keep.append(col)    
    else:
        print(col)
rets = rets[cols_to_keep]
print(rets.shape)
print(rets.columns)
# %%
#%%
T = rets.shape[0]
N = rets.shape[1]

# data_q: [N, T]  (stocks x time)  => transpose of (T x N)
data_q = rets.to_numpy(dtype=np.float64).T
# time_q: [1, T]
# simplest: store as ISO strings (MATLAB can parse if needed)
time_q = rets.index.strftime("%Y-%m-%d").to_numpy(dtype=object).reshape(1, T)
mat_struct = {"time_q": time_q, "data_q": data_q}
#%%
# This creates a .mat where `data_quarterly` is a struct with fields time_q, data_q
#savemat("adj_close_daily.mat", mat_struct, do_compression=True)
#rets.to_csv("eurostoxx50_daily_adj_close.csv")
# %%
adj_clean = adj.reindex(index=rets.index, columns=rets.columns)

# Optional sanity checks
assert list(adj_clean.columns) == list(rets.columns)
assert adj_clean.shape == rets.shape  # same T x N


# adj close [N, T]
adj_q = adj_clean.to_numpy(dtype=np.float64).T

# time [1, T] + tickers [1, N] (highly recommended)
time_q = rets.index.strftime("%Y-%m-%d").to_numpy(dtype=object).reshape(1, T)
tickers_q = np.array(rets.columns.tolist(), dtype=object).reshape(1, N)

mat_struct = {
    "time_q": time_q,
    "tickers_q": tickers_q,
    "data_q": data_q,      # log returns
    "adj_q": adj_q,        # adjusted close
}

savemat("eurostoxx50_daily_adj_close_and_rets.mat", mat_struct, do_compression=True)

# If you also want a standalone adj file:
#savemat("eurostoxx50_daily_adj_close.mat",
#        {"time_q": time_q, "tickers_q": tickers_q, "adj_q": adj_q},
#        do_compression=True)

# (Optional) also export csv in same order
adj_clean.to_csv("eurostoxx50_daily_adj_close_clean.csv")
#rets.to_csv("eurostoxx50_daily_logrets_clean.csv")
# %%

ref_csv = pd.read_csv("eurostoxx50_daily_log_returns.csv", nrows=0)
ref_tickers = ref_csv.columns.tolist()

print("Reference ticker count:", len(ref_tickers))

# --------------------------------------------------
# 2) Keep only tickers that exist in BOTH datasets
# --------------------------------------------------
common_tickers = [t for t in ref_tickers if t in rets.columns]

print("Common tickers:", len(common_tickers))

# Optional safety check
missing = set(ref_tickers) - set(rets.columns)
if len(missing) > 0:
    print("Tickers missing from current dataset:", missing)

# --------------------------------------------------
# 3) Reindex BOTH rets and adj to that ordering
# --------------------------------------------------
rets_ordered = rets.reindex(columns=common_tickers)
adj_ordered  = adj.reindex(index=rets.index, columns=common_tickers)

# Sanity checks
assert list(rets_ordered.columns) == common_tickers
assert list(adj_ordered.columns) == common_tickers

print("Final shape (T, N):", rets_ordered.shape)

# --------------------------------------------------
# 4) Convert to MATLAB format
# --------------------------------------------------
T = rets_ordered.shape[0]
N = rets_ordered.shape[1]

data_q = rets_ordered.to_numpy(dtype=np.float64).T   # [N, T]
adj_q  = adj_ordered.to_numpy(dtype=np.float64).T    # [N, T]

time_q = rets_ordered.index.strftime("%Y-%m-%d") \
         .to_numpy(dtype=object).reshape(1, T)

tickers_q = np.array(common_tickers, dtype=object).reshape(1, N)

mat_struct = {
    "time_q": time_q,
    "tickers_q": tickers_q,
    "data_q": data_q,
    "adj_q": adj_q,
}

savemat("eurostoxx50_daily_adj_close_and_rets.mat",
        mat_struct,
        do_compression=True)

print("MAT file saved with consistent ticker ordering.")


# %%
# --- Save cleaned adjusted close only (same ticker ordering) ---
T = adj_ordered.shape[0]
N = adj_ordered.shape[1]

adj_q  = adj_ordered.to_numpy(dtype=np.float64).T    # [N, T]
time_q = adj_ordered.index.strftime("%Y-%m-%d").to_numpy(dtype=object).reshape(1, T)
tickers_q = np.array(adj_ordered.columns.tolist(), dtype=object).reshape(1, N)

mat_adj = {
    "time_q": time_q,
    "tickers_q": tickers_q,
    "adj_q": adj_q,      # adjusted close, cleaned, ordered
}

savemat("eurostoxx50_daily_adj_close_clean.mat", mat_adj, do_compression=True)
print("Saved: eurostoxx50_daily_adj_close_clean.mat")# --- Save cleaned adjusted close only (same ticker ordering) ---
T = adj_ordered.shape[0]
N = adj_ordered.shape[1]

adj_q  = adj_ordered.to_numpy(dtype=np.float64).T    # [N, T]
time_q = adj_ordered.index.strftime("%Y-%m-%d").to_numpy(dtype=object).reshape(1, T)
tickers_q = np.array(adj_ordered.columns.tolist(), dtype=object).reshape(1, N)

mat_adj = {
    "time_q": time_q,
    "tickers_q": tickers_q,
    "adj_q": adj_q,      # adjusted close, cleaned, ordered
}

savemat("eurostoxx50_daily_adj_close_clean.mat", mat_adj, do_compression=True)
print("Saved: eurostoxx50_daily_adj_close_clean.mat")
# %%
