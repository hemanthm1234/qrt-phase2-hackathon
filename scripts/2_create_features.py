# %%
import pandas as pd

from technical_indicators import calculate_all_indicators_parallel
import warnings
warnings.filterwarnings("ignore")
import requests
from io import BytesIO
import os

# %%
BASE_DIR = os.path.dirname(os.getcwd()) 
DATA_DIR = os.path.join(BASE_DIR, "stores_created")

os.makedirs(DATA_DIR, exist_ok=True)

# %%
pv = pd.read_parquet("../yahoo_finance/all_prices_5000_tickers.parquet", engine="pyarrow")
# %%
# Calculate technical indicators using parellel processing
# Please read file 'technical_indicators.py' for details on the indicators being calculated
# Note this is a computationally intensive step and may take some time to complete

indicators = calculate_all_indicators_parallel(pv, n_jobs=-1)

# %%
# Downcasting to float32 to save memory, as we have a large number of features and rows
indicators = indicators.astype("float32")

# %%
# Save the indicators to a parquet file for later use in model training
indicators.to_parquet(os.path.join(DATA_DIR, "features.parquet"), compression="zstd", engine="pyarrow")



