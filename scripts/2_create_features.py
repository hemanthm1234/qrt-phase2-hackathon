# %%
import pandas as pd

# For loading class on Google Colab
!curl -O https://raw.githubusercontent.com/yszanwar/phase2_qrt_challenge/refs/heads/main/scripts/technical_indicators.py


from technical_indicators import calculate_all_indicators_parallel
import warnings
warnings.filterwarnings("ignore")
import requests
from io import BytesIO
import os

# %%
BASE_DIR = os.path.dirname(os.getcwd()) 
DATA_DIR = os.path.join(BASE_DIR, "stores")

os.makedirs(DATA_DIR, exist_ok=True)

# %%
url = "https://github.com/yszanwar/phase2_qrt_challenge/releases/download/price_data/all_prices_5000_tickers.parquet"

response = requests.get(url)
pv = pd.read_parquet(BytesIO(response.content), engine="pyarrow")

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



