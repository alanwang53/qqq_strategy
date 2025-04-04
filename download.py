from pathlib import Path
import pandas as pd
import yfinance as yf

DATA_PATH = Path("./financial_data.feather")

def download_data(START_DATE,END_DATE):
    """Download data only if local copy doesn't exist or needs updating"""
    # Try to load cached data
    if DATA_PATH.exists():
        try:
            df = pd.read_feather(DATA_PATH)
            df = df.set_index('Date')
            
            # Check if existing data covers our date range
            existing_start = df.index.min()
            existing_end = df.index.max()
            
            # Only download if we need older or newer data
            if existing_start > pd.to_datetime(START_DATE) or existing_end < pd.to_datetime(END_DATE):
                print("Existing data needs updating - downloading fresh copy")
                return _download_fresh_data(START_DATE,END_DATE)
                
            print("Loaded cached data from", DATA_PATH)
            return df
        
        except Exception as e:
            print("Error loading cached data:", e)
            return _download_fresh_data(START_DATE,END_DATE)
    
    # No cache exists - download fresh data
    return _download_fresh_data(START_DATE,END_DATE)

def _download_fresh_data(START_DATE, END_DATE):
    """Internal function to handle actual data downloading"""
    print("Downloading fresh data...")
    
    # Your existing download logic
    cls = yf.download("QQQ", start=START_DATE, end=END_DATE, auto_adjust=True)
    print(f"\n ndx available data point: {cls.index[0].strftime('%Y-%m-%d')}")
    print(f"\n ndx last available data point: {cls.index[-1].strftime('%Y-%m-%d')}")
    #qqq_price = cls["Close"]
    #qqq_price.name = "QQQ"  # Assign a unique name
     
    vix_cls = yf.download("^VIX", start=START_DATE, end=END_DATE, auto_adjust=True)
    vix     = vix_cls["Close"].dropna()
    vix.name = "VIX"
    print(f"\n vix available data point: {vix.index[0].strftime('%Y-%m-%d')}")

    gld_price = yf.download("GLD", start=START_DATE,end=END_DATE, auto_adjust=True) ["Close"].dropna()
    gld_price.name = "GLD"
    print(f"\n gld available data point: {gld_price.index[0].strftime('%Y-%m-%d')}")
    shy_price = yf.download("SHY", start=START_DATE, end=END_DATE,auto_adjust=True) ["Close"].dropna()
    shy_price.name = "SHY"
    print(f"\nshy available data point: {shy_price.index[0].strftime('%Y-%m-%d')}")
    
    
    
    df = pd.concat([  vix, gld_price, shy_price], axis=1).dropna()
    df["QQQ"]= cls["Close"].dropna()
    df["QQQ_OPEN"]= cls["Open"]
    df["QQQ_high"]= cls["High"]
    df["QQQ_low"] = cls["Low"]
    df["QQQ_Volume"] = cls["Volume"]
    
    df["SHY"] = shy_price.bfill()
    df["GLD"] = gld_price.bfill() 
    df["VIX_OPEN"] = vix_cls["Open"].dropna()
    # Print column names to verify uniqueness
    #print("Downloaded data with columns:")
    #print(df.columns)
    # Calculate daily returns
    df["qqq_returns"] = df["QQQ"].pct_change().dropna()
    df["Volume_MA_20"] = df["QQQ_Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["QQQ_Volume"] / df["Volume_MA_20"]
    df["Market_Return"] = df["QQQ"].pct_change()

    # Save to cache
    df.reset_index().to_feather(DATA_PATH)
    print("Saved fresh data to", DATA_PATH)
    
    return df