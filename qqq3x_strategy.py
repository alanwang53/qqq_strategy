
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import talib
from analysis import *
from download import *
START_DATE = "2004-11-18"     # QQQ 上市日
END_DATE = "2025-04-03"       # 回測結束日期
INITIAL_CAPITAL = 10000       # 初始資金
COMMISSION = 0.01            # 單次交易佣金率（0.1%）+ slippage 

TARGET_LEVERAGE = 3       # 目標槓桿倍數（2~3之間）
#SAFE_ASSET = "GLD"            # 避險資產代碼（可換為 "SHY"（短債）、"GLD"（黃金）、"CASH"（現金）等
MANAGEMENT_FEE = 0.0095       # 年度管理費（TQQQ 實際年費 0.95%）
ORI_DISCOUNT =  1          # encounter max drawa down 65% at beggining 
SAFE_RATIO = 0.2       # safe asset ratio
daily_fee =  MANAGEMENT_FEE  / 252
SMA_SHORT = 5 
SMA_LONG =  20
SMA_YEAR = 151

def backtest_strategy(df):
    
    df["QQQ_SMA_YEAR"] = talib.MA(df["QQQ"], timeperiod=SMA_YEAR)  # matype=1 表示SMMA
    df["QQQ_SMA_LONG"] = talib.MA(df["QQQ"], timeperiod=SMA_LONG)  # matype=1 表示SMMA
    df["QQQ_SMA_SHORT"] = talib.MA(df["QQQ"], timeperiod=SMA_SHORT)  # matype=1 表示SMMA
    df["GLD_SMA"] = df["GLD"].rolling(window=100, min_periods=1).mean() 
    # 生成信号（修复逻辑嵌套）
    cond  = ( (df["^VIX"]<21) & (df["QQQ_SMA_SHORT"]>0.99*df["QQQ_SMA_LONG"]))
    cond2 = ( df["^VIX"]>23)  | (df["QQQ_SMA_SHORT"]<0.94*df["QQQ_SMA_LONG"])
    cond4 = (df["QQQ"] >  (1.03 * df["QQQ_SMA_YEAR"]) )
    cond5 = (df["QQQ"] <  (0.99* df["QQQ_SMA_YEAR"]) )
    cond6 = (df["QQQ_SMA_SHORT"]<df["QQQ_SMA_LONG"])
    cond7 = (df["^VIX"]>66) 
    cond8 = (df["^VIX"]<60)
    
    cond8_open =  (df["VIX_OPEN"].shift(-1)<60)
    cond7_open =  (df["VIX_OPEN"].shift(-1)>66)
    cond11 = (df['VIX_OPEN'].shift(-1)<21)
    cond12 = (df['VIX_OPEN'].shift(-1)>23)
    cond13 = df['QQQ_OPEN'].shift(-1) > (1.03 * df["QQQ_SMA_YEAR"]) 
    cond14 = df['QQQ_OPEN'].shift(-1) < (0.99 * df["QQQ_SMA_YEAR"]) 
    df["Raw_Signal"] = np.select(
        condlist=[
            ( ((cond4&cond13)) | (cond&cond11) | (cond7_open&cond7) )   ,  # 条件1：高于1%阈值  
            ( ((cond5|cond14)) & (cond2|cond12)  & (cond8_open&cond8) )# 条件2：低于1%阈值 
        ],
        choicelist=[
            1,   # 满足条件1时信号为1（杠杆）
            -1   # 满足条件2时信号为-1（避险）
        ],
        default=np.nan  # 中间区域不改变信号
    )
    df["Signal"] = df["Raw_Signal"].ffill().fillna(-1)  # 前向填充中间区域

    
    #df["Filtered_Signal"] = df["Signal"].copy()
    #for i in range(3, len(df)):
    #    current_signal = df["Signal"].iloc[int(i)]
    #    prev_signals = df["Signal"].iloc[int(i)-3:int(i)-1]  # Previous 3 days (not including current)
        
        # If current signal is opposite to any of last 3 signals, keep previous filtered signal
    #    if any(prev_signals != current_signal):
    #        df["Filtered_Signal"].iloc[int(i)] = df["Filtered_Signal"].iloc[int(i)-1]
    #    else:
    #        df["Filtered_Signal"].iloc[int(i)] = current_signal
    #df["Signal"] = df["Filtered_Signal"]

    # 計算各資產收益率
    df["GLD_Return"] =  df["GLD"].pct_change().fillna(0)  # 避險資產
    df["SHY_Return"] =  df["SHY"].pct_change().fillna(0)
    df["SAFE_ASSET_Return"] = np.select(
        condlist=[
          df["GLD"]> df["GLD_SMA"] ,  # 条件1：高于1%阈值
          df["GLD"]< df["GLD_SMA"]
        ],
        choicelist=[
            df["GLD_Return"],   # gld is better
            df["SHY_Return"]   # shy is better
        ],
        default= df["SHY_Return"]  # 中间区域不改变信号
    )
    # 動態組合收益率（加權槓桿或避險）
    df["Leveraged_Return"] = TARGET_LEVERAGE * df["QQQ"].pct_change() - daily_fee
    
    df["Open_Return"] = (df["QQQ"] / df["QQQ_OPEN"] - 1) * TARGET_LEVERAGE - daily_fee
    
    df["Signal_Change"] = df["Signal"].diff().abs()
    df["Trade_Day"] = df["Signal_Change"].shift(1).eq(2).astype(int)
    
    
    PORT= (1-SAFE_RATIO)
    df["Strategy_Return"] = np.where(
        df["Signal"].shift(1) == 1,  # If we were in QQQ yesterday
        np.where(
            df["Trade_Day"] == 1,    # If trading today  
            df["Open_Return"]*(PORT),       # Use open-to-close return
            df["Leveraged_Return"]*PORT   # Else use normal close-to-close
        ),
        df["SAFE_ASSET_Return"]*PORT            # If we were in safe asset
    )
    
    
    df["Strategy_Return"] -= df["Trade_Day"] * COMMISSION *2* (1-SAFE_RATIO) 
    df["Strategy_Return"] +=  df["SAFE_ASSET_Return"]*SAFE_RATIO
    
    # 計算淨值曲線
    SMA_half = int(SMA_YEAR/2)
    df["Strategy_NAV"] = (1 + df["Strategy_Return"]).cumprod() * INITIAL_CAPITAL
    df = df.iloc[SMA_half:]
    
    
    
    return df

def strategy_today( vix_open_today, qqq_open_today):
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    today = datetime.today()

    # Calculate date 3 years ago
    three_years_ago = today - relativedelta(years=2)
    three_years_ago_str = three_years_ago.strftime("%Y-%m-%d")
    
    df = download_data(three_years_ago_str, today_str)
    df["QQQ_SMA_YEAR"] = talib.MA(df["QQQ"], timeperiod=SMA_YEAR)  # matype=1 表示SMMA
    df["QQQ_SMA_LONG"] = talib.MA(df["QQQ"], timeperiod=SMA_LONG)  # matype=1 表示SMMA
    df["QQQ_SMA_SHORT"] = talib.MA(df["QQQ"], timeperiod=SMA_SHORT)  # matype=1 表示SMMA
    df["GLD_SMA"] = df["GLD"].rolling(window=100, min_periods=1).mean() 
    # 生成信号（修复逻辑嵌套）
    cond  = ( (df["^VIX"]<21) & (df["QQQ_SMA_SHORT"]>0.99*df["QQQ_SMA_LONG"]))
    cond2 = ( df["^VIX"]>23)  | (df["QQQ_SMA_SHORT"]<0.94*df["QQQ_SMA_LONG"])
    cond4 = (df["QQQ"] >  (1.03 * df["QQQ_SMA_YEAR"]) )
    cond5 = (df["QQQ"] <  (0.99* df["QQQ_SMA_YEAR"]) )
    cond6 = (df["QQQ_SMA_SHORT"]<df["QQQ_SMA_LONG"])
    cond7 = (df["^VIX"]>66) 
    cond8 = (df["^VIX"]<60)
    
    cond8_open =  (df["VIX_OPEN"].shift(-1)<60)
    cond7_open =  (df["VIX_OPEN"].shift(-1)>66)
    cond11 = (df['VIX_OPEN'].shift(-1)<21)
    cond12 = (df['VIX_OPEN'].shift(-1)>23)
    cond13 = df['QQQ_OPEN'].shift(-1) > (1.03 * df["QQQ_SMA_YEAR"]) 
    cond14 = df['QQQ_OPEN'].shift(-1) < (0.99 * df["QQQ_SMA_YEAR"]) 
    
    cond13.iloc[-1] = qqq_open_today > (1.03 * df["QQQ_SMA_YEAR"].iloc[-1])
    cond14.iloc[-1] = qqq_open_today < (0.99 * df["QQQ_SMA_YEAR"].iloc[-1])
    
    cond8_open.iloc[-1] =  (vix_open_today<60)
    cond7_open.iloc[-1] =  (vix_open_today>66)
    cond11.iloc[-1] = (vix_open_today<21)
    cond12.iloc[-1] = (vix_open_today>23)
    
    df["Raw_Signal"] = np.select(
        condlist=[
            ( ((cond4&cond13)) | (cond&cond11) |   (cond7_open&cond7 ) )   ,  # 条件1：高于1%阈值  
            ( ((cond5|cond14)) & (cond2|cond12)  & (cond8_open&cond8) )# 条件2：低于1%阈值 
        ],
        choicelist=[
            1,   # 满足条件1时信号为1（杠杆）
            -1   # 满足条件2时信号为-1（避险）
        ],
        default=np.nan  # 中间区域不改变信号
    )
    
    
    df["Signal"] = df["Raw_Signal"].ffill().fillna(-1)  # 前向填充中间区域
    
    
    print(f"vix:{df['^VIX'].iloc[-1]} \n, year_sma:{df['QQQ_SMA_YEAR'].iloc[-1]} \n, sma_short:{df['QQQ_SMA_SHORT'].iloc[-1]} \n, sma_long:{df['QQQ_SMA_LONG'].iloc[-1]} \n, last qqq close:{df['QQQ'].iloc[-1]} ")
    print(f"GLD:{df['GLD'].iloc[-1]}, GLD_SMA:{df['GLD_SMA'].iloc[-1]}")
    print("check qqq/nasdaq100 forward pe <35 before buying leverage")
    
    print(f"\n\nlast day signal:{df['Signal'].iloc[-1]}, need trade today:{df['Signal'].iloc[-1] != df['Signal'].iloc[-2]}")
    print(f"last day raw signal:{df['Raw_Signal'].iloc[-1]}")
    if(df['Signal'].iloc[-1]==1 and df['Signal'].iloc[-1] != df['Signal'].iloc[-2]):
        print("!!!Need to trade today, Hold 0.8aum of 3x qqq")
    
    else:
        if(df['Signal'].iloc[-1]==-1 and df['Signal'].iloc[-1] != df['Signal'].iloc[-2]):
            if df['GLD'].iloc[-1]>df['GLD_SMA'].iloc[-1] :
                print("!!!!Need to trade today, Hold all aum of GLD")
            else:
                print("!!!!Need to trade today, Hold all aum of SHY")
        elif (df['Signal'].iloc[-1] == df['Signal'].iloc[-2]):
            if df['Signal'].iloc[-1]==1:
                print ("!!!!!Keep, 80% aum 3x qqq")
            if df['Signal'].iloc[-1] == -1:
                print("!!!!!Keep safe asset")
    
    return df