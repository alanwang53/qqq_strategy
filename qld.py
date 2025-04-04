import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from dateutil.relativedelta import relativedelta
# ====================
# 用戶自定義參數區
# ====================
#START_DATE = "2010-01-01"     # 回測起始日期
START_DATE = "1999-06-01"     # QQQ 上市日
END_DATE = "2025-03-22"       # 回測結束日期
INITIAL_CAPITAL = 10000       # 初始資金
COMMISSION = 0.003            # 單次交易佣金率（0.01%）+ slippage 
SMA_PERIOD = 149           # 均線週期（默認 200 日）
TARGET_LEVERAGE = 3        # 目標槓桿倍數（2~3之間）
QLD_SYMBOL = "QLD"            # 2x 槓桿 ETF 代碼
TQQQ_SYMBOL = "TQQQ"          # 3x 槓桿 ETF 代碼
SAFE_ASSET = "GLD"            # 避險資產代碼（可換為 "SHY"（短債）、"GLD"（黃金）、"CASH"（現金）等
DAILY_LEVERAGE_3 = 3            # 模擬槓桿倍數
DAILY_LEVERAGE_2 = 2            # 模擬槓桿倍數
MANAGEMENT_FEE = 0.0075       # 年度管理費（TQQQ 實際年費 0.95%）- dividend 0.2%
ORI_DISCOUNT =  0.8           # pessimistic estimation of total return
# ====================

def download_data():
    """Download and simulate TQQQ / QLD / Safe Asset data."""
    # Download QQQ price
    qqq_price = yf.download("QQQ", start=START_DATE, end=END_DATE, auto_adjust=False)["Adj Close"]
    qqq_price.name = "QQQ"  # Assign a unique name

    if SAFE_ASSET == "CASH":
        safe_price = pd.Series(1, index=qqq_price.index, name="CASH")
    else:
        safe_price = yf.download(SAFE_ASSET, start=START_DATE, end=END_DATE, auto_adjust=False)["Adj Close"]
        safe_price.name = SAFE_ASSET
    df = pd.concat([qqq_price, safe_price], axis=1).dropna()
    # Calculate daily returns
    df["qqq_returns"] = df["QQQ"].pct_change().dropna()

    # Simulate leveraged ETF returns
    daily_fee = (1 + MANAGEMENT_FEE) ** (1 / 252) - 1
    df["tqqq_returns"] = df["qqq_returns"] * DAILY_LEVERAGE_3 - daily_fee
    df["qld_returns"] = df["qqq_returns"] * DAILY_LEVERAGE_2 - daily_fee

    # Simulate price series
    df[TQQQ_SYMBOL] = (1 +  df["tqqq_returns"] ).cumprod() * qqq_price.iloc[0]
    df[QLD_SYMBOL] = (1 +   df["qld_returns"] ).cumprod() * qqq_price.iloc[0]

    # Assign unique names to each Series
    #tqqq_price.name = TQQQ_SYMBOL
    #qld_price.name = QLD_SYMBOL

    #print(f"TQQQ Series Name: {tqqq_price.name}")
    #print(f"QLD Series Name: {qld_price.name}")
    # Safe asset
    

    # Concatenate into a single DataFrame
    #df = pd.concat([qqq_price, tqqq_price, qld_price, safe_price], axis=1).dropna()
    
    # Print column names to verify uniqueness
    print("Downloaded data with columns:")
    print(df.columns)

    # Ensure column names are unique
    assert df.columns.is_unique, "Duplicate column names detected!"

    return df

def calculate_leverage_weights(target_leverage):
    """根據目標槓桿計算 QLD 和 TQQQ 的權重（線性組合）"""
    # 解方程：2*w + 3*(1-w) = target_leverage → w = 3 - target_leverage
    w_qld = 3 - target_leverage
    w_tqqq = target_leverage - 2
    return w_qld, w_tqqq

def backtest_strategy(df):
    # 驗證目標槓桿有效性
    if not (2 <= TARGET_LEVERAGE <= 3):
        raise ValueError("槓桿倍數必須在 2~3 之間！")
    
    # 計算槓桿組合權重
    w_qld, w_tqqq = calculate_leverage_weights(TARGET_LEVERAGE)
    
    # 計算 QQQ 的 SMA200
    df["QQQ_SMA"] = df["QQQ"].rolling(SMA_PERIOD).mean()
    
    # 生成交易信號（1=槓桿組合, -1=避險資產）
    #df["Signal"] = np.where(df["QQQ"] > df["QQQ_SMA"], 1, -1)
    # 生成信号（修复逻辑嵌套）
    df["Raw_Signal"] = np.select(
        condlist=[
            df["QQQ"] > 1.005 * df["QQQ_SMA"],  # 条件1：高于1%阈值
            df["QQQ"] < 0.995 * df["QQQ_SMA"]   # 条件2：低于1%阈值
        ],
        choicelist=[
            1,   # 满足条件1时信号为1（杠杆）
            -1   # 满足条件2时信号为-1（避险）
        ],
        default=np.nan  # 中间区域不改变信号
    )
    df["Signal"] = df["Raw_Signal"].ffill().fillna(-1)  # 前向填充中间区域
    # 避免在初始 SMA 計算期前交易
    df = df.iloc[SMA_PERIOD:]
    
    # 計算各資產收益率
    df[f"{QLD_SYMBOL}_Return"] =  df["QQQ"].pct_change()* DAILY_LEVERAGE_2  #df[QLD_SYMBOL].pct_change()
    df[f"{TQQQ_SYMBOL}_Return"] = df["QQQ"].pct_change()* DAILY_LEVERAGE_3
    df[f"{SAFE_ASSET}_Return"] = df[SAFE_ASSET].pct_change().fillna(0)  # 避險資產
    
    # 動態組合收益率（加權槓桿或避險）
    df["Leveraged_Return"] = w_qld * df[f"{QLD_SYMBOL}_Return"] + w_tqqq * df[f"{TQQQ_SYMBOL}_Return"]
    df["Strategy_Return"] = np.where(df["Signal"].shift(1) == 1, df["Leveraged_Return"], df[f"{SAFE_ASSET}_Return"])
    
    # 交易成本（僅在切換時收取）
    df["Trade"] = df["Signal"].diff().fillna(0).abs()
    df["Strategy_Return"] -= df["Trade"] * COMMISSION
    
    # 計算淨值曲線
    df["Strategy_NAV"] = (1 + df["Strategy_Return"]).cumprod() * INITIAL_CAPITAL
    df["BuyHold_QQQ"] = (1 + df["QQQ"].pct_change()).cumprod() * INITIAL_CAPITAL
    df["BuyHold_QLD"] = (1 + 2*df["QQQ"].pct_change()).cumprod() * INITIAL_CAPITAL
    df["BuyHold_TQQQ"] = (1 + 3*df["QQQ"].pct_change()).cumprod() * INITIAL_CAPITAL
    df["BuyHold_Safe"] = (1 + df[f"{SAFE_ASSET}_Return"]).cumprod() * INITIAL_CAPITAL
    
    return df

def calculate_sharpe_ratio(df, risk_free_rate=0.01):
    # Calculate excess returns
    excess_returns = df["Strategy_Return"] - risk_free_rate / 252
    # Annualize the Sharpe ratio
    annual_sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    return annual_sharpe_ratio

def calculate_jensens_alpha(df, years, strategy_cagr, market_cagr ,risk_free_rate=0.01):
    # Excess returns
    # 新增 Alpha 计算部分
    # 获取策略和市场的日收益率
    strategy_returns = df["Strategy_Return"].dropna()
    market_returns = df["QQQ"].pct_change().dropna()
    
    # 对齐数据索引
    common_idx = strategy_returns.index.intersection(market_returns.index)
    strategy_returns = strategy_returns.loc[common_idx]
    market_returns = market_returns.loc[common_idx]
    
    # 计算 Beta
    covariance = np.cov(strategy_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    beta = covariance / market_variance
    
    

    
    # 计算 Alpha
    alpha = strategy_cagr - (risk_free_rate + beta*(market_cagr - risk_free_rate))
    return alpha
def max_drawdown_fn(df_s):
    nav_series = df_s
    peak = nav_series.expanding().max()
    drawdown = (nav_series - peak)/peak
    return drawdown.min()
def analyze_results(df):
    final_nav = df["Strategy_NAV"].iloc[-1]* ORI_DISCOUNT
    bh_qqq = df["BuyHold_QQQ"].iloc[-1]
    bh_safe = df["BuyHold_Safe"].iloc[-1]
    
    
    date1 = datetime.strptime(START_DATE, "%Y-%m-%d")
    date2 = datetime.strptime(END_DATE, "%Y-%m-%d")


    difference_in_days = (date2 - date1).days


    years = difference_in_days / 365.2425
   

    strategy_cagr = (final_nav/INITIAL_CAPITAL)**(1/years) - 1
    
    
    
    max_drawdown = max_drawdown_fn(df["Strategy_NAV"])
    max_drawdown_qqq = max_drawdown_fn(df["BuyHold_QQQ"])
    # 计算市场年化收益率
    market_cagr = (df["BuyHold_QQQ"].iloc[-1]/INITIAL_CAPITAL)**(1/years) - 1
    alpha = calculate_jensens_alpha( df, years, strategy_cagr, market_cagr,0.02)
    annual_turnover = df["Trade"].sum() / years
    
    print(f"\n策略參數：{TARGET_LEVERAGE}倍槓桿組合 + 避險資產 [{SAFE_ASSET}]")
    print("="*50)
    print(f"策略最終淨值: ${final_nav:,.2f}")
    print(f"年化换手率: {annual_turnover:.1f} 次")
    print(f"strategy年化收益率: {strategy_cagr*100:.2f}%")
    print(f"market(qqq)年化收益率: {market_cagr*100:.2f}%")
    print(f"strategy最大回撤: {max_drawdown*100:.2f}%")
    print(f"qqq max draw down {max_drawdown_qqq*100:.2f}%")
    print(f"買入持有 QQQ 淨值: ${bh_qqq:,.2f}")
    print(f"買入持有 {SAFE_ASSET} 淨值: ${bh_safe:,.2f}")
    print(f"Annualized Jensen's Alpha: {alpha:.2f}")
    plt.figure(figsize=(12,6))
    plt.plot(df["Strategy_NAV"], label=f"策略 ({TARGET_LEVERAGE}x Leverage + {SAFE_ASSET})")
    plt.plot(df["BuyHold_QQQ"], label="Buy & Hold QQQ", linestyle="--", alpha=0.7)
    plt.plot(df["BuyHold_Safe"], label=f"Buy & Hold {SAFE_ASSET}", linestyle="--", alpha=0.7)
    plt.plot(df["BuyHold_TQQQ"], label="Buy & Hold TQQQ", linestyle="--", alpha=0.7)
    plt.plot(df["BuyHold_QLD"], label="Buy & Hold QLD", linestyle="--", alpha=0.7)
    plt.title(f"動態槓桿策略 vs 買入持有 (避險資產: {SAFE_ASSET})")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    data = download_data()
    results = backtest_strategy(data)
    analyze_results(results)