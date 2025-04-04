import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import ttest_1samp
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D 
from tabulate import tabulate  # Add this import
def max_drawdown_fn(df_s, plot=False):
    nav_series = df_s


    cumulative_max = nav_series.cummax()
    
    # 计算每日回撤
    drawdown = (nav_series - cumulative_max) / cumulative_max
    
    # 找到最大回撤的结束日期（最低点）
    max_dd_end_date = drawdown.idxmin()
    
    # 找到对应峰值的日期
    max_dd_start_date = cumulative_max[:max_dd_end_date].idxmax()
    
    # 计算回撤持续天数
    dd_duration = (max_dd_end_date - max_dd_start_date).days
    
    

     # 可视化回撤区间
    if plot:
        plt.figure(figsize=(12,6))
        plt.plot(nav_series, label='Strategy NAV')
        plt.fill_between(nav_series.index, 
                        cumulative_max, 
                        nav_series, 
                        where=(nav_series < cumulative_max),
                        color='red', alpha=0.3)
        plt.axvspan(max_dd_start_date, max_dd_end_date, color='red', alpha=0.2, label='Max Drawdown Period')
        plt.title("Strategy NAV with Maximum Drawdown Highlighted")
        plt.legend()
        plt.show()
    
    return drawdown.min(), max_dd_start_date, max_dd_end_date

def analysis(df, risk_free_rate=0.04, trading_days=252, output_file='strategy_report.txt', plot=False):
    """
    策略绩效分析函数
    
    参数：
    df : DataFrame
        必须包含两列：
        - 'Strategy_Return' : 策略日收益率（小数形式，如0.01表示1%）
        - 'Market_Return' : 基准日收益率
    
    risk_free_rate : float, 可选
        年化无风险利率（默认0.02表示2%）
    
    trading_days : int, 可选
        年化交易日数（默认252）
    
    output_file : str, 可选
        报告输出文件名
    
    返回：
    dict : 包含所有计算指标的字典
    """
    
    # ====================
    # 数据预处理
    # ====================
    data = df[['Strategy_Return', 'Market_Return']].copy().dropna()
    strat_ret = data['Strategy_Return'].dropna()
    mkt_ret = data['Market_Return'].dropna()
    start= data.index[0]
    end =  data.index[-1]
    diff_days = (end-start).days

    years = diff_days/365.25
    trading_days = 252
    rfr_one = risk_free_rate / trading_days
    # ====================
    # 基本指标计算
    # ====================
    # 净值曲线
    nav = (1 + strat_ret).cumprod()
    max_nav = nav.expanding().max()
    drawdown = (nav - max_nav) / max_nav
    
    # 最大回撤
    max_drawdown, mdd_start, mdd_end =  max_drawdown_fn(nav,plot)
    mdd_duration = (drawdown.idxmin() - nav[:drawdown.idxmin()].idxmax()).days
    
    # 年化指标
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (1 + total_return) ** (1/years) - 1
    
    mkt_nav= (1 + mkt_ret).cumprod()
    mkt_nav3 = (1 + 3*mkt_ret).cumprod()
    mkt_all_ret = mkt_nav.iloc[-1] / mkt_nav.iloc[0] - 1
    mkt_cagr = (1 + mkt_all_ret) ** (1/years) - 1
    mkt_daily = (1+ mkt_cagr) ** (1/252) -1
    # ====================
    # 风险调整指标
    # ====================
    # CAPM回归
    X = sm.add_constant(mkt_ret)
    model = sm.OLS(strat_ret - risk_free_rate/trading_days, 
                 X - risk_free_rate/trading_days).fit()
    alpha = model.params[0]
    beta = model.params[1]
    alpha_pvalue = model.pvalues[0]
    alpha_annualized = cagr - (risk_free_rate + beta*(mkt_cagr - risk_free_rate))
    # 夏普比率
    excess_ret = strat_ret - risk_free_rate/trading_days
    sharpe = excess_ret.mean() / excess_ret.std() * np.sqrt(trading_days)
    
    # ====================
    # 统计检验
    # ====================
    # T检验（策略收益是否显著> risk free rate）
    t_stat, ret_pvalue = ttest_1samp(strat_ret, rfr_one, alternative='greater')
    # T检验（策略收益是否显著> market daily return）
    t_stat_mkt, ret_pvalue_mkt = ttest_1samp(strat_ret, mkt_daily, alternative='greater')
    t_stat_mkt2x, ret_pvalue_mkt2x = ttest_1samp(strat_ret, mkt_daily*2, alternative='greater')
    t_stat_mkt2x, ret_pvalue_mkt3x = ttest_1samp(strat_ret, mkt_daily*3, alternative='greater')
    # ====================
    # 结果汇总
    # ====================
    results = {
        'Alpha_daily': alpha,
        'Alpha_annual': alpha_annualized,
        'Beta': beta,
        'CAGR': cagr,
        'Mkt Cagr': mkt_cagr,
        'Sharpe': sharpe,
        'Max_Drawdown': max_drawdown,
        'MDD_START': mdd_start,
        'MDD_END': mdd_end,
        'MDD_Duration': mdd_duration,
        'Alpha_pvalue': alpha_pvalue,
        'Return_pvalue': ret_pvalue,
        'Return_pvalue_mkt': ret_pvalue_mkt,
        'Return_pvalue_mkt2x': ret_pvalue_mkt2x,
        'Return_pvalue_mkt3x': ret_pvalue_mkt3x,
        'Total_Return': total_return,
        'Annualized_Volatility': strat_ret.std() * np.sqrt(trading_days),
        'Calmar_Ratio': cagr / abs(max_drawdown),
        'Win_Rate': (strat_ret > 0).mean(),
        'Start_Date': data.index[0].strftime('%Y-%m-%d'),
        'End_Date': data.index[-1].strftime('%Y-%m-%d'),
        'Trading_Days': len(data)
    }
    
    # ====================
    # 生成报告
    # ====================
    report = f"""
    Strategy Performance Report
    ===========================
    Period: {results['Start_Date']} to {results['End_Date']}
    Trading Days: {results['Trading_Days']} ({years:.2f} years)
    
    Return Statistics
    -----------------
    Total Return: {results['Total_Return']:.2%}
    CAGR: {results['CAGR']:.2%}
    Annualized Volatility: {results['Annualized_Volatility']:.2%}
    Win Rate: {results['Win_Rate']:.1%}
    
    Risk Metrics
    ------------
    Max Drawdown: {results['Max_Drawdown']:.2%}
    MDD Duration: {results['MDD_Duration']} days, FROM {results['MDD_START']}->{results['MDD_END']}
    Sharpe Ratio: {results['Sharpe']:.2f}
    Calmar Ratio: {results['Calmar_Ratio']:.2f}
    
    CAPM Analysis
    -------------
    Alpha_daily: {results['Alpha_daily']:.4f} (p-value: {results['Alpha_pvalue']:.4f})
    Alpha annulize: {results['Alpha_annual']:.4f})
    Beta: {results['Beta']:.2f}
    
    Significance Tests
    ------------------
    Strategy Return > risk_free: {'Rejected' if results['Return_pvalue'] > 0.05 else 'Confirmed'}
    (p-value: {results['Return_pvalue']:.4f})
    Strategy Return > mkt_daily return: {'Rejected' if results['Return_pvalue_mkt'] > 0.05 else 'Confirmed'}
    (p-value: {results['Return_pvalue_mkt']:.4f})
    Strategy Return > mkt_daily 2x return: {'Rejected' if results['Return_pvalue_mkt2x'] > 0.05 else 'Confirmed'}
    (p-value: {results['Return_pvalue_mkt2x']:.4f})
    Strategy Return > mkt_daily 3x return: {'Rejected' if results['Return_pvalue_mkt3x'] > 0.05 else 'Confirmed'}
    (p-value: {results['Return_pvalue_mkt3x']:.4f})
    """
    
    if plot:
        plt.figure(figsize=(12,6))
        plt.plot(nav, label="Strategy")
        plt.plot(mkt_nav, label="mkt", linestyle="--", alpha=0.7)
        plt.plot(mkt_nav3, label="mkt3x", linestyle="--", alpha=0.7)
        plt.legend()
        plt.show()

    # 打印并保存报告
    print(report)
    with open(output_file, 'w') as f:
        f.write(report)
    
    return results

def plot_strategy_with_signals(df):
    plt.figure(figsize=(14, 8))
    
    # 1. Plot NAV Curve
    ax1 = plt.subplot(211)
    ax1.plot(df.index, df["Strategy_NAV"], label='Strategy NAV', color='navy', linewidth=2)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Strategy Performance with Trading Signals')
    
    # 2. Highlight Holding Periods
    leverage_periods = df[df["Signal"] == 1]
    safe_periods = df[df["Signal"] == -1]
    
    ax1.fill_between(df.index, df["Strategy_NAV"], 
                    where=(df["Signal"] == 1), 
                    color='lightgreen', alpha=0.3, label='Leveraged Exposure')
    ax1.fill_between(df.index, df["Strategy_NAV"], 
                    where=(df["Signal"] == -1), 
                    color='lightcoral', alpha=0.3, label='Safe Asset')
    
    # 3. Mark Signal Changes
    signal_changes = df[df["Signal_Change"] == 2]
    for date in signal_changes.index:
        ax1.axvline(x=date, color='black', linestyle='--', alpha=0.5, linewidth=0.7)
        nav = df.loc[date, "Strategy_NAV"]
        ax1.plot(date, nav, 'o', markersize=6, 
                markerfacecolor='red' if df.loc[date, "Signal"] == -1 else 'green',
                markeredgecolor='black')
    
    # 4. Formatting
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 5. Create Legend
    legend_elements = [
        Patch(facecolor='lightgreen', alpha=0.3, label='Leveraged (TQQQ/QLD)'),
        Patch(facecolor='lightcoral', alpha=0.3, label=f'Safe Asset '),
        Line2D([0], [0], color='black', linestyle='--', label='Signal Change'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
              markersize=8, label='Switch to Leverage'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
              markersize=8, label='Switch to Safe')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    # 6. Add Secondary Axis for VIX
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["^VIX"], color='purple', alpha=0.4, linewidth=1.5)
    ax2.set_ylabel('VIX', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.grid(None)
    
    # 7. Add Holdings Table
    holding_days = pd.DataFrame({
        'Position': ['Leveraged', 'Safe Asset'],
        'Days': [len(leverage_periods), len(safe_periods)],
        '% Time': [
            f"{len(leverage_periods)/len(df)*100:.1f}%", 
            f"{len(safe_periods)/len(df)*100:.1f}%"
        ]
    }).set_index('Position')
    
    plt.subplot(212)
    plt.table(cellText=holding_days.values,
             rowLabels=holding_days.index,
             colLabels=holding_days.columns,
             loc='center',
             cellLoc='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def print_holding_periods(df):
    # Identify regime changes
    start= df.index[0]
    end =  df.index[-1]
    diff_days = (end-start).days

    years = diff_days/365.25
    annual_turnover = df["Trade_Day"].sum() / years
    df['Regime_Change'] = df['Signal'].diff().ne(0)
    df['Regime_Group'] = df['Regime_Change'].cumsum()

    # Create table of periods
    periods = []
    for group in df['Regime_Group'].unique():
        period = df[df['Regime_Group'] == group].iloc[0]
        start = df[df['Regime_Group'] == group].index[0]
        end = df[df['Regime_Group'] == group].index[-1]
        duration = (end - start).days + 1  # Inclusive
        
        periods.append({
            'Position': 'Leverage' if period['Signal'] == 1 else 'Safe',
            'Start Date': start.strftime('%Y-%m-%d'),
            'End Date': end.strftime('%Y-%m-%d'),
            'Duration (Days)': duration,
            'VIX Start': round(period['^VIX'], 1),
            'VIX End': round(df.loc[end, '^VIX'], 1)
        })

    # Convert to DataFrame
    period_df = pd.DataFrame(periods)
    
    # Print formatted table
    print(f"annual turnover: {annual_turnover:.1f} ")
    print("\nHolding Period Details:")
    print(tabulate(period_df, headers='keys', tablefmt='psql', showindex=False))
    
   
    
   