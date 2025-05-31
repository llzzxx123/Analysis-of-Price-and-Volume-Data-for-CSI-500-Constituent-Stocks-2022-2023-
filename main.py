import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# 读取数据
desktop_path=os.path.join(os.path.expanduser('~'),'Desktop')
file_path=os.path.join(desktop_path,'中证500成分股2022-2023量价数据.xlsx')
df = pd.read_excel(file_path)
df['Date'] = pd.to_datetime(df['Date'])

processed_dfs = []

# 按股票代码处理
for code, group in df.groupby('Code'):
    group = group.sort_values('Date').reset_index(drop=True)
    group['daily_return'] = group['Close'].pct_change()
    mask = (abs(group['daily_return']) > 0.2) | (
        (group['High'] == group['Low']) & 
        (group['Low'] == group['Open']) & 
        (group['Open'] == group['Close'])
    )
    cleaned_group = group[~mask].reset_index(drop=True)
    cleaned_group['daily_return'] = cleaned_group['Close'].pct_change()
    
    cleaned_group.set_index('Date', inplace=True)
    
    weekly = cleaned_group['Close'].resample('W').last().pct_change().rename('weekly_return')
    monthly = cleaned_group['Close'].resample('M').last().pct_change().rename('monthly_return')
    
    cleaned_group = cleaned_group.join([weekly, monthly], how='left')
    cleaned_group.reset_index(inplace=True)
    
    cleaned_group['cumulative_return'] = (1 + cleaned_group['daily_return'].fillna(0)).cumprod() - 1
    
    processed_dfs.append(cleaned_group)

merged_df = pd.concat(processed_dfs, ignore_index=True)

# 计算指标：夏普比率、最大回撤、年化收益率
metrics = []
for code, group in merged_df.groupby('Code'):
    if len(group) < 2:
        continue
    
    cum_return = group['cumulative_return'].iloc[-1]
    start_date = group['Date'].iloc[0]
    end_date = group['Date'].iloc[-1]
    days = (end_date - start_date).days
    if days <= 0:
        annualized_return = 0.0
    else:
        annualized_return = (1 + cum_return) ** (252 / days) - 1
    
    daily_returns = group['daily_return'].dropna()
    if len(daily_returns) < 2:
        sharpe_ratio = np.nan
    else:
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan
    
    cumulative_returns = group['cumulative_return'] + 1  # 转换为净值
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (peak - cumulative_returns) / peak
    max_drawdown = drawdown.max()
    
    metrics.append({
        'Code': code,
        'Sharpe': sharpe_ratio,
        'MaxDrawdown': max_drawdown,
        'AnnualizedReturn': annualized_return
    })

metrics_df = pd.DataFrame(metrics)

# 筛选夏普比率前10的股票
top_10 = metrics_df.nlargest(10, 'Sharpe')
top_codes = top_10['Code'].tolist()

# 绘制净值曲线
plt.figure(figsize=(12, 6))
for code in top_codes:
    stock_data = merged_df[merged_df['Code'] == code]
    plt.plot(stock_data['Date'], stock_data['cumulative_return'], label=f'Code {code}')

plt.title('Cumulative Returns of Top 10 Stocks by Sharpe Ratio')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()
