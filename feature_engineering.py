import pandas as pd
import numpy as np

df = pd.read_csv('us_macro_features_extended.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.replace('<1', 0)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=['Trend_layoffs', 'Trend_unemployment_benefits', 'Trend_jobs_near_me'])

new_features = {}
macro_cols = ['Unemployment_Rate', 'Initial_Jobless_Claims', 'Nonfarm_Payrolls', 'Fed_Funds_Rate', 
              'Yield_Curve_Spread', 'CPI_Headline', 'Consumer_Sentiment', 'Retail_Sales', 
              'Industrial_Production', 'Housing_Starts', 'VIX', 'Credit_Spread']
trend_cols = ['Trend_layoffs', 'Trend_unemployment_benefits', 'Trend_jobs_near_me']

for col in macro_cols + trend_cols:
    new_features[f'{col}_lag1'] = df[col].shift(1)
    new_features[f'{col}_lag3'] = df[col].shift(3)
    new_features[f'{col}_lag6'] = df[col].shift(6)

for col in macro_cols:
    new_features[f'{col}_3m_avg'] = df[col].rolling(window=3).mean()
    new_features[f'{col}_6m_avg'] = df[col].rolling(window=6).mean()
    new_features[f'{col}_MoM_diff'] = df[col].diff(1)   
    new_features[f'{col}_YoY_diff'] = df[col].diff(12)  
    
pct_change_cols = ['Nonfarm_Payrolls', 'CPI_Headline', 'Retail_Sales', 'Industrial_Production', 'Initial_Jobless_Claims']
for col in pct_change_cols:
    new_features[f'{col}_MoM_pct'] = df[col].pct_change(1) * 100
    new_features[f'{col}_YoY_pct'] = df[col].pct_change(12) * 100

for col in trend_cols:
    new_features[f'{col}_MoM_diff'] = df[col].diff(1)
    new_features[f'{col}_3m_diff'] = df[col].diff(3)

new_features['Yield_Curve_Inverted'] = (df['Yield_Curve_Spread'] < 0).astype(int)
new_features['Target_Unemployment_1m_ahead'] = df['Unemployment_Rate'].shift(-1)

df_eng = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
df_eng.dropna(inplace=True)

CORR_THRESHOLD = 0.85
target_name = 'Target_Unemployment_1m_ahead'

corr_matrix = df_eng.drop(columns=[target_name]).corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
target_corr = df_eng.corr()[target_name].abs()

to_drop = set()
for col in upper_tri.columns:
    correlated_cols = upper_tri.index[upper_tri[col] > CORR_THRESHOLD].tolist()
    for corr_col in correlated_cols:
        if target_corr[col] > target_corr[corr_col]:
            to_drop.add(corr_col)
        else:
            to_drop.add(col)

df_eng.drop(columns=list(to_drop), inplace=True)

df_eng.to_csv('us_macro_engineered.csv')