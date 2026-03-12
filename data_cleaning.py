import pandas as pd

df = pd.read_csv("D:\\sql_dashboard\\data\\SuperStoreOrders.csv")

df.head()
df.info()

df.isnull().sum()
df.duplicated().sum()
df['order_date'] = pd.to_datetime(df['order_date'], format='mixed', dayfirst=True)
df['ship_date'] = pd.to_datetime(df['ship_date'], format='mixed', dayfirst=True)
df[['order_date', 'ship_date']].head()
df.dtypes
df['Year'] = df['order_date'].dt.year
df['Month'] = df['order_date'].dt.month
df['Quarter'] = df['order_date'].dt.quarter
print(df['Year'].min(), df['Year'].max())

print(df['sales'].head(10))

df['sales'] = (
    df['sales']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.replace('$', '', regex=False)
)

df['sales'] = pd.to_numeric(df['sales'], errors='coerce')

df['profit'] = (
    df['profit']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.replace('$', '', regex=False)
)

df['profit'] = pd.to_numeric(df['profit'], errors='coerce')
df['is_profitable'] = df['profit'].apply(lambda x: 1 if x > 0 else 0)
print("Total Revenue:", df['sales'].sum())
print("Total Profit:", df['profit'].sum())
print("Profit Margin %:", (df['profit'].sum()/df['sales'].sum())*100)
df.groupby('region')[['sales','profit']].sum().sort_values(by='sales', ascending=False)

print(df.groupby('region')[['sales', 'profit']].sum().sort_values(by='sales', ascending=False))

region_summary = df.groupby('region')[['sales', 'profit']].sum()
region_summary['profit_margin_%'] = (region_summary['profit'] / region_summary['sales']) * 100

print(region_summary.sort_values(by='profit_margin_%', ascending=False))
print(df[['discount', 'profit']].corr())
df.groupby('discount')[['sales','profit']].mean().sort_index().head(10)
print(df.groupby('discount')[['sales', 'profit']].mean().sort_index().head(10))
region_discount = df.groupby('region')[['discount','profit']].mean()
print(region_discount.sort_values(by='discount', ascending=False))

segment_summary = df.groupby('segment')[['sales','profit','discount']].mean()
print(segment_summary)

segment_total = df.groupby('segment')[['sales','profit']].sum()
segment_total['profit_margin_%'] = (segment_total['profit']/segment_total['sales'])*100
print(segment_total)

category_summary = df.groupby('category')[['sales','profit']].sum()
category_summary['profit_margin_%'] = (category_summary['profit']/category_summary['sales'])*100
print(category_summary.sort_values(by='profit_margin_%', ascending=False))

print(df.groupby('category')[['discount','profit']].mean())

df.to_csv("data/cleaned_superstore.csv", index=False)
print(df.columns)

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Drop duplicate 'year'
df = df.drop(columns=['year'])

# Rename capital columns to lowercase (clean practice)
df = df.rename(columns={
    'Year': 'year',
    'Month': 'month',
    'Quarter': 'quarter'
})

print(df.columns)
df.to_csv(r"D:\sql_dashboard\data\cleaned_superstore_v2.csv", index=False)