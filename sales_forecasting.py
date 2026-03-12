import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import pymysql
import sqlalchemy
import cryptography

# Load cleaned dataset
df = pd.read_csv(r"D:\sql_dashboard\data\cleaned_superstore_v2.csv")

# Convert date column
df["order_date"] = pd.to_datetime(df["order_date"])

# Aggregate daily sales
sales = df.groupby("order_date")["sales"].sum().reset_index()

# Rename columns for Prophet
sales.columns = ["ds", "y"]

print(sales.head())

model = Prophet()

model.fit(sales)
future = model.make_future_dataframe(periods=180)

forecast = model.predict(future)

print(forecast.head())
fig = model.plot(forecast)

plt.title("Sales Forecast for Next 6 Months")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.show()

model.plot_components(forecast)
plt.show()

forecast[["ds","yhat","yhat_lower","yhat_upper"]].to_csv(
    "sales_forecast_results.csv",
    index=False
)

# Keep important columns
forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


# Connect to MySQL
engine = sqlalchemy.create_engine(
    "mysql+pymysql://root:R1o2o3t4.com@127.0.0.1:3306/sales_analytics"
)

# Save forecast to database
forecast_data .to_sql(
    name="sales_forecast",
    con=engine,
    if_exists="replace",
    index=False
)

print("Sales forecast table created")