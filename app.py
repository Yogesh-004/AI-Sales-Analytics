import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from prophet import Prophet

st.set_page_config(page_title="AI Sales Intelligence", layout="wide")

st.title("🚀 AI Sales Intelligence Platform")

# DATABASE CONNECTION
engine = create_engine(
    "mysql+pymysql://root:R1o2o3t4.com@localhost:3306/sales_analytics"
)

# SIDEBAR
st.sidebar.title("Control Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload Sales Dataset",
    type=["csv"]
)

# LOAD DATA
if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.success("Dataset Uploaded Successfully")

    data.to_sql(
        "sales_data",
        engine,
        if_exists="replace",
        index=False
    )

else:

    data = pd.read_sql("SELECT * FROM sales_data", engine)

# KPI SECTION
st.subheader("📊 Business KPIs")

total_sales = data["sales"].sum()
total_profit = data["profit"].sum()
total_orders = len(data)
total_customers = data["customer_name"].nunique()

c1,c2,c3,c4 = st.columns(4)

c1.metric("Total Sales", f"${total_sales:,.0f}")
c2.metric("Total Profit", f"${total_profit:,.0f}")
c3.metric("Total Orders", total_orders)
c4.metric("Customers", total_customers)

# SALES TREND
st.subheader("📈 Sales Trend")

sales_trend = data.groupby("month")["sales"].sum().reset_index()

fig1 = px.line(
    sales_trend,
    x="month",
    y="sales",
    markers=True
)

st.plotly_chart(fig1,use_container_width=True)

# CATEGORY ANALYSIS
st.subheader("📦 Category Performance")

cat_sales = data.groupby("category")["sales"].sum().reset_index()

fig2 = px.bar(
    cat_sales,
    x="category",
    y="sales",
    color="category"
)

st.plotly_chart(fig2,use_container_width=True)

# REGION ANALYSIS
st.subheader("🌎 Regional Sales")

region_sales = data.groupby("region")["sales"].sum().reset_index()

fig3 = px.pie(
    region_sales,
    names="region",
    values="sales"
)

st.plotly_chart(fig3,use_container_width=True)

# -----------------------------
# CUSTOMER SEGMENTATION (ML)
# -----------------------------

st.subheader("🧠 Customer Segmentation")

customer_data = data.groupby("customer_name").agg({
    "sales":"sum",
    "profit":"sum",
    "quantity":"sum"
}).reset_index()

X = customer_data[["sales","profit","quantity"]]

kmeans = KMeans(n_clusters=4, random_state=42)

customer_data["segment"] = kmeans.fit_predict(X)

fig4 = px.scatter(
    customer_data,
    x="sales",
    y="profit",
    color="segment",
    hover_data=["customer_name"]
)

st.plotly_chart(fig4,use_container_width=True)

# SAVE SEGMENTS
customer_data.to_sql(
    "customer_segmentation",
    engine,
    if_exists="replace",
    index=False
)

# -----------------------------
# SALES FORECAST
# -----------------------------

st.subheader("📊 Sales Forecast")

data["order_date"] = pd.to_datetime(data["order_date"])

forecast_data = data.groupby("order_date")["sales"].sum().reset_index()

forecast_data.columns = ["ds","y"]

model = Prophet()

model.fit(forecast_data)

future = model.make_future_dataframe(periods=30)

forecast = model.predict(future)

fig5 = px.line(
    forecast,
    x="ds",
    y="yhat"
)

st.plotly_chart(fig5,use_container_width=True)

# SAVE FORECAST
forecast[["ds","yhat","yhat_lower","yhat_upper"]].to_sql(
    "sales_forecast",
    engine,
    if_exists="replace",
    index=False
)

st.success("AI Analytics System Running Successfully 🚀")