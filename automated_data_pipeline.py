import pandas as pd
import pymysql
import sqlalchemy
import cryptography
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv(r"D:\sql_dashboard\data\cleaned_superstore.csv")

# Create customer segmentation dataset
customer_data = df.groupby("customer_name").agg({
    "sales": "sum",
    "profit": "sum",
    "quantity": "sum"
}).reset_index()

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)

customer_data["segment"] = kmeans.fit_predict(
    customer_data[["sales", "profit", "quantity"]]
)


# Rename segments for business meaning
segment_map = {
    0: "Low Value",
    1: "Medium Value",
    2: "High Value"
}

customer_data["segment"] = customer_data["segment"].map(segment_map)

# Connect to MySQL
engine = sqlalchemy.create_engine(
    "mysql+pymysql://root:R1o2o3t4.com@localhost/sales_analytics"
)

# Export to database
customer_data.to_sql(
    "customer_segments",
    engine,
    if_exists="replace",
    index=False
)

print("Customer segmentation data exported successfully")