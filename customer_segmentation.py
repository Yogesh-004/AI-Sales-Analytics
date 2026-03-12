import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\sql_dashboard\data\cleaned_superstore.csv")

print(df.head())
customer_data = df.groupby("customer_name").agg({
    "sales": "sum",
    "profit": "sum",
    "quantity": "sum"
}).reset_index()

print(customer_data.head())
X = customer_data[["sales", "profit", "quantity"]]

kmeans = KMeans(n_clusters=3, random_state=42)

customer_data["cluster"] = kmeans.fit_predict(X)

print(customer_data.head())

plt.scatter(customer_data["sales"], customer_data["profit"], 
            c=customer_data["cluster"])

plt.xlabel("Sales")
plt.ylabel("Profit")
plt.title("Customer Segmentation")

plt.show()
customer_data.to_csv("customer_segments.csv", index=False)