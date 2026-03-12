import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"D:\sql_dashboard\data\cleaned_superstore.csv")

print(df.head())
features = ["sales", "quantity", "discount", "shipping_cost"]

X = df[features]

y = df["is_profitable"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
importance = pd.Series(model.feature_importances_, index=features)

print(importance)
joblib.dump(model, "profit_prediction_model.pkl")