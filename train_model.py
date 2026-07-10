import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv(r"C:\Users\Asus\OneDrive\Documents\project_c.csv")

print("\nDataset Preview")
print(df.head())

print("\nDataset Shape :", df.shape)

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows :", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing values
df["Annual_spending"] = df["Annual_spending"].fillna(df["Annual_spending"].mean())
df["Orders_count"] = df["Orders_count"].fillna(df["Orders_count"].mean())

# ==========================
# Feature Selection
# ==========================
X = df[['Annual_spending', 'Orders_count']]

# ==========================
# Standard Scaling
# ==========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# Elbow Method
# ==========================
wcss = []

for k in range(1, 6):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.figure(figsize=(7,5))
plt.plot(range(1,6), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

print("\nChoose K = 3")

# ==========================
# Final Model
# ==========================
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# ==========================
# Silhouette Score
# ==========================
score = silhouette_score(X_scaled, clusters)

print("\nSilhouette Score :", round(score,3))

# ==========================
# Cluster Summary
# ==========================
summary = df.groupby("Cluster")[["Annual_spending","Orders_count"]].mean()

print("\nCluster Summary")
print(summary)

# ==========================
# Customer Type Mapping
# ==========================
sorted_clusters = summary["Annual_spending"].sort_values()

customer_type = {}

customer_type[sorted_clusters.index[0]] = "Low Value Customer"
customer_type[sorted_clusters.index[1]] = "Medium Value Customer"
customer_type[sorted_clusters.index[2]] = "High Value Customer"

df["Customer_Type"] = df["Cluster"].map(customer_type)

print("\nCustomer Type Mapping")
print(customer_type)

# ==========================
# Scatter Plot
# ==========================
plt.figure(figsize=(8,6))

plt.scatter(
    df["Annual_spending"],
    df["Orders_count"],
    c=df["Cluster"],
    cmap="viridis",
    s=80
)

plt.xlabel("Annual Spending")
plt.ylabel("Orders Count")
plt.title("Customer Segmentation")

plt.grid(True)

plt.show()

# ==========================
# Save Model
# ==========================
joblib.dump(kmeans, "flipkart_kmeans.pkl")
joblib.dump(scaler, "flipkart_scaler.pkl")

# Save Cluster Mapping
joblib.dump(customer_type, "customer_type_mapping.pkl")

# Save Clustered Dataset
df.to_csv("clustered_customers.csv", index=False)

print("\nModel Saved Successfully")
print("Files Created:")
print("✔ flipkart_kmeans.pkl")
print("✔ flipkart_scaler.pkl")
print("✔ customer_type_mapping.pkl")
print("✔ clustered_customers.csv")