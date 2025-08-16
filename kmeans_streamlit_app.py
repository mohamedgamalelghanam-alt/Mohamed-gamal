import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Title
# ----------------------------
st.title("Wholesale Customers Segmentation (KMeans Clustering)")

# ----------------------------
# Load dataset
# ----------------------------
data = pd.read_csv("Wholesale customers data.csv")

st.subheader("Dataset Preview")
st.write(data.head())

# ----------------------------
# Preprocessing
# ----------------------------
# Scale data for better clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# ----------------------------
# Sidebar - number of clusters
# ----------------------------
st.sidebar.subheader("Clustering Options")
k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 3)

# ----------------------------
# Apply KMeans
# ----------------------------
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
data["Cluster"] = clusters

st.subheader("Clustered Data")
st.write(data.head())

# ----------------------------
# Visualization
# ----------------------------
st.subheader("Clusters Visualization")

fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clusters, cmap="viridis")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.title("KMeans Clustering Visualization")
plt.colorbar(scatter)
st.pyplot(fig)
