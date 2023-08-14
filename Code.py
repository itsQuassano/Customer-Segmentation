import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load your dataset
data = pd.read_excel(r"C:\Users\Utkarsh Pandey\Desktop\Data.xlsx")

# Select relevant features for segmentation
features = ['Age', 'Income', 'SpendingScore']

# Preprocess the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[features])

# Determine the optimal number of clusters using the elbow method
inertia_values = []
for num_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia_values, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Choose the optimal number of clusters based on the elbow point
optimal_k = 3  # You may choose the appropriate number of clusters

# Apply K-means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Analyze the clusters
cluster_means = data.groupby('Cluster')[features].mean()
print(cluster_means)

# Visualize the clusters
plt.figure(figsize=(12, 8))
for cluster in range(optimal_k):
    plt.scatter(data[data['Cluster'] == cluster]['Age'],
                data[data['Cluster'] == cluster]['SpendingScore'],
                label=f'Cluster {cluster}')
plt.title('Customer Segmentation Analysis')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

# Evaluate the quality of clustering using silhouette score
silhouette_avg = silhouette_score(scaled_data, data['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')
