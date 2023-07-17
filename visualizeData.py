import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap.umap_ as umap

# Read the CSV file
df = pd.read_csv("yuval_and_matan_features.csv")

# Replace all NaN values with 0
df.fillna(0, inplace=True)

# Extract the numeric features
df = df.select_dtypes(include='number')

# Find columns with only zeros
zero_columns = df.columns[(df == 0).all()]

# Remove columns with only zeros from the data
data_filtered = df.drop(zero_columns, axis=1)

# Normalize or standardize the numeric features
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data_filtered)

# Define the range of k values to try
k_values = range(2, 10)

# Initialize list to store the within-cluster sum of squares (WCSS) for each k
wcss = []
# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions
reduced_data = pca.fit_transform(normalized_data)
# Iterate over each k value
for k in k_values:
    # Run k-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(reduced_data)

    # Append the WCSS to the list
    wcss.append(kmeans.inertia_)

# Plot the WCSS values for each k and save as image
plt.plot(k_values, wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.show()


# Run k-means clustering
kmeans = KMeans(n_clusters=3)  # Assuming 3 clusters
kmeans.fit(reduced_data)

# Get the cluster labels
cluster_labels = kmeans.labels_

# Plot the clusters
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering')
plt.show()


# Reduce perplexity value
perplexity = 5

# Perform t-SNE
tsne = TSNE(n_components=2, perplexity=perplexity)
data_tsne = tsne.fit_transform(normalized_data)

# Plot the data
plt.scatter(data_tsne[:, 0], data_tsne[:, 1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('t-SNE Visualization')
plt.show()

# Perform UMAP
umap_model = umap.UMAP(n_components=2)
data_umap = umap_model.fit_transform(normalized_data)

# Plot the data
plt.scatter(data_umap[:, 0], data_umap[:, 1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('UMAP Visualization')
plt.show()