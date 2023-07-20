import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score
import seaborn as sns
from scipy import stats

def preProc(df):
    z_scores_tresh= 3
    zeros_tresh = 0.7
    df.drop(df.columns[-1], axis=1, inplace=True)

    df.fillna(0, inplace=True)

    # Step 3: Convert the second last column to numerical values
    label_mapping = {'1st': 1, '2nd': 2, '3rd': 3}
    df['Half an hour'] = df['Half an hour'].map(label_mapping)

    # Step 4: Split the data into features (X) and labels (y)

    y = df.iloc[:, -2]  # Last column (labels)
    X = df
    X.drop(X.columns[-2], axis=1, inplace=True)

    # Step 5: Convert the labels into numerical values
    label_mapping = {'yuval': 0, 'matan': 1, 'nir': 2}
    y = y.map(label_mapping)

    threshold_zeros = X.shape[0] * zeros_tresh
    non_zero_counts = X.astype(bool).sum(axis=0)
    features_to_drop = non_zero_counts[non_zero_counts <= threshold_zeros].index
    X.drop(features_to_drop, axis=1, inplace=True)

    outlier_indices = []
    for label in y.unique():
        label_indices = y[y == label].index
        X_label = X.loc[label_indices]

        # Calculate z-scores for each feature within the label group
        z_scores = np.abs(stats.zscore(X_label, axis=0))

        # Find outliers in each feature and combine the indices of all outlier rows
        outliers = np.where(z_scores > z_scores_tresh)
        outlier_indices.extend(label_indices[outliers[0]])

    # Drop outlier rows
    X.drop(outlier_indices, inplace=True)
    y.drop(outlier_indices, inplace=True)

    return X, y

def norm(df):
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return df

def reduceDims(data, method):
    n_components = 2
    perplexity=5
    if method == 'pca':
        model = PCA(n_components=n_components)
    else:
        model = TSNE(n_components=n_components, perplexity=perplexity)

    reduce_data = model.fit_transform(data)
    return reduce_data

def Kmeans(reduced_data, k):
    # Run k-means clustering
    kmeans = KMeans(n_clusters=k, n_init=10)  # Assuming 3 clusters
    kmeans.fit(reduced_data)

    # Get the cluster labels
    cluster_labels = kmeans.labels_

    # Plot the clusters
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('K-means Clustering')
    plt.show()

def Kmeans_with_labels(reduced_data, k, real_labels):
    # Run k-means clustering
    kmeans = KMeans(n_clusters=k, n_init=10)  # Assuming 3 clusters
    kmeans.fit(reduced_data)

    # Get the cluster labels
    cluster_labels = kmeans.labels_

    # Calculate accuracy using ARI or NMI
    accuracy = calculate_accuracy(real_labels, cluster_labels)
    print(f"Accuracy: {accuracy}")

    # Plot the K-means clusters
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, marker='o', label='K-means Clusters')

    # Plot the real labels
    unique_labels = np.unique(real_labels)
    for label in unique_labels:
        label_indices = np.where(real_labels == label)[0]
        plt.scatter(reduced_data[label_indices, 0], reduced_data[label_indices, 1], marker='^', label=f'Real Label {label}')

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('K-means Clustering with Real Labels')
    plt.legend()
    plt.show()

def plot_real_labels(reduced_data, real_labels):
    unique_labels = np.unique(real_labels)
    for label in unique_labels:
        label_indices = np.where(real_labels == label)[0]
        plt.scatter(reduced_data[label_indices, 0], reduced_data[label_indices, 1], marker='^',
                    label=f'Real Label {label}')

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Real Labels')
    plt.legend()
    plt.show()
def calculate_accuracy(real_labels, cluster_labels):
    nmi = normalized_mutual_info_score(real_labels, cluster_labels)
    return nmi
def find_real_means(X, y):
    unique_labels = np.unique(y)
    real_means = []
    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        label_data = X[label_indices]
        label_mean = np.mean(label_data, axis=0)
        real_means.append(label_mean)
    return np.array(real_means)
def Kmeans_with_init(reduced_data, k, real_means):
    # Initialize K-means with the real means
    kmeans = KMeans(n_clusters=k, init=real_means, n_init=10)
    kmeans.fit(reduced_data)

    # Get the cluster labels and plot the clusters
    cluster_labels = kmeans.labels_
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, marker='o', label='K-means Clusters')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('K-means Clustering with Real Labels and Initialization')
    plt.legend()
    plt.show()
def plot_boxplot(X, y, feature_list):
    # Set up the Seaborn style
    sns.set(style="whitegrid")

    # Filter the DataFrame to include only the least important features
    X_filtered = X[feature_list]

    # Iterate through each feature and create a boxplot for each one
    for feature in X_filtered.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=y, y=X_filtered[feature], palette='viridis')
        plt.title(f'Boxplot for Least Important Feature: {feature}')
        plt.xlabel('Label')
        plt.ylabel('Feature Value')
        plt.show()

if __name__ == '__main__':
    # Read the CSV file
    df = pd.read_csv("full_features.csv")
    X, y = preProc(df)
    X_norm = norm(X)
    reduce_data = reduceDims(X_norm, 'tsne')
    plot_real_labels(reduce_data, y)
    k=3
    Kmeans(reduce_data, k)
    Kmeans_with_labels(reduce_data, k, y)
    plot_features = ['avg_brush_length']
    plot_boxplot(X, y, plot_features)
    real_means = find_real_means(reduce_data, y)
    Kmeans_with_init(reduce_data, k, real_means)


