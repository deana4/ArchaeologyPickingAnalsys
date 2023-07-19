import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def norm(df):
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
    return normalized_data

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
    kmeans = KMeans(n_clusters=k)  # Assuming 3 clusters
    kmeans.fit(reduced_data)

    # Get the cluster labels
    cluster_labels = kmeans.labels_

    # Plot the clusters
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('K-means Clustering')
    plt.show()

def gian_df_plot(df_original, best_features, smallValues = False , t = 0.5):
    from sklearn.preprocessing import MinMaxScaler
    df = df_original.copy()
    import seaborn as sns
    sns.set_style("darkgrid")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    scaler = MinMaxScaler()

    conditions = [
        df['Tested'] == 'matan',
        df['Tested'] == 'yuval',
        df['Tested'] == 'nir'
    ]

    choices = [1, 2, 3]

    df['full'] = np.select(conditions, choices, default=0)

    scaled_full = pd.DataFrame(scaler.fit_transform(df[best_features + ['full']]), index=df.index, columns=best_features + ['full'])

    if smallValues == True:
        for column in scaled_full.columns:
            if column == 'full':
                break
            scaled_full[column] = scaled_full.loc[scaled_full[column] < t, column]

    mdf_full = pd.melt(scaled_full, id_vars=['full'])

    fig = plt.figure(figsize=(25,8))


    sns.boxplot(x="variable", y="value", data=mdf_full, hue='full', hue_order=[0, 0.5, 1], width=0.4,
                notch=False, showfliers=False, dodge=True, medianprops={"color": "black"}, boxprops={'alpha': 0.5},
                palette={0: "lightcoral", 0.5: "darkturquoise", 1: "mediumseagreen"})


    sns.swarmplot(x="variable", y="value", data=mdf_full, hue='full', hue_order=[0, 0.5, 1], dodge=True, alpha = 0.6 , zorder = 1, color = "black")
    # sns.stripplot(x="variable", y="value", data=mdf_full, hue='full', hue_order=[0, 0.5, 1], jitter=0, dodge=True, alpha = 0.6 , zorder = 1, color = "black")

    plt.xlabel('Features',labelpad=20)
    plt.ylabel('Scaled value',labelpad=20)
    plt.title('Best features Matan vs Yuval', pad=20, fontweight='bold')

    matan_patch = mpatches.Patch(color='lightcoral', label='Matan', edgecolor='black', linewidth=1)
    yuval_patch = mpatches.Patch(color='darkturquoise', label='Yuval', edgecolor='black', linewidth=1)
    nir_patch = mpatches.Patch(color='mediumseagreen', label='Nir', edgecolor='black', linewidth=1)

    # Show the legend
    plt.legend(title='Group', loc='upper right', handles=[matan_patch, yuval_patch, nir_patch])

    plt.show()

if __name__ == '__main__':
    # Read the CSV file
    df = pd.read_csv("full_features.csv")
    normalized_data = norm(df)
    reduce_data = reduceDims(normalized_data, 'tsne')
    k=3
    Kmeans(reduce_data, k)
    gian_df_plot(df, ['frames_brush_per_video', '%OfPaperTwizzers', 'switch', 'frames_brush_per_video', 'frames_tw_per_video', 'avg_brush_length'])
