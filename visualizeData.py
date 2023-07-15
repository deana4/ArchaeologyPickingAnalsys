import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='umap')

# Read the CSV file
df = pd.read_csv("yuval_features.csv")

# Replace all NaN values with 0
df.fillna(0, inplace=True)

# Remove the last three columns
df = df.iloc[:, :-3]

# Reduce perplexity value
perplexity = 5

# Perform t-SNE
tsne = TSNE(n_components=2, perplexity=perplexity)
data_tsne = tsne.fit_transform(df)

# Plot the data
plt.scatter(data_tsne[:, 0], data_tsne[:, 1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('t-SNE Visualization')
plt.show()

# Perform UMAP
umap_model = umap.UMAP(n_components=2)
data_umap = umap_model.fit_transform(df)

# Plot the data
plt.scatter(data_umap[:, 0], data_umap[:, 1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('UMAP Visualization')
plt.show()