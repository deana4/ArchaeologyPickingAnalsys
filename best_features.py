import os
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

def plot_boxplot(X, y, feature_list):
    sns.set(style="whitegrid")

    # Filter the DataFrame to include only the least important features
    X_filtered = X[feature_list]

    for feature in X_filtered.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=y, y=X_filtered[feature], palette='viridis')
        plt.title(f'Boxplot for Least Important Feature: {feature}')
        plt.xlabel('Label')
        plt.ylabel('Feature Value')
        plt.tight_layout()

        # Save the plot to a file
        filename = f'boxplot_{feature}.png'
        plt.savefig(filename, dpi=300)

        # Close the plot to release memory
        plt.close()

def nonzero_count(arr):
    return (arr != 0).sum()

# Load your CSV data into a pandas DataFrame
# Replace 'your_data.csv' with the actual path to your CSV file
df = pd.read_csv("full_features.csv")

df.drop('Directory', axis=1, inplace=True)

df.fillna(0, inplace=True)

# Step 3: Convert the second last column to numerical values
label_mapping = {'1st': 1, '2nd': 2, '3rd': 3}
df['Half an hour'] = df['Half an hour'].map(label_mapping)

# Step 4: Split the data into features (X) and labels (y)

y = df['Tested'] # Last column (labels)
X = df.drop('Tested', axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the machine learning model you want to use for RFE
# Here, we'll use Logistic Regression as an example
model = LogisticRegression(solver='liblinear', max_iter=1000)

# Initialize the Recursive Feature Elimination (RFE) object
# Replace 'n_features_to_select' with the number of features you want to select
rfe = RFE(model, n_features_to_select=10)  # Select top 10 features

# Fit RFE on the training data to select the best features
rfe.fit(X_train_scaled, y_train)

# Get the selected features mask (True for selected features, False for discarded features)
selected_features_mask = rfe.support_

# Get the ranking of features (higher the rank, less important the feature)
feature_ranking = rfe.ranking_

# Filter the DataFrame to keep only the selected features
selected_features = X_train.columns[selected_features_mask]


plot_boxplot(X, y, selected_features)


for f in selected_features:
    feature_stats_by_label = df.groupby('Tested')[f].agg(
        ['mean', 'median', 'std', 'min', 'max', 'count', nonzero_count])
    # Add a custom header to the table
    custom_header = ["Label", "Mean", "Median", "Std", "Min", "Max", "Count", "Nonzero Count"]

    # Convert the DataFrame to a nicely formatted table
    table = tabulate(feature_stats_by_label, headers=custom_header, tablefmt="fancy_grid")
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.axis('off')  # Turn off axis for the table

    # Add the feature name as a headline
    headline = f"Feature Statistics for '{f}'"
    plt.text(0.5, 0.95, headline, ha='center', va='center', fontsize=14, fontweight='bold', fontfamily="monospace")

    # Add the table content
    plt.text(0.5, 0.5, table, ha='center', va='center', fontsize=10, fontfamily="monospace")

    plt.savefig(f'{f}.png', dpi=300, bbox_inches='tight')
