import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Load the Iris dataset from Seaborn
iris = sns.load_dataset('iris')

# Drop any rows with missing values
iris = iris.dropna()

# Separate features and target variable
X = iris.drop('species', axis=1)
y = iris['species']

# Map species names to numerical values
y = y.map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

# Perform PCA to reduce dimensions to 3
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d', elev=48, azim=134)

# Scatter plot
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap=mcolors.ListedColormap(["blue", "green", "red"]))

# Add legend
legend_labels = {'setosa': 'blue', 'versicolor': 'green', 'virginica': 'red'}
for label, color in legend_labels.items():
    ax.scatter([], [], [], c=color, label=label)
ax.legend(loc='best')


plt.show()
