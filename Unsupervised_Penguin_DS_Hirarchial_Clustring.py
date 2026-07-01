import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkeage

# 1. Load the Penguins dataset from Seaborn
df = sns.load_dataset('penguins')
# Drop rows with missing values for cleaner execution
df = df.dropna()

# Extract continuous physical attributes for unsupervised grouping
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]

# 2. Scale the features (essential for distance calculations)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Plot a Dendrogram to visually determine how many clusters exist
plt.figure(figsize=(10, 6))
plt.title("Hierarchical Clustering Dendrogram (Penguins)")
# Using Ward's linkage method to minimize variance within clusters
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix, truncate_mode='lastp', p=12) 
plt.xlabel("Cluster Size / Data Indices")
plt.ylabel("Distance")
plt.show()

# 4. Fit Hierarchical Agglomerative Clustering (assuming 3 target species group)
agg_cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
df['Cluster_Labels'] = agg_cluster.fit_predict(X_scaled)

# 5. Cross-tabulate with actual species label to see how well the algorithm performed
print("\n--- Confusion Matrix Matrix vs Ground Truth Species ---")
print(pd.crosstab(df['species'], df['Cluster_Labels']))