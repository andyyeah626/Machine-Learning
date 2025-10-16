import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data = load_iris()
X = data.data
y = data.target


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)


labels = kmeans.labels_
centers = kmeans.cluster_centers_


pca = PCA(n_components=2) #降维操作
X_2d = pca.fit_transform(X)
centers_2d = pca.transform(centers)

# 绘制聚类结果
plt.figure(figsize=(8,6))
plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap='viridis', s=50)
plt.scatter(centers_2d[:,0], centers_2d[:,1], c='red', s=200, alpha=0.75, marker='X')
plt.title("K-means Clustering on Iris Dataset (PCA 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
