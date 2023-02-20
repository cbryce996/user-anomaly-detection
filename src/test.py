import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# Read csv file as panda
data = pd.read_csv("../assets/data/prepared/ssh.csv")

# Scale and normalize data
scaler = StandardScaler()
data_s = scaler.fit_transform(data)
data_norm = pd.DataFrame(normalize(data_s))

# Use DBSCAN to create labels dataset
db_model = DBSCAN(eps = 0.05, min_samples = 10).fit(data_norm)
labels = db_model.labels_

# Plot histogram
plt.hist(labels, bins=len(np.unique(labels)), log=True)
plt.savefig("../assets/images/histogram.png")
plt.show()

# Reduce dimentionality
pca = PCA(n_components = 2)
data_reduce = pca.fit_transform(data_norm)
data_reduce = pd.DataFrame(data_reduce)
data_reduce.columns = list([f'P{i}' for i in range(1, len(data_reduce.columns)+1)])

# Plot scatter
colours = {}
colours[0] = 'r'
colours[1] = 'g'
colours[2] = 'b'
colours[3] = 'c'
colours[4] = 'm'
colours[-1] = 'k'

cvec = [colours[label] for label in labels]

r = plt.scatter(data_reduce['P1'], data_reduce['P2'], color ='r');
g = plt.scatter(data_reduce['P1'], data_reduce['P2'], color ='g');
b = plt.scatter(data_reduce['P1'], data_reduce['P2'], color ='b');
k = plt.scatter(data_reduce['P1'], data_reduce['P2'], color ='k');
c = plt.scatter(data_reduce['P1'], data_reduce['P2'], color ='c');
m = plt.scatter(data_reduce['P1'], data_reduce['P2'], color ='m');


plt.figure(figsize =(9, 9))
plt.scatter(data_reduce['P1'], data_reduce['P2'], c = cvec)

plt.legend((r, g, b, c, m, k), ('Label 0', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label -1'))

plt.savefig("../assets/images/scatter.png")
plt.show()

# Get clusters
n_clusters = len(np.unique(labels))-1

# Get anomalies
anomaly = list(labels).count(-1)

print (f"Clusters: {n_clusters}")
print (f"Abnormalities: {anomaly}")
print (np.unique(labels))