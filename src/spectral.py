import plot
import prepare
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering

def create_labels(data_frame):
    db_model = SpectralClustering(n_clusters = 5, random_state=0).fit(data_frame)
    labels = db_model.labels_
    return labels

def get_clusters_and_anomalies(labels):
    n_clusters = len(np.unique(labels))-1
    anomaly = list(labels).count(-1)
    normal = list(labels).count(0)
    return n_clusters, anomaly, normal

# File paths
csv_file_path = "../assets/data/processed/ssh.csv"
image_file_path = "../assets/images/spectral_scatter.png"

# Prepare Data
data = prepare.prepare_data(csv_file_path)

# Use DBSCAN to create labels dataset
labels = create_labels(data)

# Get the number of clusters and anomalies from the labels dataset
n_clusters, anomaly, normal = get_clusters_and_anomalies(labels)

# Print clusters and anomalies
print (f"Clusters: {n_clusters}")
print (f"Anomalies: {anomaly}")
print (f"Normal: {normal}")

df_clusters = pd.DataFrame(labels)

output = pd.read_csv(csv_file_path)
output['cluster'] = df_clusters
output.to_csv("../assets/data/processed/spectral.csv", index=False)

# Plot the resulting labels
plot.visualize_clusters(data, labels, image_file_path)