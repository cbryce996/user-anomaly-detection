import plot
import prepare
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def create_labels(data_frame):
    db_model = KMeans(n_clusters = 5, random_state=0, n_init="auto").fit(data_frame)
    labels = db_model.labels_
    return labels

def get_clusters_and_anomalies(labels):
    n_clusters = len(np.unique(labels))-1
    anomaly = list(labels).count(-1)
    return n_clusters, anomaly

# File paths
csv_file_path = "../assets/data/prepared/ssh.csv"
image_file_path = "../assets/images/scatter.png"

# Prepare Data
data = prepare.prepare_data(csv_file_path)

# Use DBSCAN to create labels dataset
labels = create_labels(data)

# Get the number of clusters and anomalies from the labels dataset
n_clusters, anomaly = get_clusters_and_anomalies(labels)

# Print clusters and anomalies
print (f"Clusters: {n_clusters}")
print (f"Anomalies: {anomaly}")

df_clusters = pd.DataFrame(labels)
df_clusters.to_csv("../assets/data/prepared/clusters.csv", index=False)

# Plot the resulting labels
plot.visualize_clusters(data, labels, image_file_path)