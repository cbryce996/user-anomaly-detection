import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk

from sklearn.cluster import DBSCAN
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def read_csv_file(file_path):
    data = pd.read_csv(file_path)
    return data

def scale_and_normalize_data(data):
    scaler = StandardScaler()
    data_s = scaler.fit_transform(data)
    data_frame = pd.DataFrame(data_s)
    return data_frame

def create_labels(data_frame):
    db_model = DBSCAN(eps = 0.1, min_samples = 4).fit(data_frame)
    labels = db_model.labels_
    return labels

def visualize_clusters(data_frame, labels, file_path):
    fig = plt.figure(figsize =(9, 9))
    ax = fig.add_subplot(projection="3d")

    ax.scatter(data_frame[0], data_frame[1], data_frame[2], c = labels.astype(float))

    plt.savefig(file_path)
    plt.show()

def get_clusters_and_anomalies(labels):
    n_clusters = len(np.unique(labels))-1
    anomaly = list(labels).count(-1)
    return n_clusters, anomaly

# File paths
csv_file_path = "../assets/data/prepared/ssh.csv"
image_file_path = "../assets/images/scatter.png"

# Read csv file as panda
data = read_csv_file(csv_file_path)

# Scale and normalize data
data_frame = scale_and_normalize_data(data)

# Use DBSCAN to create labels dataset
labels = create_labels(data_frame)

# Visualize the clusters using matplotlib and save the figure
visualize_clusters(data_frame, labels, image_file_path)

# Get the number of clusters and anomalies from the labels dataset
n_clusters, anomaly = get_clusters_and_anomalies(labels)

print (f"Clusters: {n_clusters}")
print (f"Anomalies: {anomaly}")