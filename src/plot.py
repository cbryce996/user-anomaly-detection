import matplotlib.pyplot as plt

def visualize_clusters(data_frame, labels, file_path):
    fig = plt.figure(figsize =(9, 9))
    ax = fig.add_subplot(projection="3d")

    ax.scatter(data_frame[0], data_frame[1], data_frame[2], c = labels.astype(float), picker=True)

    plt.savefig(file_path)
    plt.show()