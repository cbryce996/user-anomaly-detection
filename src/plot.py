import matplotlib.pyplot as plt

def visualize_clusters(data_frame, labels, file_path):
    fig = plt.figure(figsize =(9, 9))
    ax = fig.add_subplot(projection="3d")

    ax.scatter(data_frame['total_users_attempted'], data_frame['user_password_fails'], c = labels.astype(float), picker=True)

    plt.savefig(file_path)
    plt.show()

    fig = plt.figure(figsize =(9, 9))
    plt.hist(data_frame)
    plt.savefig('../assets/images/histogram.png')