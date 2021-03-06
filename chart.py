import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from sklearn import metrics


def plot_bar(x, y, x_label='', y_label='', title='', x_lim=0, y_lim=0, x_size=15, y_size=7):
    fig = plt.figure(figsize=(x_size, y_size))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(title)
    if x_lim > 0 and y_lim > 0:
        ax1.set_ylim(x_lim, y_lim)
    plt.bar(x, y, color=mcolors.BASE_COLORS);


def plot_box(x, y, x_label='', y_label='', title='', x_size=10, y_size=6):
    fig = plt.figure(figsize=(x_size, y_size))
    ax1 = fig.add_subplot(111)
    plt.boxplot(y, labels=x, vert=False, whis=50)

    plt.xlabel(x_label)
    plt.title(title)
    plt.show();


def plot_silhouettes(data, clusters, x_size=12, y_size=7, _metric='euclidean'):
    cluster_labels = np.unique(clusters)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = metrics.silhouette_samples(data, clusters, metric=_metric)
    c_ax_lower, c_ax_upper = 0, 0
    cticks = []
    fig = plt.figure(figsize=(x_size, y_size))
    for i, k in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[clusters == k]
        c_silhouette_vals.sort()
        c_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(c_ax_lower, c_ax_upper), c_silhouette_vals, height=1.0,
                 edgecolor='none', color=color)

        cticks.append((c_ax_lower + c_ax_upper) / 2)
        c_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--")

    plt.yticks(cticks, cluster_labels)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')

    plt.tight_layout()
    plt.show();
    return


# Plot Precision and Recall
def plot_precision_recall(precision, recall, k=40):
    fig = plt.figure(figsize=(15, 7))
    plt.plot(range(1, k), precision, 'bo', label='Precision')
    plt.plot(range(1, k), recall, 'r+', label='Recall')

    plt.xlabel('K values')
    plt.title('AVG_Precision and AVG_Recall values relative to values of K')
    plt.legend()

    plt.show();


# Plot Group Bar Chart
def plot_group_bar(_precision, _recall):
    labels = ['KNN User_Based', 'KNN Item-Based', 'Model-Based', 'Cluster']
    precisions = _precision
    recalls = _recall

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    ax.bar(x - width / 2, precisions, width, label='Precision')
    ax.bar(x + width / 2, recalls, width, label='Recall')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Scores')
    ax.set_title('Compare Precisions and Recalls')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()
