import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_bar(x, y, x_label='', y_label='', title='', x_lim=0, y_lim=0, x_size=15, y_size=7):
    fig = plt.figure(figsize=(x_size, y_size))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(title)
    if x_lim > 0 and y_lim > 0:
        ax1.set_ylim(x_lim, y_lim)
    plt.bar(x, y, color=mcolors.BASE_COLORS);


def plot_box(x, y, x_label='', y_label='', title='', x_size=12, y_size=7):
    fig = plt.figure(figsize=(x_size, y_size))
    ax1 = fig.add_subplot(111)
    plt.boxplot(y, labels=x, vert=False, whis=50)

    plt.xlabel(x_label)
    plt.title(title)
    plt.show();