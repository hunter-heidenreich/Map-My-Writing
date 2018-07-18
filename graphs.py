from matplotlib import pyplot as plt


def plot_bar_graph(x, y, x_label='X', y_label='Y', title='Title'):
    plt.bar(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xlim(0, len(x))
    plt.ylim(0, max(y))
    plt.show()
