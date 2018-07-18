from matplotlib import pyplot as plt


def plot_bar_graph(x, y, x_label='X', y_label='Y', title='Title', export=False, export_name='default.png'):
    """
    Plots a bar graph

    :param x: Some list of data representing x labels -- List[any]
    :param y: Some list of data representing y values -- List[any]
    :param x_label: The label of the x-axis -- str
    :param y_label: The label of the y-axis -- str
    :param title: The title of the graph -- str
    :param export: Whether or not to export the graph to a file -- bool
    :param export_name: The name of the file to save the graph to -- str
    :return: Nothing. -- It plots a bar chart
    """
    plt.bar(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xlim(0, len(x))
    plt.ylim(0, max(y))

    if export:
        plt.savefig(export_name)

    plt.show()


def plot_table(cell_data, row_labels=None, col_labels=None, export=False, export_name='default.png'):
    """
    Creates a table with matplotlib

    :param cell_data: Two dimensional data representing the values in the table -- List[List[any]]
    :param row_labels: The labels of the rows -- List[any]
    :param col_labels: The labels of the columns -- List[any]
    :param export: Whether or not to export the graph to a file -- bool
    :param export_name: The name of the file to save the graph to -- str
    :return: Nothing. -- It creates and plots a table
    """
    _ = plt.figure(figsize=(6, 1))

    _ = plt.table(cellText=cell_data, rowLabels=row_labels, colLabels=col_labels, loc='center')

    plt.axis('off')
    plt.grid(False)

    if export:
        plt.savefig(export_name, bbox_inches='tight')

    plt.show()
