import sys
import secrets
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(secrets.randbits(128))


def calc_mislabels(n, k, f):
    """Main function that calculates the amount of mislabels. First generates
    a dataset, then does a KNN search on a test set of 10.000 points using the previously
    generated dataset. Prints out the amount of mislabels, and generates some nice figures.

    Args:
        n(int): Size of the dataset
        k(int): The k-value of the KNN algorithm.
        f(float): The probability a label gets flipped, to generate outliers
    """
    fig, axs = plt.subplots(2, 2)
    data, results = gen_data(n, f, axs[0, 0])
    test_points = rng.uniform(0, 10, [10000, 2]).astype(np.float32)

    # Show the would-be correct labels for the test-points
    axs[0, 1].set_title("Test Data with correct labels", fontstyle='italic')
    correct_labels = np.fromfunction(np.vectorize(lambda i, j: 1 if inside_triangle(test_points[i]) else 0), (10000, 1), dtype=int)
    corr_blue = test_points[correct_labels.ravel() == 1]
    axs[0, 1].scatter(corr_blue[:, 0], corr_blue[:, 1], 5, 'b', linewidths=0.05, edgecolors='k')
    corr_red = test_points[correct_labels.ravel() == 0]
    axs[0, 1].scatter(corr_red[:, 0], corr_red[:, 1], 5, 'r', linewidths=0.05, edgecolors='k')

    knn = cv.ml.KNearest_create()
    knn.train(data, cv.ml.ROW_SAMPLE, results)
    ret, t_results, neighbours, dist = knn.findNearest(test_points, k)

    # Plot KNN results
    axs[1, 0].set_title("Test Data labeled with KNN", fontstyle='italic')
    blue = test_points[t_results.ravel() == 1]
    axs[1, 0].scatter(blue[:, 0], blue[:, 1], 5, 'b', linewidths=0.05, edgecolors='k')
    red = test_points[t_results.ravel() == 0]
    axs[1, 0].scatter(red[:, 0], red[:, 1], 5, 'r', linewidths=0.05, edgecolors='k')

    func = lambda i, j: validate_labels(test_points[i], t_results[i])
    # 1 is correct label, 0 is incorrect label
    validation = np.fromfunction(np.vectorize(func), (10000, 1), dtype=int)
    all_false = validation[validation == 0]
    false_ratio = all_false.shape[0] / 10000
    print(f"{all_false.shape[0]} total mislabels. Ratio of false labels: {false_ratio}\n")

    # Plot all false labels
    axs[1, 1].set_title("Mislabels", fontstyle='italic')
    false_labels = test_points[validation.ravel() == 0]
    axs[1, 1].scatter(false_labels[:, 0], false_labels[:, 1], 5, 'm', '^')
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.show()


def gen_data(n, f, axis):
    """Generate and label the dataset.

    Args:
         n(int): Size of the dataset
         f(float): The probability a label gets flipped, to generate outliers
         axis(pyplot.axis): The axis to use for plotting a figure

    Returns:
        data(np.array): The dataset of points
        results(np.array): The labels corresponding to the dataset
    """
    data = rng.uniform(0, 10, [n, 2]).astype(np.float32)
    results = gen_labels(data, f)
    blue = data[results.ravel() == 1]
    # print(blue)
    axis.set_title("Generated data", fontstyle='italic')
    axis.scatter(blue[:, 0], blue[:, 1], 5, 'b', linewidths=0.05, edgecolors='k')
    red = data[results.ravel() == 0]
    axis.scatter(red[:, 0], red[:, 1], 5, 'r', linewidths=0.05, edgecolors='k')
    return data, results


def inside_triangle(p):
    """Simple function that tests whether a point lies inside the triangle. Currently hardcoded.
    TODO: Make this work for any given triangle

    Args:
        p(tuple): A 2-element tuple representing a point

    Returns:
        bool: Whether or not the point lies inside the triangle
    """
    # Crappy looking but efficient
    # Check if point lies above diagonal, the diagonal is literally just y = x lol
    if p[1] > p[0]: return False
    # Discard all below triangle
    if p[1] < 3.0: return False
    # Discard all to the right of triangle
    if p[0] > 7.0: return False
    return True

# 1 means blue, 0 means red
def make_label(p, f):
    """Generates a label for a given point. 1 means blue, 0 means red.

    Args:
        p(tuple): The point being queried
        f(float): The probability the label gets flipped.

    Returns:
        int: A label, with 1 corresponding to blue, and 0 corresponding to red.
    """
    label = 0
    if inside_triangle(p): label = 1
    # flip label if random sample is below f
    r = rng.random()
    if r < f: label = 1 - label
    return label

def gen_labels(arr, f):
    """Generate labels for the given dataset.

    Args:
        arr(np.array): An n x 2 array of data points to query
        f(float): The probability that any given label gets flipped, aka outlier probability.

    Returns:
        np.array: The n x 1 array of labels
    """
    shape = arr.shape
    func = lambda i, j: make_label(arr[i], f)
    results = np.fromfunction(np.vectorize(func), (shape[0], 1), dtype=int)
    return results


def validate_labels(t_point, t_result):
    """Lambda used to test labels. Used in np.fromfunction.
    1 means correct label, 0 means incorrect label

    Args:
        t_point(tuple): A 2-dimensional tuple with x and y of the point
        t_result(int): the label the point got given via KNN

    Returns:
        int: An int that can be interpreted as a bool. 1 means the label was correct, 0 means incorrect.
    """
    true_label = 0
    if inside_triangle(t_point): true_label = 1
    if t_result == true_label: return 1
    else: return 0


# entrypoint
# README: Takes arguments n, k, and f
if __name__ == '__main__':
    calc_mislabels(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))