import sys
import secrets
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import math

rng = np.random.default_rng(secrets.randbits(128))
figures = True


def test_all():
    # Don't want no figures, so we turn those off
    global figures
    figures = False
    triangles = [[[3, 3], [7, 3], [5, 7]]]
    n_1 = [100, 200, 300, 400, 500, 600, 700, 800]
    k_1 = [5]
    f_1 = [0.0]
    batch_test(n_1, k_1, f_1, triangles, 1)

    n_2 = [500]
    k_2 = [5]
    f_2 = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    batch_test(n_2, k_2, f_2, triangles, 2)

    # triangles = gen_triangles()
    triangles = gen_triangles_with_interval()
    n_3 = [500]
    k_3 = [5]
    f_3 = [0.0]
    batch_test(n_3, k_3, f_3, triangles, 3)


def gen_triangles_with_interval():
    """Make seven triangles with a perimeter interval of 1.

    Returns:
        triangles(np.array) a 7x3x2 matrix containing sets of vertices

    """
    triangles = []
    triangles.append(center_triangle([[3, 3], [7, 3], [5.55815631, 7]]))  # 13
    triangles.append(center_triangle([[3, 3], [7, 3], [7.43975018, 7]]))  # 14
    triangles.append(center_triangle([[3, 3], [7, 3], [8.43684518, 7]]))  # 15
    triangles.append(center_triangle([[3, 3], [7, 3], [9.24264069, 7]]))  # 16
    triangles.append(center_triangle([[3, 3], [7, 3], [9.95749911, 7]]))  # 17?
    triangles.append(center_triangle([[3, 3], [7, 3], [10.61941080, 7]]))  # 18?
    triangles.append(center_triangle([[3, 3], [7, 3], [11.24700886, 7]]))  # 19 I guess?
    triangles.append(center_triangle([[3, 3], [7, 3], [11.8507907, 7]]))  # 20??

    return triangles

def gen_triangles():
    """Make the seven triangles for experiment 3.

    Returns:
        triangles(np.array): a 7x3x2 matrix containing sets of vertices
    """
    triangles = []
    for i in range(7):
        triangle = [[3, 3], [7, 3], [5, 7]]
        # 1.166667 is roughly the interval we need between triangle tops to make most use of the space
        triangle[2][0] += i * 1.16666666667

        triangle = center_triangle(triangle)
        triangles.append(triangle)
    return triangles


def center_triangle(triangle):
    leftmost = triangle[0][0]
    rightmost = triangle[0][0]
    if triangle[1][0] > rightmost:
        rightmost = triangle[1][0]
    if triangle[2][0] > rightmost:
        rightmost = triangle[2][0]
    center_diff = 5.0 - (((rightmost - leftmost) / 2.0) + leftmost)
    triangle[0][0] += center_diff
    triangle[1][0] += center_diff
    triangle[2][0] += center_diff
    return triangle


def triangle_perimeter(triangle):
    xd1 = triangle[1][0] - triangle[0][0]
    yd1 = triangle[1][1] - triangle[0][1]
    xd2 = triangle[2][0] - triangle[1][0]
    yd2 = triangle[2][1] - triangle[1][1]
    xd3 = triangle[0][0] - triangle[2][0]
    yd3 = triangle[0][1] - triangle[2][1]
    s1 = math.sqrt(xd1 * xd1 + yd1 * yd1)
    s2 = math.sqrt(xd2 * xd2 + yd2 * yd2)
    s3 = math.sqrt(xd3 * xd3 + yd3 * yd3)
    return s1 + s2 + s3


def batch_test(n_param, k_param, f_param, triangles, i):
    """Runs a given experiment 20 times for each given parameter of an independent variable

    Args:
        n_param(np.array(int)): Number of data points to generate
        k_param(np.array(int)): Number of neighbours for KNN
        f_param(np.array(float)): outlier probability
        triangles(np.array): Mx3x2 array of triangle vertices
        i(int): Experiment number
    """
    test_count = 20
    # containing tuples of [var, mean, std]
    results = []
    match i:
        case 1:
            print("Running experiment 1")
            for n in n_param:
                print(f"Running test with N = {n}, k = {k_param[0]}, f = {f_param[0]}")
                test_results = []
                for t in range(test_count):
                    test_results.append(calc_mislabels(n, k_param[0], f_param[0], triangles[0]))
                results.append([n, np.mean(test_results), np.std(test_results)])
        case 2:
            print("Running experiment 2")
            for f in f_param:
                print(f"Running test with N = {n_param[0]}, k = {k_param[0]}, f = {f}")
                test_results = []
                for t in range(test_count):
                    test_results.append(calc_mislabels(n_param[0], k_param[0], f, triangles[0]))
                results.append([f, np.mean(test_results), np.std(test_results)])
        case 3:
            print("Running experiment 3")
            for tri in triangles:
                print(f"Running test with triangle = {tri}")
                test_results = []
                for t in range(test_count):
                    test_results.append(calc_mislabels(n_param[0], k_param[0], f_param[0], tri))
                perimeter = triangle_perimeter(tri)
                results.append([perimeter, np.mean(test_results), np.std(test_results)])
        case _:
            raise Exception("Wrong experiment number.")

    match i:
        case 1:
            with open("results/Experiment_1.csv", 'w', newline='') as experiment_file:
                writer = csv.writer(experiment_file, dialect='excel')
                writer.writerow(["N", "Mean", "Standard Deviation"])
                writer.writerows(results)
        case 2:
            with open("results/Experiment_2.csv", 'w', newline='') as experiment_file:
                writer = csv.writer(experiment_file, dialect='excel')
                writer.writerow(["f", "Mean", "Standard Deviation"])
                writer.writerows(results)
        case 3:
            with open("results/Experiment_3.csv", 'w', newline='') as experiment_file:
                writer = csv.writer(experiment_file, dialect='excel')
                writer.writerow(["Perimeter", "Mean", "Standard Deviation"])
                writer.writerows(results)
        case _:
            Exception("Wrong experiment number.")


def calc_mislabels(n, k, f, triangle):
    """Main function that calculates the amount of mislabels. First generates
    a dataset, then does a KNN search on a test set of 10.000 points using the previously
    generated dataset. Prints out the amount of mislabels, and generates some nice figures.

    Args:
        n(int): Size of the dataset
        k(int): The k-value of the KNN algorithm.
        f(float): The probability a label gets flipped, to generate outliers
        triangle(np.array): 3x2 array of triangle vertices

    Returns:
        mislabels(int): the amount of mislabels out of 10.000
    """
    if figures: fig, axs = plt.subplots(2, 2, figsize=[12.8, 9.6])
    else: axs = np.array([[0, 0], [0, 0]])
    data, results = gen_data(n, f, axs[0, 0], triangle)
    test_points = rng.uniform(0, 10, [10000, 2]).astype(np.float32)

    if figures:
        # Show the would-be correct labels for the test-points
        axs[0, 1].set_title("Test Data with correct labels", fontstyle='italic')
        correct_labels = np.fromfunction(np.vectorize(lambda i, j: 1 if inside_defined_triangle(test_points[i], triangle) else 0), (10000, 1), dtype=int)
        corr_blue = test_points[correct_labels.ravel() == 1]
        axs[0, 1].scatter(corr_blue[:, 0], corr_blue[:, 1], 5, 'b', linewidths=0.05, edgecolors='k')
        corr_red = test_points[correct_labels.ravel() == 0]
        axs[0, 1].scatter(corr_red[:, 0], corr_red[:, 1], 5, 'r', linewidths=0.05, edgecolors='k')

    knn = cv.ml.KNearest_create()
    knn.train(data, cv.ml.ROW_SAMPLE, results)
    ret, t_results, neighbours, dist = knn.findNearest(test_points, k)

    if figures:
        # Plot KNN results
        axs[1, 0].set_title("Test Data labeled with KNN", fontstyle='italic')
        blue = test_points[t_results.ravel() == 1]
        axs[1, 0].scatter(blue[:, 0], blue[:, 1], 5, 'b', linewidths=0.05, edgecolors='k')
        red = test_points[t_results.ravel() == 0]
        axs[1, 0].scatter(red[:, 0], red[:, 1], 5, 'r', linewidths=0.05, edgecolors='k')

    func = lambda i, j: validate_labels(test_points[i], t_results[i], triangle)
    # 1 is correct label, 0 is incorrect label
    validation = np.fromfunction(np.vectorize(func), (10000, 1), dtype=int)
    all_false = validation[validation == 0]
    false_ratio = all_false.shape[0] / 10000

    if figures:
        print(f"{all_false.shape[0]} total mislabels. Ratio of false labels: {false_ratio}\n")
        # Plot all false labels
        axs[1, 1].set_title("Mislabels", fontstyle='italic')
        false_labels = test_points[validation.ravel() == 0]
        axs[1, 1].scatter(false_labels[:, 0], false_labels[:, 1], 5, 'm', '^')
        plt.xlim([0, 10])
        plt.ylim([0, 10])
        plt.show()
        fig.savefig("fig.png")
    return all_false.shape[0]


def gen_data(n, f, axis, triangle):
    """Generate and label the dataset.

    Args:
         n(int): Size of the dataset
         f(float): The probability a label gets flipped, to generate outliers
         axis(pyplot.axis): The axis to use for plotting a figure
         triangle(np.array): 3x2 array of triangle vertices

    Returns:
        data(np.array): The dataset of points
        results(np.array): The labels corresponding to the dataset
    """
    data = rng.uniform(0, 10, [n, 2]).astype(np.float32)
    results = gen_labels(data, f, triangle)

    if figures:
        # print results in figure
        blue = data[results.ravel() == 1]
        axis.set_title("Generated data", fontstyle='italic')
        axis.scatter(blue[:, 0], blue[:, 1], 5, 'b', linewidths=0.05, edgecolors='k')
        red = data[results.ravel() == 0]
        axis.scatter(red[:, 0], red[:, 1], 5, 'r', linewidths=0.05, edgecolors='k')
    return data, results


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def inside_defined_triangle(p, triangle):
    v1 = triangle[0]
    v2 = triangle[1]
    v3 = triangle[2]
    d1 = sign(p, v1, v2)
    d2 = sign(p, v2, v3)
    d3 = sign(p, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


# def inside_triangle(p):
#     """Simple function that tests whether a point lies inside the triangle. Currently hardcoded.
#
#     Args:
#         p(tuple): A 2-element tuple representing a point
#
#     Returns:
#         bool: Whether or not the point lies inside the triangle
#     """
#     # Crappy looking but efficient
#     # Check if point lies above diagonal, the diagonal is literally just y = x lol
#     if p[1] > p[0]: return False
#     # Discard all below triangle
#     if p[1] < 3.0: return False
#     # Discard all to the right of triangle
#     if p[0] > 7.0: return False
#     return True


# 1 means blue, 0 means red
def make_label(p, f, triangle):
    """Generates a label for a given point. 1 means blue, 0 means red.

    Args:
        p(tuple): The point being queried
        f(float): The probability the label gets flipped.
        triangle(np.array): 3x2 array of triangle vertices

    Returns:
        int: A label, with 1 corresponding to blue, and 0 corresponding to red.
    """
    label = 0
    #if inside_triangle(p): label = 1
    if inside_defined_triangle(p, triangle): label = 1
    # flip label if random sample is below f
    r = rng.random()
    if r < f: label = 1 - label
    return label


def gen_labels(arr, f, triangle):
    """Generate labels for the given dataset.

    Args:
        arr(np.array): An n x 2 array of data points to query
        f(float): The probability that any given label gets flipped, aka outlier probability.
        triangle(np.array): 3x2 array of triangle vertices

    Returns:
        np.array: The n x 1 array of labels
    """
    shape = arr.shape
    func = lambda i, j: make_label(arr[i], f, triangle)
    results = np.fromfunction(np.vectorize(func), (shape[0], 1), dtype=int)
    return results


def validate_labels(t_point, t_result, triangle):
    """Lambda used to test labels. Used in np.fromfunction.
    1 means correct label, 0 means incorrect label

    Args:
        t_point(tuple): A 2-dimensional tuple with x and y of the point
        t_result(int): the label the point got given via KNN
        triangle(np.array): 3x2 array of triangle vertices

    Returns:
        int: An int that can be interpreted as a bool. 1 means the label was correct, 0 means incorrect.
    """
    true_label = 0
    if inside_defined_triangle(t_point, triangle): true_label = 1
    if t_result == true_label: return 1
    else: return 0


# entrypoint
if __name__ == '__main__':
    # t = [[3, 3], [7, 3], [9, 7]]
    # t = center_triangle(t)
    # calc_mislabels(5000, 5, 0.10, t)
    test_all()