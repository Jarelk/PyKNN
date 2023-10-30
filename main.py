import sys
import secrets
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(secrets.randbits(128))


def calc_mislabels(n, k, f):
    fig, axs = plt.subplots(2, 2)
    data, results = gen_data(n, k, f, axs[0, 0])
    test_points = rng.uniform(0, 10, [10000, 2]).astype(np.float32)

    knn = cv.ml.KNearest_create()
    knn.train(data, cv.ml.ROW_SAMPLE, results)
    ret, t_results, neighbours, dist = knn.findNearest(test_points, k)

    # Plot KNN results
    axs[0, 1].set_title("Test Data labeled with KNN", fontstyle='italic')
    blue = test_points[t_results.ravel() == 1]
    axs[0, 1].scatter(blue[:, 0], blue[:, 1], 5, 'b', linewidths=0.05, edgecolors='k')
    red = test_points[t_results.ravel() == 0]
    axs[0, 1].scatter(red[:, 0], red[:, 1], 5, 'r', linewidths=0.05, edgecolors='k')

    func = lambda i, j: validate_labels(test_points[i], t_results[i])

    # 1 is correct label, 0 is incorrect label
    validation = np.fromfunction(np.vectorize(func), (10000, 1), dtype=int)
    all_false = validation[validation == 0]
    false_ratio = all_false.shape[0] / 10000
    print(false_ratio)

    # Plot all false labels
    axs[1, 1].set_title("Mislabels", fontstyle='italic')
    false_labels = test_points[validation.ravel() == 0]
    axs[1, 1].scatter(false_labels[:, 0], false_labels[:, 1], 5, 'm', '^')
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.show()


def gen_data(n, k, f, axis):
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
    label = 0
    if inside_triangle(p): label = 1
    # flip label if random sample is below f
    r = rng.random()
    if r < f: label = 1 - label
    return label

def gen_labels(arr, f):
    shape = arr.shape
    func = lambda i, j: make_label(arr[i], f)
    results = np.fromfunction(np.vectorize(func), (shape[0], 1), dtype=int)
    return results


# 1 means correct label, 0 means incorrect label
def validate_labels(t_point, t_result):
    true_label = 0
    if inside_triangle(t_point): true_label = 1
    if t_result == true_label: return 1
    else: return 0


# entrypoint
if __name__ == '__main__':
    calc_mislabels(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))