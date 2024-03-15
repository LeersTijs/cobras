from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.datasets import make_classification, make_regression
from sklearn.manifold import TSNE

import metric_learn

from cobras_ts.querier import LabelQuerier

simplefilter(action='ignore', category=FutureWarning)
"""
Code gotten from: http://contrib.scikit-learn.org/metric-learn/auto_examples/plot_metric_learning_examples.html#sphx-glr-auto-examples-plot-metric-learning-examples-py
but updated for my need
"""


def plot_lines(ax, X, constraints, color):
    for (pt1, pt2) in constraints:
        ax.plot(X[[pt1, pt2], 0], X[[pt1, pt2], 1], color=color, linestyle="--", alpha=0.4)


def plot_set_and_its_clustering(X, y, cl: list[tuple], ml: list[tuple], transformer: str, colormap=plt.cm.Paired):
    fig, axs = plt.subplots(2, 2)

    ax_normal_data = axs[0, 0]
    ax_normal_clustered = axs[1, 0]

    ax_transformed_data = axs[0, 1]
    ax_transformed_clustered = axs[1, 1]

    if X.shape[1] > 2:
        fig.suptitle("comparison of Kmeans on the normal and transformed data, the data has more than 2 dimension. So "
                     "we use TSNE to reduce it")

        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(X)

        ax_normal_data.set_title("The normal data, with the constraints")
        ax_normal_data.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=colormap)
        plot_lines(ax_normal_data, X_tsne, cl, "r")
        plot_lines(ax_normal_data, X_tsne, ml, "g")

        # Plot the result of Kmeans
        km = KMeans(n_clusters=len(set(y)))
        km.fit(X)
        kmeans_labels = km.labels_.astype(np.int32)
        ax_normal_clustered.set_title(f"Result of Kmeans, with ari: {metrics.adjusted_rand_score(kmeans_labels, y)}")
        ax_normal_clustered.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_labels, cmap=colormap)

        if transformer == "ITML":
            learner = metric_learn.ITML(preprocessor=X, prior="random")
        elif transformer == "MMC":
            print(X)
            learner = metric_learn.MMC(preprocessor=X, init="identity", diagonal=True)
        else:
            raise Exception("should never happen")

        indices = [*range(X.shape[0])]
        pairs, label = convert_cl_ml_to_pairs(indices, cl, ml)
        print(pairs)
        print(label)
        learner.fit(pairs, label)
        X_transformed = learner.transform(X)
        tsne = TSNE(n_components=2)
        X_tsne_transformed = tsne.fit_transform(X_transformed)

        # Plot the transformed data
        ax_transformed_data.set_title("The transformed data")
        ax_transformed_data.scatter(X_tsne_transformed[:, 0], X_tsne_transformed[:, 1], c=y, cmap=colormap)

        # Plot the result of Kmeans
        km = KMeans(n_clusters=len(set(y)))
        km.fit(X_transformed)
        kmeans_labels = km.labels_.astype(np.int32)
        ax_transformed_clustered.set_title(
            f"Result of Kmeans on the transformed data, with ari: {metrics.adjusted_rand_score(kmeans_labels, y)}")
        ax_transformed_clustered.scatter(X_tsne_transformed[:, 0], X_tsne_transformed[:, 1], c=kmeans_labels,
                                         cmap=colormap)

    else:
        fig.suptitle("comparison of Kmeans on the normal and transformed data")

        # Plot the normal data
        ax_normal_data.set_title("The normal data, with the constraints")
        ax_normal_data.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap)
        plot_lines(ax_normal_data, X, cl, "r")
        plot_lines(ax_normal_data, X, ml, "g")

        # Plot the result of Kmeans
        km = KMeans(n_clusters=len(set(y)))
        km.fit(X)
        kmeans_labels = km.labels_.astype(np.int32)
        ax_normal_clustered.set_title(f"Result of Kmeans, with ari: {metrics.adjusted_rand_score(kmeans_labels, y)}")
        ax_normal_clustered.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap=colormap)

        if transformer == "ITML":
            learner = metric_learn.ITML(preprocessor=X, prior="random")
        elif transformer == "MMC":
            print(X)
            learner = metric_learn.MMC(preprocessor=X, init="identity", diagonal=True)
        else:
            raise Exception("should never happen")

        indices = [*range(X.shape[0])]
        pairs, label = convert_cl_ml_to_pairs(indices, cl, ml)
        print(pairs)
        print(label)
        learner.fit(pairs, label)
        X_transformed = learner.transform(X)

        # Plot the transformed data
        ax_transformed_data.set_title("The transformed data")
        ax_transformed_data.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, cmap=colormap)

        # Plot the result of Kmeans
        km = KMeans(n_clusters=len(set(y)))
        km.fit(X_transformed)
        kmeans_labels = km.labels_.astype(np.int32)
        ax_transformed_clustered.set_title(
            f"Result of Kmeans on the transformed data, with ari: {metrics.adjusted_rand_score(kmeans_labels, y)}")
        ax_transformed_clustered.scatter(X_transformed[:, 0], X_transformed[:, 1], c=kmeans_labels, cmap=colormap)

    plt.show()


def generate_2d_dataset(dataset_type: str, seed=-1):
    """
    dataset_type = "blob", "moon", "circle", "classification"
    """
    if seed != -1:
        np.random.seed(seed)

    if dataset_type == "blob":
        X, y = make_blobs(n_samples=200, centers=3, n_features=2, cluster_std=2)

    elif dataset_type == "moon":
        X, y = make_moons(n_samples=200, noise=0.1)

    elif dataset_type == "circle":
        X_circle, y_circle = make_circles(n_samples=200, noise=0.1)

        X_circle[y_circle == 0] = 1.5 * X_circle[y_circle == 0]
        X_circle[y_circle == 1] = 0.5 * X_circle[y_circle == 1]

        X_circle[:, 1] = X_circle[:, 1] - np.full(200, 1)
        X_circle[:, 0] = X_circle[:, 0] - np.full(200, 2)

        X, y = X_circle, y_circle

    elif dataset_type == "classification":
        X, y = make_classification(n_samples=200, n_classes=3, n_clusters_per_class=2,
                                   n_informative=3, class_sep=4., n_features=10,
                                   n_redundant=0, shuffle=True)
    elif dataset_type == "combination":

        n = 75
        X_blob, y_blob = make_blobs(n_samples=n, centers=[(-3, 2), (3, 2)], n_features=2, cluster_std=0.5)
        X_blob[:, 1] += 3

        X_moon, y_moon = make_moons(n_samples=n, noise=0.05)

        X_moon[:, 1] *= 3
        X_moon[:, 0] *= 1.25

        X_moon[:, 1] = X_moon[:, 1] - np.full(n, 1)
        X_moon[:, 0] = X_moon[:, 0] + np.full(n, 2)

        X_circle, y_circle = make_circles(n_samples=n, noise=0.1)

        X_circle[y_circle == 0] = 1.5 * X_circle[y_circle == 0]
        X_circle[y_circle == 1] = 0.5 * X_circle[y_circle == 1]

        X_circle[:, 1] = X_circle[:, 1] - np.full(n, 1)
        X_circle[:, 0] = X_circle[:, 0] - np.full(n, 2)

        X = np.concatenate((X_blob, X_moon, X_circle), axis=0)
        y = np.concatenate((y_blob, y_moon + np.full(n, 2), y_circle + np.full(n, 4)))

    else:
        raise Exception(f"the given dataset_type ({dataset_type}) is not implemented")
    return X, y


def convert_cl_ml_to_pairs(indices: list[int], cl: list[tuple], ml: list[tuple]):
    conversion_dict = {}
    pairs, labels = [], []
    for (tuples, label) in [(cl, "cl"), (ml, "ml")]:
        for (i, j) in tuples:
            if i in conversion_dict and j in conversion_dict:
                pairs.append([conversion_dict[i], conversion_dict[j]])
                labels.append(-1 if label == "cl" else 1)
            else:
                try:
                    conversion_dict[i] = indices.index(i)
                    conversion_dict[j] = indices.index(j)
                    pairs.append([conversion_dict[i], conversion_dict[j]])
                    labels.append(-1 if label == "cl" else 1)
                except ValueError:
                    pass

    return pairs, labels


def generate_random_cl_ml(indices: list[int], y: list[int], amount_of_constrains: int):
    cl, ml = [], []
    querier = LabelQuerier(y)
    while len(cl) + len(ml) < amount_of_constrains:
        pt1, pt2 = np.random.choice(indices, 2, replace=False)
        pt1, pt2 = np.sort([pt1, pt2])
        if (pt1, pt2) not in cl and (pt1, pt2) not in ml:
            if querier.query_points(pt1, pt2):
                ml.append((pt1, pt2))
            else:
                cl.append((pt1, pt2))

    return cl, ml


def create_constraints(y):
    import itertools
    import random

    # aggregate indices of same class
    zeros = np.where(y == 0)[0]
    ones = np.where(y == 1)[0]
    twos = np.where(y == 2)[0]
    # make permutations of all those points in the same class
    zeros_ = list(itertools.combinations(zeros, 2))
    ones_ = list(itertools.combinations(ones, 2))
    twos_ = list(itertools.combinations(twos, 2))
    # put them together!
    sim = np.array(zeros_ + ones_ + twos_)

    # similarily, put together indices in different classes
    dis = []
    for zero in zeros:
        for one in ones:
            dis.append((zero, one))
        for two in twos:
            dis.append((zero, two))
    for one in ones:
        for two in twos:
            dis.append((one, two))

    # pick up just enough dissimilar examples as we have similar examples
    dis = np.array(random.sample(dis, len(sim)))

    # return an array of pairs of indices of shape=(2*len(sim), 2), and the
    # corresponding labels, array of shape=(2*len(sim))
    # Each pair of similar points have a label of +1 and each pair of
    # dissimilar points have a label of -1
    return (np.vstack([np.column_stack([sim[:, 0], sim[:, 1]]),
                       np.column_stack([dis[:, 0], dis[:, 1]])]),
            np.concatenate([np.ones(len(sim)), -np.ones(len(sim))]))


def main():
    np.random.seed(31)

    # metric_learner = metric_learn.MMC()
    dataset_types = ["moon", "blob", "circle", "classification", "combination"]
    dataset_types = ["combination"]
    for t in dataset_types:
        # np.random.seed(42)
        X, y = generate_2d_dataset(t)
        indices = [*range(X.shape[0])]

        cl, ml = generate_random_cl_ml(indices, y, 30)
        plot_set_and_its_clustering(X, y, cl, ml, "ITML")

        print(f"{t}, info: #instances: {len(y)}, #classes: {len(set(y))}, #features: {X.shape[1]}")


def f():
    generate_2d_dataset("combination")


if __name__ == "__main__":
    main()
    # f()
