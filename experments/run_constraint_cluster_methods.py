import matplotlib.pyplot as plt
import numpy as np
from active_semi_clustering.semi_supervised.pairwise_constraints import MPCKMeans, PCKMeans
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons

from experments.metric_learner_tests import generate_2d_dataset, generate_random_cl_ml, plot_lines


def convert_constraints_to_lists(cl: list[tuple]) -> list[list]:
    return [[i1, i2] for (i1, i2) in cl]


def plot_set_and_the_result(X, y, cl: list[tuple], ml: list[tuple], algo_str: str, split_level=-1, colormap=plt.cm.Paired):
    fig, axs = plt.subplots(1, 3)

    ax_data = axs[0]
    ax_kmeans = axs[1]
    ax_algo = axs[2]

    fig.suptitle(f"comparison of {algo_str} and Kmeans")

    # Plot the normal data
    ax_data.set_title("The normal data with the constraints")
    ax_data.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap)
    plot_lines(ax_data, X, cl, "r")
    plot_lines(ax_data, X, ml, "g")

    # Plot result of Kmeans
    km = KMeans(n_clusters=len(set(y)) if split_level == -1 else split_level)
    km.fit(X)
    kmeans_labels = km.labels_.astype(np.int32)
    ax_kmeans.set_title(f"Kmeans, with ari: {metrics.adjusted_rand_score(kmeans_labels, y)}")
    ax_kmeans.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap=colormap)

    if algo_str == "Over_cluster_kmeans":
        algo = KMeans(n_clusters=split_level * 2)
        algo.fit(X)

        plot_lines(ax_algo, X, cl, "r")
        plot_lines(ax_algo, X, ml, "g")

    elif algo_str == "CMS":
        from CMS import CMS, AutoLinearPolicy
        print(cl)

        pol = AutoLinearPolicy(X, 100)
        algo = CMS(pol, max_iterations=100, blurring=False)

        algo.fit(X, convert_constraints_to_lists(cl))
        plot_lines(ax_algo, X, cl, "r")

    elif algo_str == "PCKMeans":
        algo = PCKMeans(n_clusters=len(set(y)))
        algo.fit(X, ml=ml, cl=cl)

        plot_lines(ax_algo, X, cl, "r")
        plot_lines(ax_algo, X, ml, "g")

    elif algo_str == "MPCKMeans":
        algo = MPCKMeans(n_clusters=len(set(y)))
        algo.fit(X, ml=ml, cl=cl)

        plot_lines(ax_algo, X, cl, "r")
        plot_lines(ax_algo, X, ml, "g")

    else:
        print("what the helllll")
        raise Exception(f"the given algo: {algo_str} is not implemented")

    algo_labels = algo.labels_.astype(np.int32)
    ax_algo.set_title(f"{algo_str}, with ari: {metrics.adjusted_rand_score(algo_labels, y)}")
    ax_algo.scatter(X[:, 0], X[:, 1], c=algo_labels, cmap=colormap)

    plt.show()


def determine_split_level(X: np.ndarray, y: np.ndarray):
    si = [*range(X.shape[0])]

    ml, cl = [], []

    must_link_found = False
    max_split = X.shape[0]
    split_level = 0
    while not must_link_found:

        if len(si) == 2:
            new_si = [si[0], si[1]]
        else:
            km = KMeans(n_clusters=2)
            km.fit(X[si])
            split_labels = km.labels_.astype(np.int32)

            new_si = [[], []]
            for new_si_idx in set(split_labels):
                cur_indices = [si[idx] for idx, c in enumerate(split_labels) if c == new_si_idx]
                # si_train_indices = [x for x in cur_indices if x in si]
                new_si[new_si_idx] = cur_indices

        if len(new_si) == 1:
            split_level = max([split_level, 1])
            split_n = 2 ** int(split_level)
            return min(max_split, split_n)

        s1 = new_si[0]
        s1_centroid = np.mean(X[new_si[0], :], axis=0)
        s1_repr_idx = min(new_si[0], key=lambda x: np.linalg.norm(X[x, :] - s1_centroid))

        s2 = new_si[1]
        s2_centroid = np.mean(X[new_si[1], :], axis=0)
        s2_repr_idx = min(new_si[1], key=lambda x: np.linalg.norm(X[x, :] - s2_centroid))

        pt1 = min([s1_repr_idx, s2_repr_idx])
        pt2 = max([s1_repr_idx, s2_repr_idx])

        if y[pt1] == y[pt2]:
            ml.append((pt1, pt2))
            must_link_found = True
            continue
        else:
            cl.append((pt1, pt2))
            split_level += 1

        si_to_choose = []
        if len(s1) >= 2:
            si_to_choose.append(s1)
        if len(s2) >= 2:
            si_to_choose.append(s2)

        if len(si_to_choose) == 0:
            split_level = max([split_level, 1])
            split_n = 2 ** int(split_level)
            return min(max_split, split_n)

        si = min(si_to_choose, key=lambda x: len(x))

    split_level = max([split_level, 1])
    split_n = 2 ** int(split_level)
    return min(max_split, split_n), cl, ml


def main():
    # np.random.seed(31)

    dataset_types = ["moon", "blob", "circle", "classification", "combination"]
    # dataset_types = ["circle", "combination", "combination", "blob", "circle", "moon"]
    dataset_types = ["moon", "moon", "moon", "moon"]

    algos = ["CMS", "PCKMeans", "MPCKMeans", "Over_cluster_kmeans"]
    algo = "Over_cluster_kmeans"

    seed = -1

    for t in dataset_types:
        print(f"--------- {t} ---------")
        X, y = generate_2d_dataset(t, seed)
        # print(X.shape)
        indices = [*range(X.shape[0])]

        split_level, cl, ml = determine_split_level(X, y)
        # print(f"split_level: {split_level}, cl: {cl}, ml: {ml}\n")

        # cl, ml = generate_random_cl_ml(indices, y, 10)
        # while len(cl) != 10:
        #     cl, ml = generate_random_cl_ml(indices, y, 30)

        plot_set_and_the_result(X, y, cl, ml, algo, split_level)


if __name__ == "__main__":
    main()
    # X, y = generate_2d_dataset("moon", -1)
    # X, y = make_blobs(n_samples=20, centers=2, n_features=2, cluster_std=2, center_box=(-2, 2))
    # split_level, cl, ml = determine_split_level(X, y)
    # plot_set_and_the_result()

    # from sklearn.datasets import make_moons
    # from CMS import CMS, AutoLinearPolicy
    #
    # # Generate moons data set
    # x, y = make_moons(shuffle=False, noise=.01)
    # # Create one cannot-link constraint from center of one moon to another
    # cl = [[25, 75]]
    #
    # # Create bandwidth policy as used in our experiments
    # pol = AutoLinearPolicy(x, 100)
    # # Use nonblurring mean shift (do not move sampling points)
    # cms = CMS(pol, max_iterations=100, blurring=False, label_merge_k=.999)
    # cms.fit(x, cl)
    #
    # # Plot the result
    # from CMS.Plotting import plot_clustering
    # import matplotlib.pyplot as plt
    #
    # plot_clustering(x, cms.labels_, cms.modes_, cl=cl)
    #
    # plt.show()
