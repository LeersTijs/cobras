from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from cobras_ts.cobras_experements.cobras_split_level import COBRAS_split_lvl
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.querier import LabelQuerier
from experments.get_data_set import get_data_set

simplefilter(action='ignore', category=FutureWarning)


def test_split_lvl(name, start_budget=5, end_budget=100, jumps=5):
    data, labels = get_data_set(name)
    aris = []
    for budget in range(start_budget, end_budget + jumps, jumps):
        print("----- budget: {} -----".format(budget))
        clusterer = COBRAS_split_lvl(data, LabelQuerier(labels), budget)
        clustering, intermediate_clusterings, runtimes, ml, cl = clusterer.cluster()
        ari = metrics.adjusted_rand_score(clustering.construct_cluster_labeling(), labels)
        print(f"Ari: {ari}, amount of clusters: {len(clustering.clusters)}")
        aris.append(ari)
    return aris


def normal_run(name, start_budget=5, end_budget=100, jumps=5):
    data, labels = get_data_set(name)
    aris = []
    for budget in range(start_budget, end_budget + jumps, jumps):
        print("----- budget: {} -----".format(budget))
        clusterer = COBRAS_kmeans(data, LabelQuerier(labels), budget)
        clustering, intermediate_clusterings, runtimes, ml, cl = clusterer.cluster()
        ari = metrics.adjusted_rand_score(clustering.construct_cluster_labeling(), labels)
        print(f"Ari: {ari}, amount of clusters: {len(clustering.clusters)}")
        aris.append(ari)
    return aris


def main():
    np.random.seed(31)
    name = "wine"
    # data, labels = get_data_set(name)
    s, e, j = 105, 105, 5
    print("####### Normal: #######")
    # norm_aris = normal_run(name, start_budget=s, end_budget=e, jumps=j)
    print("####### split lvl: ########")
    spl_aris = test_split_lvl(name, start_budget=s, end_budget=e, jumps=j)

    # plt.plot(range(s, e + j, j), norm_aris, marker='x', label="norm")
    plt.plot(range(s, e + j, j), spl_aris, marker='.', label="split")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
