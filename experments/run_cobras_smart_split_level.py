import json
import re
import time
from warnings import simplefilter

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

from cobras_ts.cobras_experements.cobras_smart_split_level import COBRAS_smart_split_level
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.querier import LabelQuerier
from experments.get_data_set import get_data_set
from experments.run_cobras_split_lvl import test_normal, put_all_tests_in_one_json

simplefilter(action='ignore', category=FutureWarning)


def plot_information(name: str, k: int, number_of_clusters: list[int], split_levels: list[dict]):
    plt.axhline(k, color="r", linestyle="--")
    plt.plot(number_of_clusters)

    plt.title(name)
    plt.show()


def insert_split_levels(split_levels: list[dict], new_split_levels: list[dict]):
    l = []
    for info in split_levels:
        # {"split_level": split_level, "k": k, "#q": q}
        while new_split_levels[0]["#q"] < info["#q"]:
            l.append(new_split_levels[0])
            new_split_levels.pop(0)

        print()
    pass


def test_initial_number_of_clusters():
    names = ["iris", "ionosphere", "glass", "yeast", "wine"]
    budget = 150
    number_of_runs = 1

    for name in names:
        k = 0
        avg_number_of_clusters = []
        print(f"########### Running: {name} ###########")
        for _ in range(number_of_runs):
            data, labels = get_data_set(name)
            k = len(set(labels))

            start_time = time.time()

            clusterer = COBRAS_smart_split_level(data, LabelQuerier(labels), max_questions=budget)
            clustering, intermediate_clustering, runtimes, ml, cl, new_split_levels = clusterer.cluster()

            end_time = time.time()

            clustering_labeling = clustering.construct_cluster_labeling()
            ari = metrics.adjusted_rand_score(clustering_labeling, labels)
            t = end_time - start_time

            print(new_split_levels)
            print(f"Budget: {budget}, "
                  f"ARI: {ari}, "
                  f"#Clusters: {len(set(clustering_labeling))} "
                  f"time: {t}, "
                  f"# queries asked: {len(ml) + len(cl)}")

        print()
        avg_number_of_clusters = np.mean(np.array(avg_number_of_clusters), axis=0)
        # plot_information(name, k, avg_number_of_clusters, [])


def run_one_time(name):
    # name = "yeast"
    budget = 150
    init, diag = "random", True
    prior = "random"

    data, labels = get_data_set(name)
    # data, labels = generate_2d_dataset("combination")

    start_time = time.time()
    clusterer = COBRAS_smart_split_level(data, LabelQuerier(labels), max_questions=budget)
    # clusterer = COBRAS_kmeans(data, LabelQuerier(labels), max_questions=budget)
    # clusterer = COBRAS_incremental(data, LabelQuerier(labels),
    #                                max_questions=budget,
    #                                splitting_algo={"algo": "ITML", "prior": prior}, debug=True)

    # clusterer = COBRAS_incremental(data, LabelQuerier(labels),
    #                                max_questions=budget,
    #                                splitting_algo={"algo": "MMC", "init": init, "diagonal": diag}, debug=False)

    clustering, intermediate_clustering, runtimes, ml, cl, l = clusterer.cluster()
    end_time = time.time()

    # aris = list(map(lambda x: metrics.adjusted_rand_score(x, labels), intermediate_clustering))

    clustering_labeling = clustering.construct_cluster_labeling()
    ari = metrics.adjusted_rand_score(clustering_labeling, labels)
    t = end_time - start_time
    print(l)
    print(f"Budget: {budget}, "
          f"ARI: {ari}, "
          f"time: {t}, "
          f"amount of queries asked: {len(ml) + len(cl)}")


def test_smart(name, seed, max_budget, n):
    data, labels = get_data_set(name)
    np.random.seed(seed)

    runs = {}
    max_queries_asked = 0

    for index in range(n):
        start_time = time.time()

        clusterer = COBRAS_smart_split_level(data, LabelQuerier(labels), max_questions=max_budget)
        clustering, intermediate_clustering, runtimes, ml, cl, split_levels = clusterer.cluster()
        end_time = time.time()

        aris = list(map(lambda x: metrics.adjusted_rand_score(x, labels), intermediate_clustering))

        clustering_labeling = clustering.construct_cluster_labeling()
        ari = metrics.adjusted_rand_score(clustering_labeling, labels)
        t = end_time - start_time
        queries = len(ml) + len(cl)
        print(f"Budget: {max_budget}, "
              f"ARI: {ari}, "
              f"time: {t}, "
              f"amount of queries asked: {queries}")

        if queries > max_queries_asked:
            max_queries_asked = queries
        runs[index] = {"#queries": queries, "ari": aris, "time": runtimes}

    aris = [0] * max_queries_asked
    times = [0] * max_queries_asked
    count = [0] * max_queries_asked
    for run_number in range(n):
        for index in range(max_queries_asked):
            if index < runs[run_number]["#queries"]:
                aris[index] += runs[run_number]["ari"][index]
                times[index] += runs[run_number]["time"][index]
                count[index] += 1
    for i in range(max_queries_asked):
        aris[i] /= count[i]
        times[i] /= count[i]

    return {"#queries": max_queries_asked, "ari": aris, "time": times, "runs": runs}


def test_smart_split_level(data_sets, tests, path, max_budget, n):
    for name in data_sets:
        print(f"################# Using: {name} #################")
        seed = 31
        dataset_result = {}
        for algo in tests:
            print(f"------------- testing: {algo} -------------")
            match algo:
                case "normal":
                    normal = test_normal(name, seed, {}, max_budget, n)
                    dataset_result["normal"] = normal
                case "smart":
                    smart = test_smart(name, seed, max_budget, n)
                    dataset_result["smart"] = smart
                case _:
                    raise ValueError("he")
            print()
        json_object = json.dumps(dataset_result, indent=2)
        json_object = re.sub(r'": \[\s+', '": [', json_object)
        json_object = re.sub(r'(\d),\s+', r'\1, ', json_object)
        json_object = re.sub(r'(\d)\n\s+]', r'\1]', json_object)
        with open(f"{path}/{name}.json", "w") as f:
            f.write(json_object)


if __name__ == "__main__":
    # for data_set in ["iris", "ionosphere", "glass", "yeast", "wine"]:
    #
    #     _, labels = get_data_set(data_set)
    #     print(f"########### Running: {data_set}, k: {len(set(labels))} ###########")
    #     for _ in range(2):
    #         run_one_time(data_set)
    #
    #     print()
    all_sets = ["iris", "ionosphere", "glass", "yeast", "wine"]
    path = "testing_smart_split_level/only_ground_k"
    test_smart_split_level(all_sets, ["normal", "smart"], path, 150, 3)
    put_all_tests_in_one_json(path + "/", all_sets)
