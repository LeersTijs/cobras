import json
import re
import time
from warnings import simplefilter

import numpy as np
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning

from cobras_ts.cobras_experements.cobras_incremental import COBRAS_incremental
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.querier import LabelQuerier
from experments.get_data_set import get_data_set
from experments.run_cobras_split_lvl import test_normal, avg_over_runs, avg_over_parameters, put_all_tests_in_one_json

from metric_learner_tests import generate_2d_dataset

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=RuntimeWarning)
simplefilter(action='ignore', category=ConvergenceWarning)


def test_mmc(name: str, seed_number: int, info: dict, budget: int, amount_of_runs: int):
    data, labels = get_data_set(name)
    runs = {i: {} for i in range(amount_of_runs)}

    hyper_parameters = [f"{diag}, {init}" for diag in info["diagonal"] for init in info["init"]]

    for diag in info["diagonal"]:
        print(f"----- diagonal: {diag}")
        for init in info["init"]:
            print(f"--- init: {init}")
            # np.random.seed(seed_number)
            for index in range(amount_of_runs):
                try:
                    start_time = time.time()
                    clusterer = COBRAS_incremental(data, LabelQuerier(labels),
                                                   max_questions=budget,
                                                   splitting_algo={"algo": "MMC", "diagonal": diag, "init": init})
                    clustering, intermediate_clustering, runtimes, ml, cl = clusterer.cluster()
                    end_time = time.time()

                    aris = list(map(lambda x: metrics.adjusted_rand_score(x, labels), intermediate_clustering))

                    clustering_labeling = clustering.construct_cluster_labeling()
                    ari = metrics.adjusted_rand_score(clustering_labeling, labels)
                    t = end_time - start_time
                    print(f"Budget: {budget}, "
                          f"ARI: {ari}, "
                          f"time: {t}, "
                          f"amount of queries asked: {len(ml) + len(cl)}")
                except Exception as e:
                    print("Error happened: ", e)
                    aris, runtimes = [], []
                    ml, cl = [], []
                runs[index][f"{diag}, {init}"] = {"#queries": len(ml) + len(cl), "ari": aris, "time": runtimes}
            print()

    result = avg_over_runs(hyper_parameters, runs, amount_of_runs)
    result["avg"] = avg_over_parameters(hyper_parameters, result)
    result["runs"] = runs

    return result


def test_itml(name: str, seed_number: int, info: dict, budget: int, amount_of_runs: int):
    data, labels = get_data_set(name)

    # Get result of every run. (Struct: runs = { 0: "identity": {"#": x, "ari": x, "time": x}, "covariance": {}}
    runs = {i: {} for i in range(amount_of_runs)}
    for prior in info["prior"]:
        print(f"------ prior: {prior}")
        # np.random.seed(seed_number)
        for index in range(amount_of_runs):
            try:
                start_time = time.time()
                clusterer = COBRAS_incremental(data, LabelQuerier(labels),
                                               max_questions=budget,
                                               splitting_algo={"algo": "ITML", "prior": prior})
                clustering, intermediate_clustering, runtimes, ml, cl = clusterer.cluster()
                end_time = time.time()

                aris = list(map(lambda x: metrics.adjusted_rand_score(x, labels), intermediate_clustering))

                clustering_labeling = clustering.construct_cluster_labeling()
                ari = metrics.adjusted_rand_score(clustering_labeling, labels)
                t = end_time - start_time
                print(f"Budget: {budget}, "
                      f"ARI: {ari}, "
                      f"time: {t}, "
                      f"amount of queries asked: {len(ml) + len(cl)}")
            except Exception as e:
                print("Error happened: ", e)
                aris, runtimes = [], []
                ml, cl = [], []
            runs[index][prior] = {"#queries": len(ml) + len(cl), "ari": aris, "time": runtimes}
        print()

    result = avg_over_runs(info["prior"], runs, amount_of_runs)
    result["avg"] = avg_over_parameters(info["prior"], result)
    result["runs"] = runs

    return result


def test_incremental_learner(data_sets, tests, path, seed, max_budget, n):
    for name in data_sets:
        print(f"############### Using DataSet: {name} ###############")
        dataset_result = {}
        for (algo, info) in tests:
            print(f"------------- testing: {algo} -------------")
            match algo:
                case "normal":
                    normal = test_normal(name, seed, info, max_budget, n)
                    dataset_result["normal"] = normal
                case "MMC":
                    mmc = test_mmc(name, seed, info, max_budget, n)
                    dataset_result["mmc"] = mmc
                case "ITML":
                    itml = test_itml(name, seed, info, max_budget, n)
                    dataset_result["itml"] = itml
                case _:
                    raise ValueError("he")
            print()
        json_object = json.dumps(dataset_result, indent=2)
        json_object = re.sub(r'": \[\s+', '": [', json_object)
        json_object = re.sub(r'(\d),\s+', r'\1, ', json_object)
        json_object = re.sub(r'(\d)\n\s+]', r'\1]', json_object)
        with open(f"{path}/{name}.json", "w") as f:
            f.write(json_object)


def main():
    path = "testing_incremental_learner_fully/"

    all_sets = ["iris", "ionosphere", "glass", "yeast", "wine"]
    # all_sets = ["wine", "iris"]

    data_sets = ["iris", "ionosphere", "glass", "yeast", "wine"]
    # data_sets = ["iris", "wine"]

    tests_to_run = [("normal", None),
                    ("MMC", {"diagonal": [False, True],
                             "init": ["identity", "covariance", "random"]}),
                    ("ITML", {"prior": ["identity", "covariance", "random"]})]
    seed, max_budget, n = 12, 150, 4

    test_incremental_learner(data_sets, tests_to_run, path, seed, max_budget, n)
    put_all_tests_in_one_json(path, all_sets)


def run_cobras_incremental():
    np.random.seed(31)
    name = "yeast"
    budget = 150
    init, diag = "random", True
    prior = "random"

    data, labels = get_data_set(name)
    # data, labels = generate_2d_dataset("combination")

    start_time = time.time()
    clusterer = COBRAS_kmeans(data, LabelQuerier(labels), max_questions=budget)
    # clusterer = COBRAS_incremental(data, LabelQuerier(labels),
    #                                max_questions=budget,
    #                                splitting_algo={"algo": "ITML", "prior": prior}, debug=True)

    # clusterer = COBRAS_incremental(data, LabelQuerier(labels),
    #                                max_questions=budget,
    #                                splitting_algo={"algo": "MMC", "init": init, "diagonal": diag}, debug=False)

    clustering, intermediate_clustering, runtimes, ml, cl = clusterer.cluster()
    end_time = time.time()

    # aris = list(map(lambda x: metrics.adjusted_rand_score(x, labels), intermediate_clustering))

    clustering_labeling = clustering.construct_cluster_labeling()
    ari = metrics.adjusted_rand_score(clustering_labeling, labels)
    t = end_time - start_time
    print(f"Budget: {budget}, "
          f"ARI: {ari}, "
          f"time: {t}, "
          f"amount of queries asked: {len(ml) + len(cl)}")


if __name__ == "__main__":
    run_cobras_incremental()
    # main()
    # test_initial_number_of_clusters()
