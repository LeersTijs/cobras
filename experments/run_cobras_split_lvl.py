import json
import time
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from cobras_ts.cobras import COBRAS
from cobras_ts.cobras_experements.cobras_split_level import COBRAS_split_lvl
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.querier import LabelQuerier
from experments.get_data_set import get_data_set

simplefilter(action='ignore', category=FutureWarning)


def test_split_lvl(name, start_budget=5, end_budget=100, jumps=5, n=10, split_budget=np.inf):
    data, labels = get_data_set(name)
    dict_data = {
        "budgets": [],
        "ari": [],
    }
    # return aris
    for budget in range(start_budget, end_budget + jumps, jumps):
        print("----- budget: {} -----".format(budget))
        sum_ari = 0
        for _ in range(n):
            clusterer = COBRAS_split_lvl(data, LabelQuerier(labels), budget)
            clustering, intermediate_clusterings, runtimes, ml, cl = clusterer.cluster(split_lvl_budget=split_budget)

            sum_ari += metrics.adjusted_rand_score(clustering.construct_cluster_labeling(), labels)

        avg_ari = sum_ari / n
        dict_data["budgets"].append(budget)
        dict_data["ari"].append(avg_ari)
        print("sum ari: {}, avg: {}".format(sum_ari, avg_ari))

    return dict_data


def test_budgets():
    np.random.seed(31)
    paths = [("wine_vs_normalized_with_budget15.json", 15),
             ("wine_vs_normalized_with_budget10.json", 10),
             ("wine_vs_normalized_with_budget5.json", 5),
             ("wine_vs_normalized.json", np.inf)
             ]
    names = ["wine_normal", "wine"]
    for path, split_budget in paths:
        print(f"@@@@@@@@@ budged: {split_budget} @@@@@@@@@")
        results = dict()
        for name in names:
            print("############### {} ###############".format(name))
            budget = 150
            result = test_split_lvl(name=name,
                                    start_budget=5,
                                    end_budget=budget,
                                    jumps=5,
                                    n=5,
                                    split_budget=split_budget)
            # print(result)
            results[name] = result

        with open(path, 'w') as f:
            json.dump(results, f)


def test_normal(name, seed_number, info, budgets):
    data, labels = get_data_set(name)
    aris, times = [], []
    for budget in budgets:
        np.random.seed(seed_number)

        start_time = time.time()
        clusterer = COBRAS_kmeans(data, LabelQuerier(labels),
                                  max_questions=budget)
        clustering, _, runtimes, ml, cl = clusterer.cluster()
        end_time = time.time()
        ari = metrics.adjusted_rand_score(clustering.construct_cluster_labeling(), labels)
        t = end_time - start_time
        print(f"Budget: {budget}, "
              f"ARI: {ari}, "
              f"time: {t}")
        aris.append(ari), times.append(t)
    return {"ari": aris, "time": times}


def test_mmc(name, seed_number, info, budgets):
    data, labels = get_data_set(name)
    result = dict()
    for diag in info["diagonal"]:
        print(f"----- diagonal: {diag}")
        for init in info["init"]:
            print(f"--- init: {init}")
            aris, times = [], []
            for budget in budgets:
                np.random.seed(seed_number)
                start_time = time.time()
                clusterer = COBRAS_split_lvl(data, LabelQuerier(labels),
                                             max_questions=budget,
                                             splitting_algo={"algo": "MMC", "diagonal": diag, "init": init})
                clustering, _, runtimes, ml, cl = clusterer.cluster()
                end_time = time.time()
                ari = metrics.adjusted_rand_score(clustering.construct_cluster_labeling(), labels)
                t = end_time - start_time
                print(f"Budget: {budget}, "
                      f"ARI: {ari}, "
                      f"time: {t}")
                aris.append(ari), times.append(t)
            result[f"{diag}, {init}"] = {"ari": aris, "time": times}
            print()
    return result


def test_itml(name, seed_number, info, budgets):
    data, labels = get_data_set(name)
    result = dict()
    for prior in info["prior"]:
        print(f"------ prior: {prior}")
        aris = []
        times = []
        for budget in budgets:
            try:
                np.random.seed(seed_number)
                start_time = time.time()
                clusterer = COBRAS_split_lvl(data, LabelQuerier(labels),
                                             max_questions=budget,
                                             splitting_algo={"algo": "ITML", "prior": prior})
                clustering, _, runtimes, ml, cl = clusterer.cluster()
                end_time = time.time()
                ari = metrics.adjusted_rand_score(clustering.construct_cluster_labeling(), labels)
                t = end_time - start_time
                print(f"Budget: {budget}, "
                      f"ARI: {ari}, "
                      f"time: {t}")
                aris.append(ari), times.append(t)
            except Exception as e:
                print("Error happened:", e)
                aris.append(0), times.append(0)
        print()
        result[prior] = {"ari": aris, "time": times}
    return result


def test_metric_learners():
    tests = [("ITML", {"prior": ["covariance"]})]
    # tests = [("", None),
    #          ("MMC", {"diagonal": [False, True],
    #                   "init": ["identity", "covariance", "random"]}),
    #          ("ITML", {"prior": ["identity", "covariance", "random"]})]
    name = "wine"
    seed_number = 31
    # budgets = range(5, 155, 5)
    budgets = range(60, 61, 5)
    result = {"budget": [*budgets]}
    for (algo, info) in tests:
        print(f"--------------- testing: {algo} ---------------")
        match algo:
            case "":
                normal = test_normal(name, seed_number, info, budgets)
                result["normal"] = normal
            case "MMC":
                mmc = test_mmc(name, seed_number, info, budgets)
                result["mmc"] = mmc
            case "ITML":
                itml = test_itml(name, seed_number, info, budgets)
                result["itml"] = itml
            case _:
                raise ValueError("he")
        print()

        json_object = json.dumps(result, indent=4)
        with open("something.json", "w") as f:
            f.write(json_object)


if __name__ == "__main__":
    test_metric_learners()
