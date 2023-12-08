import json
import re
import time
from warnings import simplefilter

import numpy as np
from sklearn import metrics

from cobras_ts.cobras_experements.cobras_split_level import COBRAS_split_lvl
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.querier import LabelQuerier
from experments.get_data_set import get_data_set

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=RuntimeWarning)


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


def test_normal(name, seed_number, info, budget, amount_of_runs):
    data, labels = get_data_set(name)
    np.random.seed(seed_number)

    runs = {}
    max_queries_asked = 0

    for index in range(amount_of_runs):
        start_time = time.time()
        clusterer = COBRAS_kmeans(data, LabelQuerier(labels),
                                  max_questions=budget)
        clustering, intermediate_clustering, runtimes, ml, cl = clusterer.cluster()
        end_time = time.time()

        aris = list(map(lambda x: metrics.adjusted_rand_score(x, labels), intermediate_clustering))

        clustering_labeling = clustering.construct_cluster_labeling()
        ari = metrics.adjusted_rand_score(clustering_labeling, labels)
        t = end_time - start_time
        queries = len(ml) + len(cl)
        print(f"Budget: {budget}, "
              f"ARI: {ari}, "
              f"time: {t}, "
              f"amount of queries asked: {queries}")
        if queries > max_queries_asked:
            max_queries_asked = queries
        runs[index] = {"#queries": queries, "ari": aris, "time": runtimes}

    aris = [0] * max_queries_asked
    times = [0] * max_queries_asked
    count = [0] * max_queries_asked
    for run_number in range(amount_of_runs):
        for index in range(max_queries_asked):
            if index < runs[run_number]["#queries"]:
                aris[index] += runs[run_number]["ari"][index]
                times[index] += runs[run_number]["time"][index]
                count[index] += 1
    for i in range(max_queries_asked):
        aris[i] /= count[i]
        times[i] /= count[i]

    return {"#queries": max_queries_asked, "ari": aris, "time": times, "runs": runs}


def test_mmc(name, seed_number, info, budget, amount_of_runs):
    data, labels = get_data_set(name)

    runs = {}

    for index in range(amount_of_runs):

        result_of_one_run = dict()
        for diag in info["diagonal"]:
            print(f"----- diagonal: {diag}")
            for init in info["init"]:
                print(f"--- init: {init}")
                try:
                    np.random.seed(seed_number)
                    start_time = time.time()
                    clusterer = COBRAS_split_lvl(data, LabelQuerier(labels),
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
                    print("This probs happens cus initial_k = 1")
                    aris, runtimes = [], []
                    ml, cl = [], []
                result_of_one_run[f"{diag}, {init}"] = {"#queries": len(ml) + len(cl), "ari": aris, "time": runtimes}
                print()

        # Avg over the 6 kinds of MMC
        max_queries_asked = max(
            [result_of_one_run[f"{diag}, {init}"]["#queries"] for diag in info["diagonal"] for init in info["init"]])

        avg_ari = [0] * max_queries_asked
        avg_time = [0] * max_queries_asked
        count = [0] * max_queries_asked
        for i in range(max_queries_asked):
            for diag in info["diagonal"]:
                for init in info["init"]:
                    try:
                        avg_ari[i] += result_of_one_run[f"{diag}, {init}"]["ari"][i]
                        avg_time[i] += result_of_one_run[f"{diag}, {init}"]["time"][i]
                        count[i] += 1
                    except (IndexError, KeyError):
                        continue

        for i in range(max_queries_asked):
            if count[i] > 0:
                avg_ari[i] /= count[i]
                avg_time[i] /= count[i]

        result_of_one_run["avg"] = {"#queries": max_queries_asked, "ari": avg_ari, "time": avg_time}
        runs[index] = result_of_one_run

    result = {}
    for diag in info["diagonal"]:
        for init in info["init"]:
            max_queries_asked = max([runs[index][f"{diag}, {init}"]["#queries"] for index in range(amount_of_runs)])

            aris = [0] * max_queries_asked
            times = [0] * max_queries_asked
            count = [0] * max_queries_asked

            for run_number in range(amount_of_runs):
                for index in range(max_queries_asked):
                    if index < runs[run_number][f"{diag}, {init}"]["#queries"]:
                        aris[index] += runs[run_number][f"{diag}, {init}"]["ari"][index]
                        times[index] += runs[run_number][f"{diag}, {init}"]["time"][index]
                        count[index] += 1
            for i in range(max_queries_asked):
                aris[i] /= count[i]
                times[i] /= count[i]
            result[f"{diag}, {init}"] = {"#queries": max_queries_asked, "ari": aris, "time": times}

    max_queries_asked = max(
        [result[f"{diag}, {init}"]["#queries"] for diag in info["diagonal"] for init in info["init"]])
    avg_ari = [0] * max_queries_asked
    avg_time = [0] * max_queries_asked
    count = [0] * max_queries_asked
    for i in range(max_queries_asked):
        for diag in info["diagonal"]:
            for init in info["init"]:
                try:
                    avg_ari[i] += result[f"{diag}, {init}"]["ari"][i]
                    avg_time[i] += result[f"{diag}, {init}"]["time"][i]
                    count[i] += 1
                except (IndexError, KeyError):
                    continue

    for i in range(max_queries_asked):
        if count[i] > 0:
            avg_ari[i] /= count[i]
            avg_time[i] /= count[i]

    result["avg"] = {"#queries": max_queries_asked, "ari": avg_ari, "time": avg_time}

    result["runs"] = runs

    return result


def test_itml(name, seed_number, info, budget, amount_of_runs):
    data, labels = get_data_set(name)

    runs = {}

    for index in range(amount_of_runs):

        result_of_one_run = dict()
        for prior in info["prior"]:
            print(f"------ prior: {prior}")
            try:
                np.random.seed(seed_number)
                start_time = time.time()
                clusterer = COBRAS_split_lvl(data, LabelQuerier(labels),
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
                print("Error happened:", e)
                aris, runtimes = [], []
                ml, cl = [], []
            print()
            result_of_one_run[prior] = {"#queries": len(ml) + len(cl), "ari": aris, "time": runtimes}

        # Avg over the 3 kinds of ITML
        max_queries_asked = max([result_of_one_run[prior]["#queries"] for prior in info["prior"]])
        avg_ari = [0] * max_queries_asked
        avg_time = [0] * max_queries_asked
        count = [0] * max_queries_asked
        for i in range(max_queries_asked):
            for prior in info["prior"]:
                try:
                    avg_ari[i] += result_of_one_run[prior]["ari"][i]
                    avg_time[i] += result_of_one_run[prior]["time"][i]
                    count[i] += 1
                except (IndexError, KeyError):
                    continue

        for i in range(max_queries_asked):
            if count[i] > 0:
                avg_ari[i] /= count[i]
                avg_time[i] /= count[i]

        result_of_one_run["avg"] = {"#queries": max_queries_asked, "ari": avg_ari, "time": avg_time}
        runs[index] = result_of_one_run

    result = {}
    for prior in info["prior"]:
        max_queries_asked = max([runs[index][prior]["#queries"] for index in range(amount_of_runs)])

        aris = [0] * max_queries_asked
        times = [0] * max_queries_asked
        count = [0] * max_queries_asked
        for run_number in range(amount_of_runs):
            for index in range(max_queries_asked):
                if index < runs[run_number][prior]["#queries"]:
                    aris[index] += runs[run_number][prior]["ari"][index]
                    times[index] += runs[run_number][prior]["time"][index]
                    count[index] += 1
        for i in range(max_queries_asked):
            aris[i] /= count[i]
            times[i] /= count[i]
        result[prior] = {"#queries": max_queries_asked, "ari": aris, "time": times}

    max_queries_asked = max([result[prior]["#queries"] for prior in info["prior"]])
    avg_ari = [0] * max_queries_asked
    avg_time = [0] * max_queries_asked
    count = [0] * max_queries_asked
    for i in range(max_queries_asked):
        for prior in info["prior"]:
            try:
                avg_ari[i] += result[prior]["ari"][i]
                avg_time[i] += result[prior]["time"][i]
                count[i] += 1
            except (IndexError, KeyError):
                continue

    for i in range(max_queries_asked):
        if count[i] > 0:
            avg_ari[i] /= count[i]
            avg_time[i] /= count[i]

    result["avg"] = {"#queries": max_queries_asked, "ari": avg_ari, "time": avg_time}

    result["runs"] = runs

    return result


def test_metric_learners():
    tests = [("normal", None),
             ("MMC", {"diagonal": [False, True],
                      "init": ["identity", "covariance", "random"]}),
             ("ITML", {"prior": ["identity", "covariance", "random"]})]
    seed_number = 31
    budget = 150
    amount_of_runs = 2

    names = ["iris"]
    # names = ["iris", "wine", "ionosphere", "glass", "yeast"]

    for name in names:
        print(f"################ Using DataSet: {name} ################")
        dataset_result = {}
        for (algo, info) in tests:
            print(f"--------------- testing: {algo} ---------------")
            match algo:
                case "normal":
                    normal = test_normal(name, seed_number, info, budget, amount_of_runs)
                    dataset_result["normal"] = normal
                case "MMC":
                    mmc = test_mmc(name, seed_number, info, budget, amount_of_runs)
                    dataset_result["mmc"] = mmc
                case "ITML":
                    itml = test_itml(name, seed_number, info, budget, amount_of_runs)
                    dataset_result["itml"] = itml
                case _:
                    raise ValueError("he")
            print()
        json_object = json.dumps(dataset_result, indent=2)
        json_object = re.sub(r'": \[\s+', '": [', json_object)
        json_object = re.sub(r'(\d),\s+', r'\1, ', json_object)
        json_object = re.sub(r'(\d)\n\s+]', r'\1]', json_object)
        with open(f"testing_metric_learning_full_budget/{name}.json", "w") as f:
            f.write(json_object)

    # put_all_tests_in_one_json()


def put_all_tests_in_one_json():
    names = ["iris", "wine", "ionosphere", "glass", "yeast"]
    path = "testing_metric_learning_full_budget/"
    result = {}
    for name in names:
        real_path = path + f"{name}.json"
        with open(real_path) as f:
            data = json.load(f)
            result[name] = data
    json_object = json.dumps(result, indent=2)
    json_object = re.sub(r'": \[\s+', '": [', json_object)
    json_object = re.sub(r'(\d),\s+', r'\1, ', json_object)
    json_object = re.sub(r'(\d)\n\s+]', r'\1]', json_object)
    with open(path + "everything.json", "w") as f:
        f.write(json_object)


if __name__ == "__main__":
    test_metric_learners()
