import functools
import json
import re
import time
from copy import copy
from warnings import simplefilter

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn import metrics

from cobras_ts.cobras_experements.cobras_incr_budget import COBRAS_incr_budget
from cobras_ts.cobras_experements.cobras_incremental import COBRAS_incremental
from cobras_ts.cobras_experements.cobras_smart_split_level import COBRAS_smart_split_level
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.querier import LabelQuerier
from experments.get_data_set import get_data_set

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


# simplefilter(action='ignore', category=RuntimeWarning)
# simplefilter(action='ignore', category=ConvergenceWarning)

def put_tests_in_one_json(path: str, all_sets: list[str]):
    result = {}
    for name in all_sets:
        real_path = path + f"/{name}.json"
        with open(real_path) as f:
            data = json.load(f)
            result[name] = data

    json_object = json.dumps(result, indent=2)
    json_object = re.sub(r'": \[\s+', '": [', json_object)
    json_object = re.sub(r'(\d),\s+', r'\1, ', json_object)
    json_object = re.sub(r'(\d)\n\s+]', r'\1]', json_object)
    with open(path + "/everything.json", "w") as f:
        f.write(json_object)


def average_over_aris_and_times(max_queries_asked: int, n: int, runs: dict):
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

    return aris, times


def average_over_parameters(hyper_parameters: list[str], data: dict):
    max_queries_asked = max([data[parm]["#queries"] for parm in hyper_parameters])
    avg_ari = [0] * max_queries_asked
    avg_time = [0] * max_queries_asked
    count = [0] * max_queries_asked
    for i in range(max_queries_asked):
        for parm in hyper_parameters:
            try:
                avg_ari[i] += data[parm]["ari"][i]
                avg_time[i] += data[parm]["time"][i]
                count[i] += 1
            except (IndexError, KeyError):
                continue

    for i in range(max_queries_asked):
        if count[i] > 0:
            avg_ari[i] /= count[i]
            avg_time[i] /= count[i]

    return max_queries_asked, avg_ari, avg_time


def convert_hyper_parm_to_key(hyper_parameters: dict) -> str:
    dict_key = list(hyper_parameters.values())
    dict_key.pop(0)
    dict_key = functools.reduce(lambda x, y: str(y) if x == "" else x + ", " + str(y), dict_key, "")
    return dict_key


def test_normal(name: str, info: list[tuple], budget: int, n: int, seed=-1):
    if seed != -1:
        np.random.seed(seed)

    runs = {}
    max_queries_asked = 0

    for index in range(n):
        data, labels = get_data_set(name)

        start_time = time.time()
        clusterer = COBRAS_kmeans(data, LabelQuerier(labels),
                                  max_questions=budget)
        clustering, intermediate_clustering, runtimes, ml, cl = clusterer.cluster()
        end_time = time.time()

        aris = list(map(lambda x: metrics.adjusted_rand_score(x, labels), intermediate_clustering))
        # nmis = list(map(lambda x: metrics.normalized_mutual_info_score(x, labels), intermediate_clustering))

        clustering_labeling = clustering.construct_cluster_labeling()
        ari = metrics.adjusted_rand_score(clustering_labeling, labels)
        # nmi = metrics.normalized_mutual_info_score(clustering_labeling, labels)
        t = end_time - start_time
        queries = len(ml) + len(cl)
        print(f"Budget: {budget}, "
              f"ARI: {ari}, "
              # f"NMI: {nmi}, "
              f"time: {t}, "
              f"amount of queries asked: {queries}")
        if queries > max_queries_asked:
            max_queries_asked = queries
        runs[index] = {"#queries": queries, "ari": aris, "time": runtimes}

    aris, times = average_over_aris_and_times(max_queries_asked, n, runs)

    return {"#queries": max_queries_asked, "ari": aris, "time": times, "runs": runs}


def test_smart(name: str, info: list[tuple], budget: int, n: int, seed=-1):
    if seed != -1:
        np.random.seed(seed)

    runs = {}
    max_queries_asked = 0

    for index in range(n):
        data, labels = get_data_set(name)
        k = len(set(labels))

        start_time = time.time()
        clusterer = COBRAS_smart_split_level(data, LabelQuerier(labels),
                                             max_questions=budget, ground_truth_k=k)
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

    aris, times = average_over_aris_and_times(max_queries_asked, n, runs)

    return {"#queries": max_queries_asked, "ari": aris, "time": times, "runs": runs}


def test_incremental(name: str, info: list[tuple], budget: int, n: int, seed=-1):
    # if seed != -1:
    #     np.random.seed(seed)
    result = {}

    for metric_learner_name, parameters_to_test in info:
        print(metric_learner_name, parameters_to_test)
        result[metric_learner_name] = {}

        keys = []

        for hyper_parameters in parameters_to_test:
            print(metric_learner_name, hyper_parameters)

            if seed != -1:
                np.random.seed(seed)

            runs = {i: {} for i in range(n)}
            max_queries_asked = 0

            for index in range(n):
                try:
                    data, labels = get_data_set(name)

                    start_time = time.time()
                    clusterer = COBRAS_incremental(data, LabelQuerier(labels),
                                                   max_questions=budget,
                                                   splitting_algo=hyper_parameters)
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

                    if len(ml) + len(cl) > max_queries_asked:
                        max_queries_asked = len(ml) + len(cl)
                except Exception as e:
                    print("Error happened: ", e)
                    aris, runtimes = [], []
                    ml, cl = [], []
                runs[index] = {"#queries": len(ml) + len(cl), "ari": aris, "time": runtimes}

            dict_key = convert_hyper_parm_to_key(hyper_parameters)

            avg_ari, avg_time = average_over_aris_and_times(max_queries_asked, n, runs)
            result[metric_learner_name][dict_key] = {"#queries": max_queries_asked,
                                                     "ari": avg_ari, "time": avg_time,
                                                     "runs": runs}
            keys.append(dict_key)

        max_budget, overall_avg_ari, overall_avg_time = average_over_parameters(keys, result[metric_learner_name])
        result[metric_learner_name]["#queries"] = max_budget
        result[metric_learner_name]["ari"] = overall_avg_ari
        result[metric_learner_name]["time"] = overall_avg_time

    return result


def test_incr_budget(name: str, info: list[tuple], budget: int, n: int, seed=-1):
    result = {}
    min_number_of_questions = 25

    for metric_learner_name, parameters_to_test in info:
        # print(metric_learner_name, parameters_to_test)
        result[metric_learner_name] = {}

        keys = []

        for hyper_parameters in parameters_to_test:
            print(f"Learner: {metric_learner_name} with: {hyper_parameters.values()}")
            print(metric_learner_name, hyper_parameters)

            if seed != -1:
                np.random.seed(seed)

            runs = {i: {} for i in range(n)}
            max_queries_asked = 0

            for index in range(n):
                try:
                    data, labels = get_data_set(name)

                    start_time = time.time()
                    clusterer = COBRAS_incr_budget(data, LabelQuerier(labels),
                                                   max_questions=budget,
                                                   splitting_algo=hyper_parameters,
                                                   min_number_of_questions=min_number_of_questions,
                                                   debug=False)
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

                    if len(ml) + len(cl) > max_queries_asked:
                        max_queries_asked = len(ml) + len(cl)
                except Exception as e:
                    print("Error happened: ", e)
                    aris, runtimes = [], []
                    ml, cl = [], []
                runs[index] = {"#queries": len(ml) + len(cl), "ari": aris, "time": runtimes}

            dict_key = convert_hyper_parm_to_key(hyper_parameters)

            avg_ari, avg_time = average_over_aris_and_times(max_queries_asked, n, runs)
            result[metric_learner_name][dict_key] = {"#queries": max_queries_asked,
                                                     "ari": avg_ari, "time": avg_time,
                                                     "runs": runs}
            keys.append(dict_key)

        max_budget, overall_avg_ari, overall_avg_time = average_over_parameters(keys, result[metric_learner_name])
        result[metric_learner_name]["#queries"] = max_budget
        result[metric_learner_name]["ari"] = overall_avg_ari
        result[metric_learner_name]["time"] = overall_avg_time

    return result


def test_cobras(data_sets: list[str], tests: list[tuple], path: str, max_budget: int, n: int, seed=-1):
    for name in data_sets:
        print(f"############### Using: {name} ###############")
        dataset_result = {}

        for (algo, info) in tests:
            print(f"------------- testing: {algo} -------------")
            match algo:
                case "normal":
                    normal = test_normal(name, info, max_budget, n, seed)
                    dataset_result["normal"] = normal
                case "smart":
                    smart = test_smart(name, info, max_budget, n, seed)
                    dataset_result["smart"] = smart
                case "incremental":
                    incremental = test_incremental(name, info, max_budget, n, seed)
                    dataset_result["incremental"] = incremental
                case "incr_budget":
                    incr_budget = test_incr_budget(name, info, max_budget, n, seed)
                    dataset_result["incr_budget"] = incr_budget
                case _:
                    raise ValueError("The given algo is not implemented")
            print()

        json_object = json.dumps(dataset_result, indent=2)
        json_object = re.sub(r'": \[\s+', '": [', json_object)
        json_object = re.sub(r'(\d),\s+', r'\1, ', json_object)
        json_object = re.sub(r'(\d)\n\s+]', r'\1]', json_object)
        with open(f"{path}/{name}.json", "w") as f:
            f.write(json_object)


def graph_every_dataset(other, all_data, data_sets, uses_metric_learner: bool, split_in_multiple_graphs=1):
    split_lists = np.array_split(data_sets, split_in_multiple_graphs)
    # print(split_lists)

    for datasets in split_lists:
        print(f"current datasets: {datasets}")
        colors = ["b", "g", "r", "c", "m", "y", "peru", "orange", "lime", "yellow"]
        i = 0
        lines = []

        for dataset in datasets:
            print(dataset)
            data = all_data[dataset]
            normal_data = data["normal"]
            other_data = data[other]

            normal_budget = [*range(normal_data["#queries"])]
            plt.plot(normal_budget, normal_data["ari"], color=colors[i])

            if uses_metric_learner:
                # mmc and itml are in it
                mmc = other_data["mmc"]
                itml = other_data["itml"]
                mmc_budget = [*range(mmc["#queries"])]
                itml_budget = [*range(itml["#queries"])]

                plt.plot(mmc_budget, mmc["ari"], color=colors[i], linestyle="--")
                plt.plot(itml_budget, itml["ari"], color=colors[i], linestyle=":")
            else:
                # same as normal
                other_budget = [*range(other_data["#queries"])]
                plt.plot(other_budget, other_data["ari"], color=colors[i], linestyle="--")

            lines.append(Line2D([0, 1], [0, 1], linestyle="-", color=colors[i]))
            i += 1

        names = copy(list(datasets))

        names.append("COBRAS")
        lines.append(Line2D([0, 1], [0, 1], linestyle="-", color="k"))

        if uses_metric_learner:
            names.append(f"{other}+mmc")
            lines.append(Line2D([0, 1], [0, 1], linestyle="--", color="k"))

            names.append(f"{other}+itml")
            lines.append(Line2D([0, 1], [0, 1], linestyle=":", color="k"))
        else:
            names.append(other)
            lines.append(Line2D([0, 1], [0, 1], linestyle="--", color="k"))

        plt.legend(lines, names, loc=4)
        plt.show()


def graph_dataset(other: str, dataset: str, data: dict):
    normal_color = "black"
    itml_color = "cornflowerblue"
    mmc_color_diag = "orange"
    mmc_color_not_diag = "orangered"

    itml_avg_color = "blue"
    mmc_avg_color = "red"

    id_linestyle = (0, (5, 5))
    cov_linestyle = (0, (3, 1, 1, 1))
    rand_linestyle = (0, (1, 1))

    opacity = 0.6

    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    normal_data = data["normal"]
    normal_budget = [*range(normal_data["#queries"])]
    axs[0].plot(normal_budget, normal_data["ari"], color=normal_color)
    axs[1].plot(normal_budget, normal_data["time"], color=normal_color)

    axs[0].set_ylabel("ari")
    axs[1].set_ylabel("time (s)")
    axs[1].set_xlabel("budget")

    # Plotting MMC
    mmc_data = data[other]["mmc"]

    filtered_list = list(filter(lambda x: x not in ["#queries", "ari", "time"], list(mmc_data.keys())))
    for k in filtered_list:

        color_used = mmc_color_not_diag
        if "True" in k:
            color_used = mmc_color_diag

        linestyle_used = rand_linestyle
        if "id" in k:
            linestyle_used = id_linestyle
        elif "cov" in k:
            linestyle_used = cov_linestyle

        current_data = mmc_data[k]
        axs[0].plot([*range(current_data["#queries"])], current_data["ari"], color=color_used, linestyle=linestyle_used,
                    alpha=opacity)
        axs[1].plot([*range(current_data["#queries"])], current_data["time"], color=color_used,
                    linestyle=linestyle_used, alpha=opacity)

    axs[0].plot([*range(mmc_data["#queries"])], mmc_data["ari"], color=mmc_avg_color)
    axs[1].plot([*range(mmc_data["#queries"])], mmc_data["time"], color=mmc_avg_color)

    # Plotting ITML
    itml_data = data[other]["itml"]

    filtered_list = list(filter(lambda x: x not in ["#queries", "ari", "time"], list(itml_data.keys())))
    for k in filtered_list:
        current_data = itml_data[k]

        linestyle_used = rand_linestyle
        if "id" in k:
            linestyle_used = id_linestyle
        elif "cov" in k:
            linestyle_used = cov_linestyle

        axs[0].plot([*range(current_data["#queries"])], current_data["ari"], color=itml_color, linestyle=linestyle_used,
                    alpha=opacity)
        axs[1].plot([*range(current_data["#queries"])], current_data["time"], color=itml_color,
                    linestyle=linestyle_used, alpha=opacity)

    axs[0].plot([*range(itml_data["#queries"])], itml_data["ari"], color=itml_avg_color)
    axs[1].plot([*range(itml_data["#queries"])], itml_data["time"], color=itml_avg_color)

    fig.suptitle(f"normal vs {other} (itml and mmc), on dataset: {dataset}")

    names = ["COBRAS", "ITML", "MMC", "identity", "covariance", "random"]
    lines = [Line2D([0, 1], [0, 1], color="black"),
             Line2D([0, 1], [0, 1], color=itml_avg_color),
             Line2D([0, 1], [0, 1], color=mmc_avg_color),
             Line2D([0, 1], [0, 1], color="black", linestyle=id_linestyle),
             Line2D([0, 1], [0, 1], color="black", linestyle=cov_linestyle),
             Line2D([0, 1], [0, 1], color="black", linestyle=rand_linestyle)]

    axs[0].legend(lines, names, loc="lower right")
    plt.show()


def graph_normal_vs_experiment(other: str, path: str, uses_metric_learner: bool):
    # example other = incremental
    # path should point to a folder that has an "everything file"
    with open(path + "/everything.json") as f:
        all_data = json.load(f)

    # Getting all the datasets that are in the "everything file"
    data_sets = list(all_data.keys())
    print(data_sets)

    graph_every_dataset(other, all_data, data_sets, uses_metric_learner,
                        split_in_multiple_graphs=1 if uses_metric_learner else 1)

    if uses_metric_learner:
        for dataset in data_sets:
            graph_dataset(other, dataset, all_data[dataset])

    return


def main():
    # tests = [
    #     ("normal", None),
    #     ("smart", None),
    #     ("incremental",
    #      [
    #          ("mmc", {}), ("itml", {})
    #      ])
    # ]
    mmc_hyper_parameters = [
        {"algo": "MMC", "diagonal": False, "init": "identity"},
        {"algo": "MMC", "diagonal": False, "init": "covariance"},
        {"algo": "MMC", "diagonal": False, "init": "random"},
        {"algo": "MMC", "diagonal": True, "init": "identity"},
        {"algo": "MMC", "diagonal": True, "init": "covariance"},
        {"algo": "MMC", "diagonal": True, "init": "random"}
    ]

    itml_hyper_parameters = [
        {"algo": "ITML", "prior": "identity"},
        {"algo": "ITML", "prior": "covariance"},
        {"algo": "ITML", "prior": "random"}
    ]

    tests = [
        ("normal", None),
        ("smart", None),
        # ("incr_budget",
        #  [("mmc", mmc_hyper_parameters),
        #   ("itml", itml_hyper_parameters)
        #   ])
    ]

    all_sets = ["iris", "ionosphere", "glass", "yeast", "wine"]
    # test_sets = ["iris", "wine"]
    # test_sets = ["iris", "ionosphere", "glass", "yeast", "wine"]
    test_sets = ["iris", "ionosphere", "glass", "yeast", "wine", "ecoli", "spambase", "breast", "dermatology"]
    # test_sets = ["yeast"]

    path = "testing_smart_split_level/each_iteration_split_level_and_ground_k_avg"  # No "/" at the end
    seed = 31
    test_cobras(test_sets, tests, path, 150, 5, seed)
    put_tests_in_one_json(path, test_sets)
    graph_normal_vs_experiment("smart", path, False)


if __name__ == "__main__":
    main()
    # ari = metrics.adjusted_rand_score([1, 1, 1, 0, 0, 2, 0, 1], [2, 2, 2, 1, 1, 0, 1, 2])
    # print(ari)
