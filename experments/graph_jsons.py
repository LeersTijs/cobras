import json
from copy import copy

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_budgets():
    paths = [("wine_vs_normalized_with_budget15.json", " budget = 15"),
             ("wine_vs_normalized_with_budget10.json", " budget = 10"),
             ("wine_vs_normalized_with_budget5.json", " budget = 5"),
             ("wine_vs_normalized.json", "")
             ]
    names = ["wine_normal", "wine"]

    for (path, info) in paths:
        f = open(path)
        raw_data = json.load(f)
        for name in names:
            data = raw_data[name]
            budg = data["budgets"]
            ari = data["ari"]
            plt.plot(budg, ari, label=name + info, marker='x')

    plt.xlabel("budget")
    plt.ylabel("ari")

    plt.legend()
    plt.show()


def normal_vs_mmc(dataset, normal, mmc, info):
    line_styles = ["--", ":", "-."]
    colors = ["g", "c"]

    normal_budgets = [*range(normal["#queries"])]
    # print(len(normal_budgets))

    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    axs[0].plot(normal_budgets, normal["ari"], label="normal", color="r")
    axs[1].plot(normal_budgets, normal["time"], color="r")

    axs[0].set_ylabel("ari")
    axs[1].set_ylabel("time (s)")
    axs[1].set_xlabel("budget")

    markers = [".", "o", "x", "+", "v", ">"]
    i = 0
    j = 0
    for diag in info["diagonal"]:
        color = colors[j]
        for init in info["init"]:
            style = line_styles[i]
            budgets = [*range(mmc[f"{diag}, {init}"]["#queries"])]
            extra_info = ""
            if len(budgets) == 0:
                extra_info = " : Error happened"
            axs[0].plot(budgets, mmc[f"{diag}, {init}"]["ari"], label=f"{diag}, {init}" + extra_info, color=color, linestyle=style)
            axs[1].plot(budgets, mmc[f"{diag}, {init}"]["time"], color=color, linestyle=style)
            # plt.plot(budgets, raw_data["mmc"][f"{diag}, {init}"]["ari"], label=f"{diag}, {init}")
            i += 1
        j += 1
        i = 0

    budgets = [*range(mmc["avg"]["#queries"])]
    axs[0].plot(budgets, mmc["avg"]["ari"], label="avg", color="b")
    axs[1].plot(budgets, mmc["avg"]["time"], color="b")

    fig.suptitle(f"normal vs mmc, on dataset: {dataset}")
    fig.legend()
    plt.show()


def normal_vs_itml(dataset, normal, itml, info):
    # colors = ["g", "c", "m"]
    line_styles = ["--", ":", "-."]
    normal_budgets = [*range(normal["#queries"])]
    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    axs[0].plot(normal_budgets, normal["ari"], label="normal", color="r")
    axs[1].plot(normal_budgets, normal["time"], color="r")

    axs[0].set_ylabel("ari")
    axs[1].set_ylabel("time (s)")
    axs[1].set_xlabel("budget")

    markers = [".", "o", "x", "+", "v", ">"]
    i = 0
    for init in info["prior"]:
        budgets = [*range(itml[init]["#queries"])]
        extra_info = ""
        if len(budgets) == 0:
            extra_info = " : Error happened"
        axs[0].plot(budgets, itml[init]["ari"], label=init + extra_info, color="g", linestyle=line_styles[i])
        axs[1].plot(budgets, itml[init]["time"], color="g", linestyle=line_styles[i])
        i += 1

    budgets = [*range(itml["avg"]["#queries"])]
    axs[0].plot(budgets, itml["avg"]["ari"], label="avg", color="b")
    axs[1].plot(budgets, itml["avg"]["time"], color="b")

    fig.suptitle(f"normal vs itml, on dataset: {dataset}")
    fig.legend()
    plt.show()


def only_best():
    mmc = "True, covariance"
    itml = "covariance"
    path = "./something.json"
    f = open(path)
    raw_data = json.load(f)
    budgets = raw_data["budget"]
    print(budgets)

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(budgets, raw_data["normal"]["ari"], label="normal", color="r")
    axs[0].plot(budgets, raw_data["mmc"][mmc]["ari"], label="mmc " + mmc, marker=".")
    axs[0].plot(budgets, raw_data["itml"][itml]["ari"], label="itml " + itml, marker=".")

    axs[0].set_ylabel("ari")

    axs[1].plot(budgets, raw_data["normal"]["time"], label="normal", color="r")
    axs[1].plot(budgets, raw_data["mmc"][mmc]["time"], label="mmc " + mmc, marker=".")
    axs[1].plot(budgets, raw_data["itml"][itml]["time"], label="itml " + itml, marker=".")

    axs[1].set_ylabel("time (s)")

    plt.title("normal vs mmc vs itml")
    plt.xlabel("budget")
    plt.legend()
    plt.show()


def graph_all_datasets_mmc_vs_itml(datasets: list[str], all_data: dict):
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    i = 0
    lines = []

    for dataset in datasets:
        # print(dataset)
        data = all_data[dataset]
        normal = data["normal"]
        normal_budgets = [*range(normal["#queries"])]

        mmc = data["mmc"]["avg"]
        mmc_budgets = [*range(mmc["#queries"])]

        itml = data["itml"]["avg"]
        itml_budgets = [*range(itml["#queries"])]

        plt.plot(normal_budgets, normal["ari"], label=dataset, color=colors[i])
        plt.plot(mmc_budgets, mmc["ari"], color=colors[i], linestyle="--")
        plt.plot(itml_budgets, itml["ari"], color=colors[i], linestyle=":")

        lines.append(Line2D([0, 1], [0, 1], linestyle="-", color=colors[i]))

        i += 1

    lines.append(Line2D([0,1],[0,1], linestyle="-", color="k"))
    lines.append(Line2D([0, 1], [0, 1], linestyle="--", color="k"))
    lines.append(Line2D([0, 1], [0, 1], linestyle=":", color="k"))

    names = copy(datasets)
    names.append("COBRAS")
    names.append("mmc")
    names.append("itml")

    plt.legend(lines, names)
    plt.show()


def graph_dataset(dataset: str, result: dict, info: dict):
    normal_vs_mmc(dataset, result["normal"], result["mmc"], info["MMC"])
    normal_vs_itml(dataset, result["normal"], result["itml"], info["ITML"])


def graph_testing_metric_learning():
    names = ["iris", "ionosphere", "glass", "yeast", "wine"]
    # names = ["iris"]
    info = {
        "normal": None,
        "MMC": {
            "diagonal": [False, True],
            "init": ["identity", "covariance", "random"]
        },
        "ITML": {
            "prior": ["identity", "covariance", "random"]
        }
    }
    # info = {
    #     "normal": None
    # }
    # path = "testing_metric_learning_full_budget/everything.json"
    # path = "testing_trans_min_queries_20/everything.json"
    # path = "testing_metric_learning/everything.json"
    path = "testing_transformation_full_budget/everything.json"
    with open(path) as f:
        all_data = json.load(f)

    graph_all_datasets_mmc_vs_itml(names, all_data)

    for dataset in names:
        graph_dataset(dataset, all_data[dataset], info)


if __name__ == "__main__":
    graph_testing_metric_learning()
