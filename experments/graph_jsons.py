import json
import matplotlib.pyplot as plt


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


def normal_vs_mmc():
    path = "./something.json"
    f = open(path)
    raw_data = json.load(f)
    budgets = raw_data["budget"]
    print(budgets)

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(budgets, raw_data["normal"]["ari"], label="normal", color="r")
    axs[1].plot(budgets, raw_data["normal"]["time"], label="normal", color="r")

    axs[0].set_ylabel("ari")
    axs[1].set_ylabel("time (s)")

    # plt.plot(budgets, raw_data["normal"]["ari"], label="normal", color="r")

    markers = [".", "o", "x", "+", "v", ">"]
    i = 0
    for diag in [False, True]:
        for init in ["identity", "covariance", "random"]:
            axs[0].plot(budgets, raw_data["mmc"][f"{diag}, {init}"]["ari"], label=f"{diag}, {init}")
            axs[1].plot(budgets, raw_data["mmc"][f"{diag}, {init}"]["time"], label=f"{diag}, {init}")
            # plt.plot(budgets, raw_data["mmc"][f"{diag}, {init}"]["ari"], label=f"{diag}, {init}")
            i += 1

    plt.title("normal vs mmc")
    plt.legend()
    plt.show()


def normal_vs_itml():
    path = "./something.json"
    f = open(path)
    raw_data = json.load(f)
    budgets = raw_data["budget"]
    print(budgets)

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(budgets, raw_data["normal"]["ari"], label="normal", color="r")
    axs[1].plot(budgets, raw_data["normal"]["time"], label="normal", color="r")

    axs[0].set_ylabel("ari")
    axs[1].set_ylabel("time (s)")

    markers = [".", "o", "x", "+", "v", ">"]
    i = 0
    for init in ["identity", "covariance", "random"]:
        axs[0].plot(budgets, raw_data["itml"][init]["ari"], label=init)
        axs[1].plot(budgets, raw_data["itml"][init]["time"], label=init)
        i += 1

    plt.title("normal vs itml")
    plt.legend()
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


if __name__ == "__main__":
    normal_vs_mmc()
    normal_vs_itml()
    only_best()
