import json
import matplotlib.pyplot as plt


def main():
    path = "result_selected_si_with_clv.json"

    f = open(path)
    json_data_clv = json.load(f)

    path = "result.json"
    f1 = open(path)
    json_data_normal = json.load(f1)

    names = ["iris", "wine", "ionosphere", "glass", "yeast"]

    for name in names:
        data = json_data_clv[name]
        budg = data["budgets"]
        clv = data["clv"]
        ari = data["ari"]
        plt.plot(budg, ari, label=name + " clv", marker='o')

        data = json_data_normal[name]
        budg = data["budgets"]
        clv = data["clv"]
        ari = data["ari"]
        plt.plot(budg, ari, label=name + " normal", marker='x')

    plt.xlabel("budget")
    plt.ylabel("ari")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
