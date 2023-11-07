import json
import matplotlib.pyplot as plt


def main():
    # path = "result.json"
    #
    # f = open(path)
    # json_data_clv = json.load(f)

    path = "test.json"
    f1 = open(path)
    json_test = json.load(f1)

    names = ["wine"]

    for name in names:
        data = json_test[name]
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
