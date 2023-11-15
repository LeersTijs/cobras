import json
import matplotlib.pyplot as plt


def main():
    path = "wine_result_normal.json"
    #
    f = open(path)
    normal = json.load(f)

    path = "wine_result_times_two.json"
    f1 = open(path)
    times_two = json.load(f1)

    names = ["wine"]

    for name in names:
        data = normal[name]
        budg = data["budgets"]
        ari = data["ari"]
        plt.plot(budg, ari, label=name + " normal", marker='x')

        data = times_two[name]
        budg = data["budgets"]
        ari = data["ari"]
        plt.plot(budg, ari, label=name + " times two", marker='.')

        plt.xlabel("budget")
        plt.ylabel("ari")

        plt.legend()
        plt.show()

    for name in names:
        data = normal[name]
        budg = data["budgets"]
        clv = data["clv"]
        plt.plot(budg, clv, label=name + " normal", marker='x')

        data = times_two[name]
        budg = data["budgets"]
        clv = data["clv"]
        plt.plot(budg, clv, label=name + " times two", marker='.')

        plt.xlabel("budget")
        plt.ylabel("clv")

        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
