import json
import matplotlib.pyplot as plt


def main():
    path = "result.json"

    f = open(path)
    json_data = json.load(f)

    names = ["iris", "wine", "ionosphere", "glass", "yeast"]

    for name in names:
        data = json_data[name]
        budg = data["budgets"]
        clv = data["clv"]
        ari = data["ari"]
        plt.plot(budg, ari, label=name, marker='o')
        plt.xlabel("budget")
        plt.ylabel("ari")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
