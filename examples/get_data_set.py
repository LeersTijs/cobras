import numpy as np


def iris(label):
    match label:
        case 'Iris-setosa':
            return 0
        case 'Iris-versicolor':
            return 1
        case 'Iris-virginica':
            return 2


def ionosphere(label):
    match label:
        case "g":
            return 0
        case "b":
            return 1


def map_yeast_to_matrix(raw):
    data = []
    labels = []
    for raw_string_el in raw:
        row = raw_string_el.split()
        data.append(list(map(lambda element: float(element),row[1:-1])))
        labels.append(row[-1])
    l = list(set(labels))
    labels = list(map(lambda label: l.index(label), labels))
    return np.array(data), labels


def get_data_set(name: str):
    name = name.lower()
    path = "D:/School/2023-2024/thesis/dataSets/"
    data, labels = [], []
    match name:
        case "iris":
            path += "Iris/iris.data"
            data = np.loadtxt(path, delimiter=',', usecols=[0, 1, 2, 3])
            labels = np.genfromtxt(path, delimiter=',', dtype=str, usecols=[4])
            labels = list(map(iris, labels))
        case "wine":
            path += "Wine/wine.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(1, 14))
            labels = np.loadtxt(path, delimiter=',', dtype=int, usecols=[0])
        case "ionosphere":
            path += "Ionosphere/ionosphere.data"
            data = np.loadtxt(path, delimiter=",", usecols=range(0, 34))
            labels = np.genfromtxt(path, delimiter=",", dtype=str, usecols=[34])
            labels = list(map(ionosphere, labels))
        case "glass":
            path += "Glass/glass.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(1, 10))
            labels = np.loadtxt(path, delimiter=',', dtype=int, usecols=[10])
        case "yeast":
            path += "Yeast/yeast.data"
            data, labels = map_yeast_to_matrix(np.genfromtxt(path, delimiter=',', dtype=str))

    return data, labels


def get_data_summery(name: str, data, labels):
    dim = len(data[0])
    classes = len(set(labels))
    return "name: {}, #instances: {}, #dimensions: {}, #classes: {}".format(name, len(data), dim, classes)


def main():
    names = ["iris", "yeast"]
    # names = ["iris", "wine", "ionosphere", "glass", "yeast"]
    for name in names:
        data, labels = get_data_set(name)
        print(get_data_summery(name, data, labels))
        print(type(labels))
        print(type(data))
        print(data[0])
        print(labels[0])
        print()
        print(data)


if __name__ == "__main__":
    main()
    # path = "D:/School/2023-2024/thesis/dataSets/Yeast/yeast.data"
    # data = np.genfromtxt(path, delimiter=',', dtype=str)
    # print(data)
    # data, labels = map_yeast_to_matrix(data)
    # print(data[0])
    # print(labels[0])
