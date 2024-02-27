import numpy as np
from sklearn import preprocessing


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


def ecoli(label):
    a = {"cp": 0,
         "im": 1,
         "pp": 2,
         "imU": 3,
         "om": 4,
         "omL": 5,
         "imL": 6,
         "imS": 7}
    return a[label]


def map_yeast_to_matrix(raw):
    data = []
    labels = []
    for raw_string_el in raw:
        row = raw_string_el.split()
        data.append(list(map(lambda element: float(element), row[1:-1])))
        labels.append(row[-1])
    l = list(set(labels))
    labels = list(map(lambda label: l.index(label), labels))
    return np.array(data), labels


def saving_norm_wine_data():
    path = "D:/School/2023-2024/thesis/dataSets/"
    path += "Wine/wine.data"
    data = np.loadtxt(path, delimiter=',', usecols=range(1, 14))
    labels = np.loadtxt(path, delimiter=',', dtype=int, usecols=[0])
    normalized_data = preprocessing.normalize(data, axis=1)
    labels_with_data = np.column_stack((labels, normalized_data))
    np.savetxt("D:/School/2023-2024/thesis/dataSets/Normalized_wine/normal_wine.data", labels_with_data, delimiter=",")


def saving_norm_wine_data_ax0():
    path = "D:/School/2023-2024/thesis/dataSets/"
    path += "Wine/wine.data"
    data = np.loadtxt(path, delimiter=',', usecols=range(1, 14))
    labels = np.loadtxt(path, delimiter=',', dtype=int, usecols=[0])
    normalized_data = preprocessing.normalize(data, axis=0)
    labels_with_data = np.column_stack((labels, normalized_data))
    np.savetxt("D:/School/2023-2024/thesis/dataSets/Normalized_wine_ax0/normal_wine_ax0.data", labels_with_data,
               delimiter=",")


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
        case "test":
            path += "Test/test.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(1, 14))
            labels = np.loadtxt(path, delimiter=',', dtype=int, usecols=[0])
        case "normal_wine":
            path += "Normalized_wine/normal_wine.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(1, 14))
            labels = np.loadtxt(path, delimiter=',', usecols=[0]).astype(np.int32)
        case "normal_wine_ax0":
            path += "Normalized_wine_ax0/normal_wine_ax0.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(1, 14))
            labels = np.loadtxt(path, delimiter=',', usecols=[0]).astype(np.int32)

        case "dermatology":
            path += "dermatology/dermatology.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(0, 33))
            labels = np.loadtxt(path, delimiter=',', usecols=[34]).astype(np.int32)
        case "ecoli":
            path += "ecoli/ecoli.data"
            data = np.loadtxt(path, usecols=range(1, 8))
            labels = np.genfromtxt(path, dtype=str, usecols=[8])
            labels = list(map(ecoli, labels))
        case "hepatitis":
            path += "hepatitis/hepatitis.data"
            raise Exception("hepatitis has alot of missing values that I do not feel like dealing with. :)")

        case "spambase":
            path += "spambase/spambase.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(0, 57))
            labels = np.loadtxt(path, delimiter=',', usecols=[57])

        case "breast":
            path += "breast+cancer+wisconsin+original/breast-cancer-wisconsin.data"
            data = np.loadtxt(path, delimiter=',', usecols=[1, 2, 3, 4, 5, 7, 8, 9])
            labels = np.loadtxt(path, delimiter=',', usecols=[10])

        case _:
            raise Exception(f"the dataset: {name} is not available")

    return data, labels


def get_data_summery(name: str, data, labels):
    dim = len(data[0])
    classes = len(set(labels))
    return "name: {}, #instances: {}, #dimensions: {}, #classes: {}".format(name, len(data), dim, classes)


def main():
    # names = ["iris", "yeast"]
    # names = ["iris", "wine", "ionosphere", "glass", "yeast", "dermatology"]
    names = ["iris", "wine", "ionosphere", "glass", "yeast", "dermatology", "ecoli", "spambase", "breast", "dermatology"]
    # names = ["wine_normal"]
    # names = ["wine", "normal_wine", "normal_wine_ax0"]
    for name in names:
        data, labels = get_data_set(name)
        print(get_data_summery(name, data, labels))
        # print(type(labels))
        # print(type(data))
        # print(data[0])
        # print(labels[0])
        print()


if __name__ == "__main__":
    # saving_norm_wine_data_ax0()
    # saving_norm_wine_data()
    main()
    # path = "D:/School/2023-2024/thesis/dataSets/Yeast/yeast.data"
    # data = np.genfromtxt(path, delimiter=',', dtype=str)
    # print(data)
    # data, labels = map_yeast_to_matrix(data)
    # print(data[0])
    # print(labels[0])
