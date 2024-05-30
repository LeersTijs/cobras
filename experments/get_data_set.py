import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from sklearn import preprocessing
from sklearn.datasets import make_moons


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


def generate_2D_datasets(seed=31):
    np.random.seed(seed)

    all_X, all_y = [], []

    for idx, (off_i, off_j) in enumerate([(0, 0), (4, 0), (4, 3), (0, 3)]):
        X, y = make_moons(n_samples=200, noise=0.05)
        # print(y)
        X[:, 0] += off_i
        X[:, 1] += off_j

        # max_y = len(set(all_y))
        # y = list(map(lambda x: idx, y))

        all_X.extend(X)
        all_y.extend(y)

    all_X = np.array(all_X)
    all_y = np.array(all_y)

    # colors = ["b", "g", "r", "c", "m", "y", "peru", "orange", "lime", "yellow"]
    # all_y = list(map(lambda x: colors[x], all_y))
    print(all_X.shape)
    print(all_y.shape)
    # print(all_y)

    # plt.plot(*all_X[all_y == 0].T, "bo", label="cluster 1")
    # plt.plot(*all_X[all_y == 1].T, "go", label="cluster 2")
    # plt.scatter(all_X[all_y == 0, 0], all_X[all_y == 0, 1], c="b", label="cluster 1")
    # plt.scatter(all_X[all_y == 1, 0], all_X[all_y == 1, 1], c="g", label="cluster 2")
    # plt.scatter(all_X[all_y == 2, 0], all_X[all_y == 2, 1], c="r", label="cluster 3")
    # plt.scatter(all_X[all_y == 3, 0], all_X[all_y == 3, 1], c="yellow", label="cluster 4")
    #
    # # plt.scatter(all_X[:, 0], all_X[:, 1], c=all_y, label=["cluster 1", "cluster 2"])
    # # plt.legend(["cluster 1", "cluster 2"])
    # plt.legend()
    # plt.show()
    return all_X, all_y


def get_data_set(name: str):
    name = name.lower()
    path = "D:/School/2023-2024/thesis/dataSets/"
    data, labels = [], []
    match name:
        case "iris":
            path += "iris/iris.data"
            data = np.loadtxt(path, delimiter=',', usecols=[0, 1, 2, 3])
            labels = np.genfromtxt(path, delimiter=',', dtype=str, usecols=[4])
            labels = list(map(iris, labels))
        case "wine":
            path += "wine/wine.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(1, 14))
            labels = np.loadtxt(path, delimiter=',', dtype=int, usecols=[0])
        case "ionosphere":
            path += "ionosphere/ionosphere.data"
            data = np.loadtxt(path, delimiter=",", usecols=range(0, 34))
            labels = np.genfromtxt(path, delimiter=",", dtype=str, usecols=[34])
            labels = list(map(ionosphere, labels))
        case "glass":
            path += "glass/glass.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(1, 10))
            labels = np.loadtxt(path, delimiter=',', dtype=int, usecols=[10])
        case "yeast":
            path += "yeast/yeast.data"
            data, labels = map_yeast_to_matrix(np.genfromtxt(path, delimiter=',', dtype=str))
        case "test":
            path += "test/test.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(1, 14))
            labels = np.loadtxt(path, delimiter=',', dtype=int, usecols=[0])
        case "normal_wine":
            path += "normalized_wine/normal_wine.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(1, 14))
            labels = np.loadtxt(path, delimiter=',', usecols=[0]).astype(np.int32)
        case "normal_wine_ax0":
            path += "normalized_wine_ax0/normal_wine_ax0.data"
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
            all_data_str = np.loadtxt(path, delimiter=',', dtype=str)
            mask = np.any(all_data_str == "?", axis=1)
            mask = np.invert(mask)
            all_data = np.array(all_data_str[mask], dtype=np.int32)

            data = all_data[:, 1:10]
            labels = all_data[:, 10]
        case "8moons":
            data, labels = generate_2D_datasets()

        case _:
            raise Exception(f"the dataset: {name} is not available")

    return data, labels


def get_data_summery(name: str, data, labels):
    dim = len(data[0])
    classes = len(set(labels))
    return "name: {}, #instances: {}, #dimensions: {}, #classes: {}".format(name, len(data), dim, classes)


def get_norm_data_set(name: str):
    name = name.lower()
    path = "D:/School/2023-2024/thesis/dataSets/"
    data, labels = [], []
    match name:
        case "iris":
            path += "iris/norm.data"
            data = np.loadtxt(path, delimiter=',', usecols=[0, 1, 2, 3])
            labels = np.loadtxt(path, delimiter=',', dtype=int, usecols=[4])
        case "wine":
            path += "wine/norm.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(0, 13))
            labels = np.loadtxt(path, delimiter=',', dtype=int, usecols=[13])
        case "ionosphere":
            path += "ionosphere/norm.data"
            data = np.loadtxt(path, delimiter=",", usecols=range(0, 34))
            labels = np.loadtxt(path, delimiter=",", dtype=int, usecols=[34])
        case "glass":
            path += "glass/norm.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(0, 9))
            labels = np.loadtxt(path, delimiter=',', dtype=int, usecols=[9])
        case "yeast":
            path += "yeast/norm.data"
            # data, labels = map_yeast_to_matrix(np.genfromtxt(path, delimiter=',', dtype=str))
            data = np.loadtxt(path, delimiter=',', usecols=range(0, 8))
            labels = np.loadtxt(path, delimiter=',', dtype=int, usecols=[8])

        case "test":
            path += "test/norm.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(1, 14))
            labels = np.loadtxt(path, delimiter=',', dtype=int, usecols=[0])
        case "normal_wine":
            path += "normalized_wine/norm.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(1, 14))
            labels = np.loadtxt(path, delimiter=',', usecols=[0]).astype(np.int32)
        case "normal_wine_ax0":
            path += "normalized_wine_ax0/norm.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(1, 14))
            labels = np.loadtxt(path, delimiter=',', usecols=[0]).astype(np.int32)

        case "dermatology":
            path += "dermatology/norm.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(0, 33))
            labels = np.loadtxt(path, delimiter=',', usecols=[33]).astype(np.int32)
        case "ecoli":
            path += "ecoli/norm.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(0, 7))
            labels = np.loadtxt(path, delimiter=',', dtype=int, usecols=[7])
        case "hepatitis":
            path += "hepatitis/norm.data"
            raise Exception("hepatitis has alot of missing values that I do not feel like dealing with. :)")

        case "spambase":
            path += "spambase/norm.data"
            data = np.loadtxt(path, delimiter=',', usecols=range(0, 57))
            labels = np.loadtxt(path, delimiter=',', dtype=int, usecols=[57])

        case "breast":
            path += "breast+cancer+wisconsin+original/norm.data"
            data = np.loadtxt(path, delimiter=',', usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            labels = np.loadtxt(path, delimiter=',', usecols=[9])

        case _:
            raise Exception(f"the dataset: {name} is not available")

    return data, labels


def normalize_datasets():
    names = ["iris", "wine", "ionosphere", "glass", "yeast", "ecoli", "spambase", "breast", "dermatology"]
    names = ["breast"]

    path_to_datasets = "D:/School/2023-2024/thesis/dataSets/"
    for name in names:
        print(f"------ Normalizing {name} ------")
        data, labels = get_data_set(name)

        normalized_data = preprocessing.MinMaxScaler().fit_transform(data)
        print(f"OG: {data[0]}, norm: {normalized_data[0]}")

        print(data.shape[1])
        fmt = ['%10.10f' for _ in range(data.shape[1])] + ['%d']

        added_labels = np.column_stack((normalized_data, labels))

        correct_name = name if not name == "breast" else "breast+cancer+wisconsin+original"
        np.savetxt(fname=path_to_datasets + f"{correct_name}/norm.data", X=added_labels, fmt=fmt, delimiter=',')


def count_missing_values_breast():
    path = "D:/School/2023-2024/thesis/dataSets/"
    path += "breast+cancer+wisconsin+original/breast-cancer-wisconsin.data"
    data = np.loadtxt(path, delimiter=',', dtype=str, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    unique, counts = np.unique(data, return_counts=True)
    print(dict(zip(unique, counts)))
    # labels = np.loadtxt(path, delimiter=',', usecols=[10])


def main():
    names = ["breast"]
    # names = ["iris", "wine", "ionosphere", "glass", "yeast", "dermatology"]
    # names = ["iris", "wine", "ionosphere", "glass", "yeast", "ecoli", "spambase", "breast", "dermatology"]
    # names = ["wine_normal"]
    # names = ["wine", "normal_wine", "normal_wine_ax0"]
    for name in names:
        data, labels = get_norm_data_set(name)
        print(get_data_summery(name, data, labels))

        data, labels = get_data_set(name)
        # print(get_data_summery(name, data, labels))
        # print(name)
        # print(type(labels))
        # print(type(data))
        print(get_data_summery(name, data, labels))
        # print(data[0])
        # print(labels[0])
        # print()


def get_breast():
    starting_path = "D:/School/2023-2024/thesis/dataSets/"

    # Normal
    path_to_full_data = starting_path + "breast+cancer+wisconsin+original/breast-cancer-wisconsin.data"

    all_data_str = np.loadtxt(path_to_full_data, delimiter=',', dtype=str)
    mask = np.any(all_data_str == "?", axis=1)
    mask = np.invert(mask)
    all_data = np.array(all_data_str[mask], dtype=np.int32)
    print(all_data.shape)

    data = all_data[:, 1:10]
    labels = all_data[:, 10]
    print(data.shape, labels.shape)

    # normalize the data
    # data, labels = get_data_set(name)
    #
    normalized_data = preprocessing.MinMaxScaler().fit_transform(data)
    print(f"OG: {data[0]}, norm: {normalized_data[0]}")
    #
    print(data.shape[1])
    fmt = ['%10.10f' for _ in range(data.shape[1])] + ['%d']
    #
    added_labels = np.column_stack((normalized_data, labels))
    print(added_labels.shape)
    correct_name = "breast+cancer+wisconsin+original"
    #
    # correct_name = name if not name == "breast" else "breast+cancer+wisconsin+original"
    np.savetxt(fname=starting_path + f"{correct_name}/norm.data", X=added_labels, fmt=fmt, delimiter=',')

    # Get normalized data
    path_to_norm_data = starting_path + "breast+cancer+wisconsin+original/norm.data"
    data_norm = np.loadtxt(path_to_norm_data, delimiter=',', usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    labels_norm = np.loadtxt(path_to_norm_data, delimiter=',', usecols=[9])
    print(data_norm.shape, labels_norm.shape)


if __name__ == "__main__":
    # saving_norm_wine_data_ax0()
    # saving_norm_wine_data()
    # main()
    # count_missing_values_breast()
    # normalize_datasets()
    # data, labels = get_norm_data_set("breast")
    # print(data.shape, labels.shape)
    # print(data.dtype)
    # print(get_data_summery("breast", data, labels))
    mpl.style.use("seaborn-v0_8-poster")
    generate_2D_datasets()
    # count_missing_values_breast()

    # path = "D:/School/2023-2024/thesis/dataSets/Yeast/yeast.data"
    # data = np.genfromtxt(path, delimiter=',', dtype=str)
    # print(data)
    # data, labels = map_yeast_to_matrix(data)
    # print(data[0])
    # print(labels[0])

    # get_breast()
