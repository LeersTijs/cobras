from experments.get_data_set import get_data_set
import metric_learn
import time

from warnings import simplefilter
from sklearn.cluster import KMeans

simplefilter(action='ignore', category=FutureWarning)


def __convert_index_to_local(indices, cl, ml):
    conversion_dict = {}
    pairs, labels = [], []
    for (tuples, l) in [(cl, "cl"), (ml, "ml")]:
        for (i, j) in tuples:
            if i in conversion_dict and j in conversion_dict:
                pairs.append([conversion_dict[i], conversion_dict[j]])
                labels.append(-1 if l == "cl" else 1)
            else:
                try:
                    conversion_dict[i] = indices.index(i)
                    conversion_dict[j] = indices.index(j)
                    pairs.append([conversion_dict[i], conversion_dict[j]])
                    labels.append(-1 if l == "cl" else 1)
                except ValueError:
                    pass

    return pairs, labels


def test_mmc(data, indices, cl, ml):
    data_to_cluster = data[indices, :]
    pairs, labels = __convert_index_to_local(indices, cl, ml)
    start_time = time.time()

    mmc = metric_learn.MMC(preprocessor=data_to_cluster)
    mmc.fit(pairs, labels)
    Data_transformed = mmc.transform(data_to_cluster)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(Data_transformed)
    end_time = time.time()
    print(kmeans.labels_)
    print(f"time: {end_time - start_time}")


def test_itml(data, indices, cl, ml):
    data_to_cluster = data[indices, :]
    pairs, labels = __convert_index_to_local(indices, cl, ml)
    start_time = time.time()

    itml = metric_learn.ITML(preprocessor=data_to_cluster)
    itml.fit(pairs, labels)
    Data_transformed = itml.transform(data_to_cluster)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(Data_transformed)
    end_time = time.time()
    print(kmeans.labels_)
    print(f"time: {end_time - start_time}")


def test_sdml():
    from metric_learn import SDML_Supervised
    from sklearn.datasets import load_iris
    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']
    sdml = SDML_Supervised(n_constraints=200)
    sdml.fit(X, Y)
    # print(f"time: {end_time - start_time}")


def main():
    data, labels = get_data_set("test")
    indices = [0, 1, 2, 3, 4, 5, 6]
    cl = [(0, 3), (1, 5), (4, 6)]
    ml = [(1, 2), (3, 4), (5, 6)]
    print("True labels")
    print(labels[indices])

    print("MMC")
    # test_mmc(data, indices, cl, ml)

    print("\nITML")
    # test_itml(data, indices, cl, ml)

    print("\nLSML")
    print("No, the fit uses quadruplets. The method is made for the case were pairwise constraint are difficlult to "
          "obtain")

    print("\nSDML")
    test_sdml()
    # print("Unable to run it, problem with skggm")

    print("\nSCML")
    print("No, it uses triplets. (A, B, C). so A should be closer to B then C")


if __name__ == "__main__":
    # main()
    import sys
    print(sys.version)