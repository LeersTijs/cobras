import numpy as np
from sklearn import metrics
from warnings import simplefilter

from cobras_ts.cobras_km_clv import COBRAS_km_clv
from cobras_ts.querier.labelquerier import LabelQuerier


simplefilter(action='ignore', category=FutureWarning)


def convert_to_num(label):
    match label:
        case 'Iris-setosa':
            return 0
        case 'Iris-versicolor':
            return 1
        case 'Iris-virginica':
            return 2


budget = 100
path = 'D:/School/2023-2024/thesis/dataSets/Iris/iris.data'
X = np.loadtxt(path, delimiter=',', usecols=[0, 1, 2, 3])
labels = np.genfromtxt(path, delimiter=',', dtype=str, usecols=[4])

labels = list(map(convert_to_num, labels))

clusterer = COBRAS_km_clv(X, LabelQuerier(labels), budget)
clustering, intermediate_clusterings, runtimes, ml, cl = clusterer.cluster()

print(metrics.adjusted_rand_score(clustering.construct_cluster_labeling(), labels))
