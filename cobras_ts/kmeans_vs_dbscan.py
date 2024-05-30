from warnings import simplefilter
import matplotlib as mpl
import numpy as np
# import hdbscan

from cobras_ts.cobras_hdbscan import COBRAS_hdbscan
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.querier import LabelQuerier
from experments.get_data_set import get_data_set, get_norm_data_set

from sklearn.cluster import KMeans, HDBSCAN
from sklearn import metrics
from sklearn.manifold import TSNE

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action="ignore", category=RuntimeWarning)


def kmeans_vs_hdbscan():
    X, y = get_norm_data_set("ecoli")

    km = KMeans(n_clusters=len(set(y)))
    km.fit(X)
    y_km = km.labels_.astype(np.int32)

    hd = HDBSCAN()
    hd.fit(X)
    y_hd = hd.labels_.astype(np.int32)

    print("k: ", len(set(y)))
    print("kmeans: ", metrics.adjusted_rand_score(y, y_km))
    print("hdscan: ", metrics.adjusted_rand_score(y, y_hd))

    print("using tsne")
    X_embedded = TSNE(n_components=2).fit_transform(X)
    km = KMeans(n_clusters=len(set(y)))
    km.fit(X_embedded)
    y_km = km.labels_.astype(np.int32)

    hd = HDBSCAN()
    hd.fit(X_embedded)
    y_hd = hd.labels_.astype(np.int32)

    print("kmeans: ", metrics.adjusted_rand_score(y, y_km))
    print("hdscan: ", metrics.adjusted_rand_score(y, y_hd))



def main():
    mpl.style.use("seaborn-v0_8-poster")
    X, y = get_data_set("8moons")

    # clusterer = COBRAS_kmeans(X, LabelQuerier(y), max_questions=150, verbose=True)
    # clustering, intermediate_clustering, runtimes, ml, cl, clv, cv = clusterer.cluster()

    clusterer = COBRAS_hdbscan(X, LabelQuerier(y), max_questions=150, verbose=True)
    clustering, intermediate_clustering, runtimes, ml, cl, clv, cv = clusterer.cluster()


if __name__ == "__main__":
    kmeans_vs_hdbscan()
