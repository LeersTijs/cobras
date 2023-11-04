from sklearn import metrics
from warnings import simplefilter
import json

from cobras_ts.cobras_km_clv import COBRAS_km_clv
from cobras_ts.querier.labelquerier import LabelQuerier
from get_data_set import get_data_set

simplefilter(action='ignore', category=FutureWarning)


def test_clv(dataset: str, start_budget=5, end_budget=100, jumps=5, n=10):
    results = {
        "budgets": [],
        "clv": [],
        "ari": [],
    }

    data, labels = get_data_set(dataset)

    for budget in range(start_budget, end_budget + jumps, jumps):
        print("----- budget: {} -----".format(budget))
        sum_clv = 0
        sum_ari = 0
        for _ in range(n):
            clusterer = COBRAS_km_clv(data, LabelQuerier(labels), budget)
            clustering, intermediate_clusterings, runtimes, ml, cl = clusterer.cluster()

            for cluster in clustering.clusters:
                for si in cluster.super_instances:
                    sum_clv += si.calculate_clv(cl)
                    # print("clv: {}, id: {}".format(clv, si.i))
                    # sum_clv += clv

            sum_ari += metrics.adjusted_rand_score(clustering.construct_cluster_labeling(), labels)

        avg_ari = sum_ari / n
        avg_clv = sum_clv / n
        results["budgets"].append(budget)
        results["clv"].append(avg_clv)
        results["ari"].append(avg_ari)
        print("sum ari: {}, avg: {}".format(sum_ari, avg_ari))
        print("sum clv: {}, avg: {}".format(sum_clv, avg_clv))

    return results


if __name__ == "__main__":
    names = ["iris", "wine", "ionosphere", "glass", "yeast"]
    results = dict()
    for name in names:
        print("############### {} ###############".format(name))
        result = test_clv(dataset=name,
                          start_budget=5,
                          end_budget=100,
                          jumps=5,
                          n=10)
        # print(result)
        results[name] = result

    with open('result.json', 'w') as f:
        json.dump(results, f)
