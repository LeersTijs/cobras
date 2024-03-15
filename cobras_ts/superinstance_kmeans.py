from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import entropy

from cobras_ts.superinstance import SuperInstance


class SuperInstance_kmeans(SuperInstance):

    def __init__(self, data, indices, train_indices, parent=None):
        """
            Chooses the super-instance representative as the instance closest to the super-instance centroid
        """
        super(SuperInstance_kmeans, self).__init__(data, indices, train_indices, parent)

        self.centroid = np.mean(data[indices, :], axis=0)
        self.si_train_indices = [x for x in indices if x in train_indices]

        try:
            self.representative_idx = min(self.si_train_indices,
                                          key=lambda x: np.linalg.norm(self.data[x, :] - self.centroid))
        except:
            raise ValueError('Super instance without training instances')

    def distance_to(self, other_superinstance: SuperInstance_kmeans):
        """
            The distance between two super-instances is equal to the distance between there centroids  
        """
        return np.linalg.norm(self.centroid - other_superinstance.centroid)

    def single_link_distance_to(self, other_superinstance: SuperInstance_kmeans):
        """
            Returns the distance between the two closest instances of these two si.
        """
        instance_pairs = itertools.product(self.indices, other_superinstance.indices)
        distances = [np.linalg.norm(self.data[pair[0]] - self.data[pair[1]]) for pair in instance_pairs]
        return min(distances)

    def get_information(self, ground_truth_labels: list[int], cl: list[tuple[int, int]], ml: list[tuple[int, int]]):
        size = len(self.indices)
        if size == 1:
            max_dist = -1
            dist_distribution = [-1]
            avg_dist = -1
            var_dist = 0
            med_dist = -1
            label_distribution = np.zeros(len(set(ground_truth_labels)))
            label_distribution[ground_truth_labels[self.indices[0]]] = 1
            ent = 0
            clv = 0
        else:
            number_of_labels = len(set(ground_truth_labels))

            instance_pairs = itertools.combinations(self.indices, 2)
            dist_distribution = []
            for (idx1, idx2) in instance_pairs:
                dist = np.linalg.norm(self.data[idx1] - self.data[idx2])
                dist_distribution.append(dist)
            max_dist = max(dist_distribution)
            avg_dist = np.average(dist_distribution)
            var_dist = np.var(dist_distribution)
            med_dist = np.median(dist_distribution)

            label_distribution = np.zeros(number_of_labels)
            for idx in self.indices:
                label_distribution[ground_truth_labels[idx] - 1] += 1

            normalized_distribution = label_distribution / sum(label_distribution)
            ent = entropy(normalized_distribution)

            clv = 0
            for (idx1, idx2) in cl:
                if idx1 in self.indices and idx2 in self.indices:
                    clv += 1

        round_dec = 5
        si_info = {
            "size": size,
            "max_dist": round(max_dist, round_dec),
            "dist_distribution": dist_distribution,
            "avg_dist": round(avg_dist, round_dec),
            "var_dist": round(var_dist, round_dec),
            "med_dist": round(med_dist, round_dec),
            "clv": clv,
            "label_distribution": list(map(lambda x: int(x), label_distribution)),
            "label_entropy": round(ent, round_dec)
        }
        return si_info
