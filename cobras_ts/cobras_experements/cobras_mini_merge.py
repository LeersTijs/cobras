import copy
import itertools
import time

import numpy as np
from sklearn.cluster import KMeans

from cobras_ts.cluster import Cluster
from cobras_ts.clustering import Clustering
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.superinstance_kmeans import SuperInstance_kmeans


def is_si_connected_by_constraint(con: tuple[int, int], si1: SuperInstance_kmeans,
                                  si2: SuperInstance_kmeans) -> bool:
    id1, id2 = con
    return (id1 in si1.indices and id2 in si2.indices) or (id2 in si1.indices and id1 in si2.indices)


def combine_ml_connected_sis(si_that_are_ml_connected: list[tuple[SuperInstance_kmeans, SuperInstance_kmeans]]) \
        -> list[set[SuperInstance_kmeans]]:
    # Code can be optimized by using the set operators. But the optimization (generated by ChatGPT 3.5)
    # is only around 1 sec faster if we call this function 1_000_000 times, so ya it does not really matter.

    result = []
    while si_that_are_ml_connected:

        si1, si2 = si_that_are_ml_connected.pop(0)

        # Look if the current pair is not already in a resulting set
        found, index = False, None
        for i, connected_set in enumerate(result):
            if si1 in connected_set or si2 in connected_set:
                connected_set.add(si1)
                connected_set.add(si2)
                found = True
                index = i
                break

        if not found:
            index = len(result)
            result.append({si1, si2})

        # Look if the current pair is in one of the next pairs
        remaining_pairs = []
        for next_pair in si_that_are_ml_connected:
            next_si1, next_si2 = next_pair
            if next_si1 in result[index] or next_si2 in result[index]:
                result[index].add(next_pair[0])
                result[index].add(next_pair[1])
            else:
                remaining_pairs.append(next_pair)
        si_that_are_ml_connected = remaining_pairs

    return result


class COBRAS_mini_merge(COBRAS_kmeans):

    def cluster(self):
        """Perform clustering

        :return: if cobras.store_intermediate_results is set to False, this method returns a single Clustering object
                 if cobras.store_intermediate_results is set to True, this method returns a tuple containing the following items:

                     - a :class:`~clustering.Clustering` object representing the resulting clustering
                     - a list of intermediate clustering labellings for each query (each item is a list of clustering labels)
                     - a list of timestamps for each query
                     - the list of must-link constraints that was queried
                     - the list of cannot-link constraints that was queried
        """
        self.start_time = time.time()

        # initially, there is only one super-instance that contains all data indices
        # (i.e. list(range(self.data.shape[0])))
        initial_superinstance = self.create_superinstance(list(range(self.data.shape[0])))

        self.ml = []
        self.cl = []

        self.clustering = Clustering([Cluster([initial_superinstance])])

        # the split level for this initial super-instance is determined,
        # the super-instance is split, and a new cluster is created for each of the newly created super-instances
        initial_k = self.determine_split_level(initial_superinstance,
                                               copy.deepcopy(self.clustering.construct_cluster_labeling()))
        # split the super-instance and place each new super-instance in its own cluster
        superinstances = self.split_superinstance_using_ml_cl(initial_superinstance, initial_k)
        self.clustering.clusters = []
        for si in superinstances:
            self.clustering.clusters.append(Cluster([si]))

        # the first bottom up merging step
        # the resulting cluster is the best clustering we have so use this as first valid clustering
        self.merge_containing_clusters(copy.deepcopy(self.clustering.construct_cluster_labeling()))
        last_valid_clustering = copy.deepcopy(self.clustering)

        # while we have not reached the max number of questions
        while len(self.ml) + len(self.cl) < self.max_questions:
            # notify the querier that there is a new clustering
            # such that this new clustering can be displayed to the user
            self.querier.update_clustering(self.clustering)

            # after inspecting the clustering the user might be satisfied
            # let the querier check whether or not the clustering procedure should continue
            # note: at this time only used in the notebook queriers
            if not self.querier.continue_cluster_process():
                break

            # choose the next super-instance to split
            to_split, originating_cluster = self.identify_superinstance_to_split()
            if to_split is None:
                break

            # clustering to store keeps the last valid clustering
            clustering_to_store = None
            if self.intermediate_results:
                clustering_to_store = self.clustering.construct_cluster_labeling()

            # remove the super-instance to split from the cluster that contains it
            originating_cluster.super_instances.remove(to_split)
            if len(originating_cluster.super_instances) == 0:
                self.clustering.clusters.remove(originating_cluster)

            # - splitting phase -
            # determine the splitlevel
            split_level = self.determine_split_level(to_split, clustering_to_store)
            # split the chosen super-instance
            new_super_instances = self.split_superinstance(to_split, split_level)

            # add the new super-instances to the clustering (each in their own cluster)
            new_clusters = self.add_new_clusters_from_split(new_super_instances)
            if not new_clusters:
                # it is possible that splitting a super-instance does not lead to a new cluster:
                # e.g. a super-instance constains 2 points, of which one is in the test set
                # in this case, the super-instance can be split into two new ones, but these will be joined
                # again immediately, as we cannot have super-instances containing only test points (these cannot be
                # queried)
                # this case handles this, we simply add the super-instance back to its originating cluster,
                # and set the already_tried flag to make sure we do not keep trying to split this superinstance
                originating_cluster.super_instances.append(to_split)
                to_split.tried_splitting = True
                to_split.children = None

                if originating_cluster not in self.clustering.clusters:
                    self.clustering.clusters.append(originating_cluster)

                continue
            else:
                self.clustering.clusters.extend(new_clusters)

            # perform the merging phase
            fully_merged = self.merge_containing_clusters(clustering_to_store)
            # if the merging phase was able to complete before the query limit was reached
            # the current clustering is a valid clustering
            if fully_merged:
                last_valid_clustering = copy.deepcopy(self.clustering)

        # clustering procedure is finished
        # change the clustering result to the last valid clustering
        self.clustering = last_valid_clustering

        # return the correct result based on what self.store_intermediate_results contains
        if self.store_intermediate_results:
            return self.clustering, [clust for clust, _, _ in self.intermediate_results], [runtime for _, runtime, _ in
                                                                                           self.intermediate_results], self.ml, self.cl
        else:
            return self.clustering

    def split_superinstance_using_ml_cl(self, si, k):
        """
            Splits the given super-instance using k-means
        """
        data_to_cluster = self.data[si.indices, :]

        # We double the splitting level
        new_k = min([2 * k, len(si.indices)])
        print(f"si: {si.indices}, k: {k}, new_k: {new_k}")
        km = KMeans(new_k)
        km.fit(data_to_cluster)

        split_labels = km.labels_.astype(np.int32)

        training = []
        no_training = []

        for new_si_idx in set(split_labels):
            # go from super instance indices to global ones
            cur_indices = [si.indices[idx] for idx, c in enumerate(split_labels) if c == new_si_idx]

            si_train_indices = [x for x in cur_indices if x in self.train_indices]
            if len(si_train_indices) != 0:
                training.append(SuperInstance_kmeans(self.data, cur_indices, self.train_indices, si))
            else:
                no_training.append((cur_indices, np.mean(self.data[cur_indices, :], axis=0)))

        for indices, centroid in no_training:
            closest_train = min(training, key=lambda x: np.linalg.norm(self.data[x.representative_idx, :] - centroid))
            closest_train.indices.extend(indices)

        if new_k == len(si.indices):  # normally this is the same as: new_k == k:
            # If this is the case we are already setting every instance of this si in its own little si
            print("normaly nver.")
            children = training
        else:
            # We now have more Si's in training than k, so we have to perform a mini merge
            children = self.mini_merge(si, k, training)
        si.children = children
        return children

    def mini_merge(self, si: SuperInstance_kmeans, split_level: int, sis: list[SuperInstance_kmeans]) \
            -> list[SuperInstance_kmeans]:
        """
        :param si: the superinstance that is the parent of sis
        :param split_level: the returned list should have a length of split_level
        :param sis: the all the superinstances that are currently created after splitting 2 * k
        """
        resulting_sis = [SuperInstance_kmeans(self.data, s.indices, self.train_indices, s.parent) for s in sis]
        print(list(map(lambda x: x.indices, resulting_sis)))

        # First look if any that are in the current list have an ML connecting them.
        print(f"ml: {self.ml}, cl: {self.cl}")

        for (x, y) in self.ml:
            # print(x, y)
            if x in si.indices and y in si.indices:
                # we know that this ml is inside the sis
                # It could still be from one si inside sis or two si inside sis
                pairs_of_si = itertools.combinations(resulting_sis, 2)
                si_that_are_ml_connected = list(
                    filter(lambda pair: is_si_connected_by_constraint((x, y), pair[0], pair[1]),
                           pairs_of_si))

                if si_that_are_ml_connected:
                    if len(si_that_are_ml_connected) > 1:
                        print("at the moment I am not dealing with this case")
                        si_that_are_ml_connected = [si_that_are_ml_connected[0]]

                    for (si1, si2) in si_that_are_ml_connected:
                        print(si1.indices, si2.indices)
                        new_si = self.merge_two_si(si1, si2, si)

                        resulting_sis.remove(si1)
                        resulting_sis.remove(si2)
                        resulting_sis.append(new_si)
                else:
                    print("get fucked")

        return resulting_sis

    def merge_two_si(self, si1: SuperInstance_kmeans, si2: SuperInstance_kmeans, si: SuperInstance_kmeans) \
            -> SuperInstance_kmeans:
        indices = si1.indices.copy()
        indices.extend(si2.indices.copy())
        return SuperInstance_kmeans(self.data, indices, self.train_indices, si)
