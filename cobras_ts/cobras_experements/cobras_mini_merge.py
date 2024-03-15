import copy
import itertools
import math
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from cobras_ts.cluster import Cluster
from cobras_ts.clustering import Clustering
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.superinstance_kmeans import SuperInstance_kmeans
from experments.metric_learner_tests import plot_lines


class COBRAS_mini_merge(COBRAS_kmeans):

    def __init__(self, data, querier, max_questions, train_indices=None, store_intermediate_results=True, verbose=True, n=2):
        super().__init__(data, querier, max_questions, train_indices, store_intermediate_results)
        self.prev_iterations = None
        self.verbose = verbose
        self.n = n

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
            # new_super_instances = self.split_superinstance(to_split, split_level)
            new_super_instances = self.split_superinstance_using_ml_cl(to_split, split_level)

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
        new_k = min([self.n * k, len(si.indices)])
        if self.verbose:
            print(f"k: {k}, new_k: {new_k}")
        km = KMeans(new_k)
        km.fit(data_to_cluster)

        # The normal splitting stuf: It puts all the clusters from k-means into each own super-instance
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

        # The mini-merge
        if new_k == len(si.indices):  # normally this is the same as: new_k == k:
            # If this is the case we are already setting every instance of this si in its own little si
            if self.verbose:
                print("I am here")
            children = training
        else:
            # We now have more Si's in training than k, so we have to perform a mini merge
            children = self.mini_merge(si, k, training)

        si.children = children
        return children

    def mini_merge(self, parent: SuperInstance_kmeans, split_level: int, sis: list[SuperInstance_kmeans]) \
            -> list[SuperInstance_kmeans]:
        """
        :param parent: the superinstance that is the parent of sis
        :param split_level: the returned list should have a length of split_level
        :param sis: the all the superinstances that are currently created after splitting 2 * k
        """

        resulting_sis = [SuperInstance_kmeans(self.data, s.indices, self.train_indices, s.parent) for s in sis]
        if self.verbose:
            print("inside mini_merge")
            self.remember_sis(resulting_sis, f"Goal split_level: {split_level}")
        # print("og sis:")
        # print(list(map(lambda x: x.indices, resulting_sis)))

        # First look if any that are in the current list have an ML connecting them.
        # print(f"ml: {self.ml}, cl: {self.cl}")

        # We merge every ML together, it could be that there are multiple ml that are connecting different si
        # So we first filter out any pairs that are not ml linked.
        # This could result in pairs like this: [(a, b), (b, c)]
        # So we merge them into a set of si that are al reachable with mls [{a, b, c}]
        # And we merge them.
        # TODO: What if there was a cl between a, c? Deal with this case (now were do not take that into account)
        pairs_of_si = itertools.combinations(resulting_sis, 2)
        sis_that_are_ml_connected = list(
            filter(lambda pair: is_si_connected_by_constraint(self.ml, pair[0], pair[1]), pairs_of_si))
        combined_sis = combine_ml_connected_sis(sis_that_are_ml_connected)
        for sis_ml_connected in combined_sis:
            new_si = self.merge_sis(sis_ml_connected, parent)
            for si in sis_ml_connected:
                try:
                    resulting_sis.remove(si)
                except ValueError:
                    print("Tried to remove an si that is not in the sis some weird shit: skip it :)")
            resulting_sis.append(new_si)

        if self.verbose:
            self.remember_sis(resulting_sis, "Merged all the ML")

        # self.plot_sis(resulting_sis, f"Merged all the must-links")

        found_pair_to_merge = True
        while len(resulting_sis) > split_level and found_pair_to_merge:
            found_pair_to_merge = False
            pairs_of_si = itertools.combinations(resulting_sis, 2)

            # At the moment the distance between the centroids is used
            # TODO: Update this distance to single link (need to create a new function for this)
            # pairs_of_si = list(map(lambda pair_si: (pair_si, pair_si[0].distance_to(pair_si[1])), pairs_of_si))
            pairs_of_si = list(map(lambda pair_si: (pair_si, pair_si[0].single_link_distance_to(pair_si[1])), pairs_of_si))

            pairs_of_si.sort(key=lambda pair_si: pair_si[1])

            # for (pair, dist) in pairs_of_si:
            #     print(pair[0].indices, pair[1].indices, dist)

            for (pair, dist) in pairs_of_si:
                if not is_si_connected_by_constraint(self.cl, pair[0], pair[1]):
                    new_si = self.merge_sis({pair[0], pair[1]}, parent)
                    resulting_sis.remove(pair[0])
                    resulting_sis.remove(pair[1])

                    resulting_sis.append(new_si)
                    found_pair_to_merge = True
                    break

            if self.verbose:
                self.remember_sis(resulting_sis, f"found merge?: {found_pair_to_merge}")
                # self.plot_sis(resulting_sis, f"current_split_level: {len(resulting_sis)}")

        if not found_pair_to_merge and self.verbose:
            print("We did not find a pair of si that are not connected by a ml")
            print(f"so we did not get the splitting_level of {split_level}, but got: {len(resulting_sis)}")

        if self.verbose:
            # print("resulting_sis")
            # print(list(map(lambda x: x.indices, resulting_sis)))
            self.plot_sis()

        return resulting_sis

    def merge_sis(self, sis: set[SuperInstance_kmeans], parent: SuperInstance_kmeans) -> SuperInstance_kmeans:
        indices = []
        for si in sis:
            indices.extend(si.indices.copy())
        return SuperInstance_kmeans(self.data, indices, self.train_indices, parent)

    def remember_sis(self, sis: list[SuperInstance_kmeans], title=""):
        if self.prev_iterations is None:
            self.prev_iterations = []

        current_labeling = np.zeros(self.data.shape[0])
        for si, label in zip(sis, range(len(sis))):
            current_labeling[si.indices] = label
        self.prev_iterations.append((current_labeling, title))

    def plot_sis(self):
        colormap = plt.cm.Paired

        number_of_iterations = len(self.prev_iterations)
        if number_of_iterations < 4:
            l = {1: (1, 1), 2: (1, 2), 3: (1, 3)}
            nrows, ncols = l[number_of_iterations]
        else:
            ncols = math.ceil(math.sqrt(number_of_iterations))
            nrows = math.ceil(number_of_iterations / ncols)

        fig, ax = plt.subplots(nrows, ncols)

        if nrows == 1:
            for i in range(ncols):
                labeling, title = self.prev_iterations[i]
                ax[i].set_title(title)
                ax[i].scatter(self.data[:, 0], self.data[:, 1], c=labeling, cmap=colormap)
                plot_lines(ax[i], self.data, self.cl, "r")
                plot_lines(ax[i], self.data, self.ml, "g")
        else:
            for i in range(nrows):
                for j in range(ncols):
                    index = i * ncols + j
                    if index >= len(self.prev_iterations):
                        break
                    labeling, title = self.prev_iterations[index]
                    ax[i][j].set_title(title)
                    ax[i][j].scatter(self.data[:, 0], self.data[:, 1], c=labeling, cmap=colormap)
                    plot_lines(ax[i][j], self.data, self.cl, "r")
                    plot_lines(ax[i][j], self.data, self.ml, "g")

        # # Converting sis to labeling
        # labeling = np.zeros(self.data.shape[0])
        # for si, label in zip(sis, range(len(sis))):
        #     labeling[si.indices] = label
        #
        # ax.set_title(title)
        # ax.scatter(self.data[:, 0], self.data[:, 1], c=labeling, cmap=colormap)
        # plot_lines(ax, self.data, self.cl, "r")
        # plot_lines(ax, self.data, self.ml, "g")
        plt.show()




def is_si_connected_by_constraint(cons: list[tuple[int, int]], si1: SuperInstance_kmeans, si2: SuperInstance_kmeans) \
        -> bool:
    for con in cons:
        id1, id2 = con
        if (id1 in si1.indices and id2 in si2.indices) or (id2 in si1.indices and id1 in si2.indices):
            return True
    return False


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
