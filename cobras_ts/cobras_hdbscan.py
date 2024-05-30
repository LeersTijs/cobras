import copy
import time

import numpy as np
from matplotlib import pyplot as plt

from cobras_ts.cluster import Cluster
from cobras_ts.clustering import Clustering
from cobras_ts.superinstance_kmeans import SuperInstance_kmeans
from cobras_ts.cobras import COBRAS

from sklearn.cluster import HDBSCAN, KMeans


class COBRAS_hdbscan(COBRAS):

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

        # # the split level for this initial super-instance is determined,
        # # the super-instance is split, and a new cluster is created for each of the newly created super-instances
        # initial_k = self.determine_split_level(initial_superinstance,
        #                                        copy.deepcopy(self.clustering.construct_cluster_labeling()))

        # split the super-instance and place each new super-instance in its own cluster
        # superinstances = self.split_superinstance(initial_superinstance, initial_k)
        superinstances = self.split_superinstance_using_hdbscan(initial_superinstance)
        self.clustering.clusters = []
        for si in superinstances:
            self.clustering.clusters.append(Cluster([si]))

        clv = 0
        for cluster in self.clustering.clusters:
            for si in cluster.super_instances:
                for (x, y) in self.cl:
                    if x in si.indices and y in si.indices:
                        clv += 1
        self.clv.append((len(self.cl) + len(self.ml), clv))
        self.cv.append((len(self.cl) + len(self.ml), self.count_constraint_violations()))


        # the first bottom up merging step
        # the resulting cluster is the best clustering we have so use this as first valid clustering
        self.merge_containing_clusters(copy.deepcopy(self.clustering.construct_cluster_labeling()))
        last_valid_clustering = copy.deepcopy(self.clustering)

        if self.verbose:
            self.colors = ["b", "g", "r", "c", "m", "y", "peru", "orange", "lime", "yellow"]
            print(f"#queries: {len(self.cl) + len(self.ml)}")
            clustering_labels = self.clustering.construct_cluster_labeling()
            print(clustering_labels)
            clustering_labels = list(map(lambda i: self.colors[i], clustering_labels))
            plt.scatter(self.data[:, 0], self.data[:, 1], c=clustering_labels)
            plt.show()

        queries_asked = len(self.cl) + len(self.ml)
        budget_already_in_it = False
        for b, _ in self.clv:
            if queries_asked == b:
                budget_already_in_it = True
        if not budget_already_in_it:
            clv = 0
            for cluster in self.clustering.clusters:
                for si in cluster.super_instances:
                    for (x, y) in self.cl:
                        if x in si.indices and y in si.indices:
                            clv += 1
            self.clv.append((len(self.cl) + len(self.ml), clv))
            self.cv.append((len(self.cl) + len(self.ml), self.count_constraint_violations()))

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
            # print(split_level)
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

            clv = 0
            for cluster in self.clustering.clusters:
                for si in cluster.super_instances:
                    for (x, y) in self.cl:
                        if x in si.indices and y in si.indices:
                            clv += 1
            self.clv.append((len(self.cl) + len(self.ml), clv))
            self.cv.append((len(self.cl) + len(self.ml), self.count_constraint_violations()))

            # perform the merging phase
            fully_merged = self.merge_containing_clusters(clustering_to_store)
            # if the merging phase was able to complete before the query limit was reached
            # the current clustering is a valid clustering
            if fully_merged:
                last_valid_clustering = copy.deepcopy(self.clustering)

            queries_asked = len(self.cl) + len(self.ml)
            budget_already_in_it = False
            for b, _ in self.clv:
                if queries_asked == b:
                    budget_already_in_it = True
            if not budget_already_in_it:
                clv = 0
                for cluster in self.clustering.clusters:
                    for si in cluster.super_instances:
                        for (x, y) in self.cl:
                            if x in si.indices and y in si.indices:
                                clv += 1
                self.clv.append((len(self.cl) + len(self.ml), clv))
                self.cv.append((len(self.cl) + len(self.ml), self.count_constraint_violations()))

            if self.verbose:
                self.colors = ["b", "g", "r", "c", "m", "y", "peru", "orange", "lime", "yellow"]
                queries_asked = len(self.cl) + len(self.ml)
                print(f"#queries: {queries_asked}")
                # if queries_asked == 129 or queries_asked == 36:
                clustering_labels = self.clustering.construct_cluster_labeling()
                print(clustering_labels)
                clustering_labels = list(map(lambda i: self.colors[i], clustering_labels))
                plt.scatter(self.data[:, 0], self.data[:, 1], c=clustering_labels)
                plt.show()

        # clustering procedure is finished
        # change the clustering result to the last valid clustering
        self.clustering = last_valid_clustering

        # return the correct result based on what self.store_intermediate_results contains
        if self.store_intermediate_results:
            return self.clustering, [clust for clust, _, _ in self.intermediate_results], [runtime for _, runtime, _ in
                                                                                           self.intermediate_results], self.ml, self.cl, self.clv, self.cv
        else:
            return self.clustering

    def split_superinstance_using_hdbscan(self, si):
        data_to_cluster = self.data[si.indices, :]
        km = HDBSCAN()
        km.fit(data_to_cluster)

        split_labels = km.labels_.astype(np.int32)
        while -1 in split_labels:
            split_labels = list(map(lambda x: x + 1, split_labels))

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

        si.children = training

        return training

    def split_superinstance(self, si, k):
        """
            Splits the given super-instance using k-means
        """

        data_to_cluster = self.data[si.indices, :]
        km = KMeans(k)
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

        si.children = training

        return training

    def create_superinstance(self, indices, parent=None):
        """
            Creates a super-instance of type SuperInstance_kmeans
        """

        return SuperInstance_kmeans(self.data, indices, self.train_indices, parent)
