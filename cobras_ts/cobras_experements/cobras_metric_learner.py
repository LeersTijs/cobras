import copy
import time

import metric_learn

from cobras_ts.cluster import Cluster
from cobras_ts.clustering import Clustering
from cobras_ts.cobras_kmeans import COBRAS_kmeans

import numpy as np


class COBRAS_metric_learner(COBRAS_kmeans):

    def __init__(self, data: np.ndarray, querier, max_questions, train_indices=None, store_intermediate_results=True,
                 metric_learner_info: dict = None):
        self.og_data = data.copy()

        super().__init__(data, querier, max_questions, train_indices, store_intermediate_results)

        self.metric_learner_info = metric_learner_info

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
        superinstances = self.split_superinstance(initial_superinstance, initial_k)
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

        self.transform_data()

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

            self.transform_data()

        # clustering procedure is finished
        # change the clustering result to the last valid clustering
        self.clustering = last_valid_clustering

        # return the correct result based on what self.store_intermediate_results contains
        if self.store_intermediate_results:
            return self.clustering, [clust for clust, _, _ in self.intermediate_results], [runtime for _, runtime, _ in
                                                                                           self.intermediate_results], self.ml, self.cl, self.clv, self.cv
        else:
            return self.clustering

    def transform_constraints(self):
        # pairs, labels = [], []
        # for (tuples, label) in [(self.cl, "cl"), (self.ml, "ml")]:
        #     for pair in tuples:
        #         pairs.append(pair)
        #         labels.append(-1 if label == "cl" else 1)
        #
        # return pairs, labels
        pairs = self.cl + self.ml
        labels = [-1] * len(self.cl) + [1] * len(self.ml)
        return pairs, labels

    def transform_data(self):
        pairs, labels = self.transform_constraints()

        match self.metric_learner_info["algo"]:
            case "MMC":
                metric_learner = metric_learn.MMC(preprocessor=self.og_data,
                                                  init=self.metric_learner_info["init"],
                                                  diagonal=self.metric_learner_info["diagonal"])
            case "ITML":
                metric_learner = metric_learn.ITML(preprocessor=self.og_data,
                                                   prior=self.metric_learner_info["prior"])

            case _:
                raise ValueError(f'The given metric learner ({self.metric_learner_info["info"]}) is not implemented')

        metric_learner.fit(pairs, labels)
        data_transormed = metric_learner.transform(self.og_data)

        self.data = data_transormed
