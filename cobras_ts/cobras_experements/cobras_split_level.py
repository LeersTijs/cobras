import copy

import numpy as np
from sklearn.cluster import KMeans
import time
from pptree import *

from cobras_ts.cluster import Cluster
from cobras_ts.clustering import Clustering
from cobras_ts.cobras import COBRAS
from cobras_ts.cobras_experements.superinstance_split_level import SuperInstance_split_lvl
from cobras_ts.cobras_experements.splitting_level_node import Splitting_level_node


class COBRAS_split_lvl(COBRAS):

    def __init__(self, data, querier, max_questions, train_indices=None, store_intermediate_results=True):
        super().__init__(data, querier, max_questions, train_indices, store_intermediate_results)
        self.query_counter = 0

    def split_superinstance(self, si, k):
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
                training.append(SuperInstance_split_lvl(self.data, cur_indices, self.train_indices, si))
            else:
                no_training.append((cur_indices, np.mean(self.data[cur_indices, :], axis=0)))

        for indices, centroid in no_training:
            closest_train = min(training, key=lambda x: np.linalg.norm(self.data[x.representative_idx, :] - centroid))
            closest_train.indices.extend(indices)

        si.children = training

        return training

    def create_superinstance(self, indices, parent=None):
        return SuperInstance_split_lvl(data=self.data,
                                       indices=indices,
                                       train_indices=self.train_indices,
                                       parent=parent)

    def determine_split_level(self, superinstance, clustering_to_store,
                              depth=0, debug=False, node=None, budget=np.inf):
        """
        Determine the splitting level for the given super-instance using a small amount of queries

        For each query that is posed during the execution of this method the given clustering_to_store is stored as an intermediate result.
        The provided clustering_to_store should be the last valid clustering that is available

        :return: the splitting level k
        :rtype: int
        """
        if debug and node is None:
            node = Splitting_level_node(len(superinstance.indices))

        if len(self.ml) + len(self.cl) >= self.max_questions:
            print("max_questions")
            # return a flag that indicates that we should make prediction
            return 0, False
        if len(superinstance.indices) == 1:
            return 1, False

        # need to make a 'deep copy' here, we will split this one a few times just to determine an appropriate splitting
        # level
        si = self.create_superinstance(superinstance.indices)

        # split si into 2:
        if len(si.indices) == 2:
            new_superinstances = [self.create_superinstance([si.indices[0]]),
                                  self.create_superinstance([si.indices[1]])]
        else:
            new_superinstances = self.split_superinstance(si, 2)

        if len(new_superinstances) == 1:
            # We cannot split any further along this branch
            return 1, False

        left_child, right_child = new_superinstances[0], new_superinstances[1]
        pt1 = min([left_child.representative_idx, right_child.representative_idx])
        pt2 = max([left_child.representative_idx, right_child.representative_idx])

        self.query_counter += 1
        over_budget = False
        if self.querier.query_points(pt1, pt2):
            self.ml.append((pt1, pt2))
            if debug:
                Splitting_level_node(len(left_child.indices), parent=node, link="ml")  # Left node
                Splitting_level_node(len(right_child.indices), parent=node, link="ml")  # Right node
            if self.store_intermediate_results:
                self.intermediate_results.append(
                    (clustering_to_store, time.time() - self.start_time, len(self.ml) + len(self.cl)))
            split_level = 1  # if depth != 0 else 2
            if self.query_counter > budget:
                over_budget = True
        else:
            self.cl.append((pt1, pt2))
            if debug:
                left_node = Splitting_level_node(len(left_child.indices), parent=node, link="cl")  # Left node
                right_node = Splitting_level_node(len(right_child.indices), parent=node, link="cl")  # Right node
            else:
                left_node, right_node = None, None
            if self.query_counter > budget:
                over_budget = True

            if self.store_intermediate_results:
                self.intermediate_results.append(
                    (clustering_to_store, time.time() - self.start_time, len(self.ml) + len(self.cl)))
            if not over_budget:
                max_split = len(si.indices)
                left_split, over_budget = self.determine_split_level(left_child, clustering_to_store, depth=depth + 1,
                                                                     debug=debug, node=left_node, budget=budget)
                if over_budget:
                    right_split = left_split
                    right_node.return_value = left_split
                else:
                    right_split, over_budget = self.determine_split_level(right_child, clustering_to_store,
                                                                          depth=depth + 1, debug=debug, node=right_node, budget=budget)
                min_split = min(max_split, left_split + right_split)
                # Make use of the return flag, to predict the right_child
                split_level = max(2, min_split)
            else:
                split_level = 2

        if debug:
            node.return_value = split_level
            if depth == 0:
                print_tree(node)
        print(f"over_budget: {over_budget}, counter: {self.query_counter}, max: {budget}")
        return split_level, over_budget

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
        self.query_counter = 0
        initial_k, _ = self.determine_split_level(initial_superinstance,
                                                  copy.deepcopy(self.clustering.construct_cluster_labeling()),
                                                  debug=True, budget=10)
        print(f"initial_k = {initial_k}")
        # split the super-instance and place each new super-instance in its own cluster
        superinstances = self.split_superinstance(initial_superinstance, initial_k)
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
            self.query_counter = 0
            split_level, _ = self.determine_split_level(to_split, clustering_to_store, debug=True)
            print(f"split_level = {split_level}")
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
