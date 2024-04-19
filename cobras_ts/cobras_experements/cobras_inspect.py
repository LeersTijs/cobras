import copy
import time
from enum import Enum

import numpy as np
from kneed import KneeLocator
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from cobras_ts.cluster import Cluster
from cobras_ts.clustering import Clustering
from cobras_ts.cobras_experements.nfa import NFA
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.superinstance_kmeans import SuperInstance_kmeans


class Split_estimators(Enum):
    GROUND_TRUTH = 0
    NORMAL = 1
    FULL_TREE_SEARCH = 2
    ELBOW = 3
    SILHOUETTE_ANALYSIS = 4
    CALINSKI_HARABASZ_INDEX = 5
    DAVIES_BOULDIN_INDEX = 6
    GAPSTATISTICS = 7
    X_MEANS = 8


class COBRAS_inspect(COBRAS_kmeans):

    def __init__(self, data, querier, max_questions, ground_truth_labels: list[int], verbose=True, use_nfa=False,
                 starting_heur="size",
                 split_estimator=Split_estimators.NORMAL):
        super().__init__(data, querier, max_questions)

        if len(ground_truth_labels) != len(data):
            raise ValueError("The ground truth labels should be equal size as the data")

        self.ground_truth_labels = ground_truth_labels
        self.verbose = verbose
        # self.ground_split_level = ground_split_level
        self.query_counter = 0

        self.split_estimator = split_estimator
        self.max_tested_k = 10

        if use_nfa:
            state_names = ["size", "max_dist", "avg_dist", "med_dist", "var_dist"]
            size_prob = np.array([0.0894, 0.01, 0.0572, 0.0625, 0.1519])
            max_prob = np.array([0.2076, 0.0953, 0.1034, 0.1173, 0.1408])
            avg_prob = np.array([0.4363, 0.2288, 0.1101, 0.1094, 0.145])
            med_prob = np.array([0.4242, 0.219, 0.1175, 0.1404, 0.1265])
            var_prob = np.array([0.2747, 0.1012, 0.1046, 0.0833, 0.1122])

            size_prob /= np.sum(size_prob)
            max_prob /= np.sum(max_prob)
            avg_prob /= np.sum(avg_prob)
            med_prob /= np.sum(med_prob)
            var_prob /= np.sum(var_prob)

            print(np.sum(size_prob))
            print(np.sum(max_prob))
            print(np.sum(avg_prob))
            print(np.sum(med_prob))
            print(np.sum(var_prob))

            self.nfa = NFA(np.array([size_prob, max_prob, avg_prob, med_prob, var_prob]).tolist(), state_names,
                           starting_heur)
            self.initial_select = True
            print(self.nfa.get_transition_matrix())
        else:
            self.initial_select = False
            self.nfa = None
        self.current_heur = starting_heur

        self.counter_heuristics = {"size": 0,
                                   "max_dist": 0,
                                   "avg_dist": 0,
                                   "med_dist": 0,
                                   "var_dist": 0,
                                   "nothing": 0,
                                   "total": 0}
        self.information_gain = []

        self.counter_k = {
            Split_estimators.NORMAL: 0,
            Split_estimators.FULL_TREE_SEARCH: 0,
            Split_estimators.ELBOW: 0,
            Split_estimators.SILHOUETTE_ANALYSIS: 0,
            Split_estimators.CALINSKI_HARABASZ_INDEX: 0,
            Split_estimators.DAVIES_BOULDIN_INDEX: 0,
            Split_estimators.GAPSTATISTICS: 0,
            Split_estimators.X_MEANS: 0,
            "total": 0
        }

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
        initial_k = self.determine_split_level(initial_superinstance,
                                               copy.deepcopy(self.clustering.construct_cluster_labeling()))
        # if self.ground_split_level:
        #     initial_k = initial_superinstance.get_information(self.ground_truth_labels, self.cl, self.ml)["k"]
        #     print(initial_k)
        # else:
        #     self.query_counter = 0
        #     initial_k = self.determine_split_level(initial_superinstance,
        #                                            copy.deepcopy(self.clustering.construct_cluster_labeling()))
        # split the super-instance and place each new super-instance in its own cluster
        superinstances = self.split_superinstance(initial_superinstance, initial_k)
        self.clustering.clusters = []
        for si in superinstances:
            self.clustering.clusters.append(Cluster([si]))

        if self.verbose:
            print(initial_k)
            print("##############################################################")
        self.print_info_about_all_current_sis("After initial split", False)

        # the first bottom up merging step
        # the resulting cluster is the best clustering we have so use this as first valid clustering
        self.merge_containing_clusters(copy.deepcopy(self.clustering.construct_cluster_labeling()))
        last_valid_clustering = copy.deepcopy(self.clustering)

        # if self.verbose:
        self.print_info_about_all_current_sis("after initial merge")

        # while we have not reached the max number of questions
        while len(self.ml) + len(self.cl) < self.max_questions and metrics.adjusted_rand_score(
                self.clustering.construct_cluster_labeling(), self.ground_truth_labels) != 1.0:
            # notify the querier that there is a new clustering
            # such that this new clustering can be displayed to the user
            self.querier.update_clustering(self.clustering)

            # after inspecting the clustering the user might be satisfied
            # let the querier check whether or not the clustering procedure should continue
            # note: at this time only used in the notebook queriers
            if not self.querier.continue_cluster_process():
                break

            # choose the next super-instance to split

            if self.nfa is not None and not self.initial_select:
                self.current_heur = self.nfa.random_step()
            self.initial_select = False

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
            split_level = self.determine_split_level(to_split, clustering_to_store)
            # if self.ground_split_level:
            #     split_level = to_split.get_information(self.ground_truth_labels, self.cl, self.ml)["k"]
            # else:
            #     self.query_counter = 0
            #     split_level = self.determine_split_level(to_split, clustering_to_store)
            # split the chosen super-instance
            new_super_instances = self.split_superinstance(to_split, split_level)

            if self.verbose:
                print("###################################### New iteration ######################################")
                print(split_level)
            self.print_sis_parent_with_children(to_split, new_super_instances)

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

            # if self.verbose:
            #     self.print_info_about_all_current_sis("Right after the splitting phase")

            # perform the merging phase
            fully_merged = self.merge_containing_clusters(clustering_to_store)
            # if the merging phase was able to complete before the query limit was reached
            # the current clustering is a valid clustering
            if fully_merged:
                last_valid_clustering = copy.deepcopy(self.clustering)

            # if self.verbose:
            self.print_info_about_all_current_sis("After merging")

        # clustering procedure is finished
        # change the clustering result to the last valid clustering
        self.clustering = last_valid_clustering

        # return the correct result based on what self.store_intermediate_results contains
        if self.store_intermediate_results:
            return (self.clustering, [clust for clust, _, _ in self.intermediate_results], [runtime for _, runtime, _ in
                                                                                            self.intermediate_results],
                    self.ml, self.cl, self.counter_heuristics, self.information_gain, self.counter_k)
        else:
            return self.clustering

    def split_superinstance(self, si, k):
        if self.split_estimator == Split_estimators.X_MEANS:
            return self.split_superinstance_using_x_means(si)
        else:
            return super().split_superinstance(si, k)

    def split_superinstance_using_x_means(self, si, only_calc_k=False):
        data_to_cluster = self.data[si.indices, :]
        initial_k = 2
        initial_centers = kmeans_plusplus_initializer(data_to_cluster, initial_k).initialize()

        xmeans_instance = xmeans(data_to_cluster, initial_centers, 20)
        xmeans_instance.process()
        clusters = xmeans_instance.get_clusters()

        if only_calc_k:
            return len(clusters)

        # The clusters only has the indexes that reference to the Si and not to self.data
        # Need to convert them to split_labels
        split_labels = np.zeros(len(data_to_cluster), dtype=np.int32)
        for label in range(1, len(clusters)):
            for idx in clusters[label]:
                split_labels[idx] = label

        training = []
        no_training = []

        for new_si_idx in set(split_labels):
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

    def determine_split_level(self, superinstance: SuperInstance_kmeans, clustering_to_store):
        true_k = superinstance.get_information(self.ground_truth_labels, self.cl, self.ml)["k"]

        if len(superinstance.indices) <= 2:
            return true_k

        self.counter_k[Split_estimators.ELBOW] += true_k == self.determine_split_level_elbow(superinstance, clustering_to_store)
        self.counter_k[Split_estimators.SILHOUETTE_ANALYSIS] += true_k == self.determine_split_level_with_a_scorer(superinstance, clustering_to_store, silhouette_score)
        self.counter_k[Split_estimators.CALINSKI_HARABASZ_INDEX] += true_k == self.determine_split_level_with_a_scorer(superinstance, clustering_to_store, calinski_harabasz_score)
        self.counter_k[Split_estimators.DAVIES_BOULDIN_INDEX] += true_k == self.determine_split_level_with_a_scorer(superinstance, clustering_to_store, davies_bouldin_score)
        self.counter_k[Split_estimators.NORMAL] += true_k == super().determine_split_level(superinstance, clustering_to_store, False)
        self.counter_k[Split_estimators.FULL_TREE_SEARCH] += true_k == self.full_tree_search(superinstance, clustering_to_store, use_queries=False)
        self.counter_k[Split_estimators.X_MEANS] += true_k == self.split_superinstance_using_x_means(superinstance, only_calc_k=True)
        self.counter_k["total"] += 1

        match self.split_estimator:
            case Split_estimators.GROUND_TRUTH:
                return superinstance.get_information(self.ground_truth_labels, self.cl, self.ml)["k"]
            case Split_estimators.NORMAL:
                return super().determine_split_level(superinstance, clustering_to_store)
            case Split_estimators.FULL_TREE_SEARCH:
                return self.full_tree_search(superinstance, clustering_to_store)
            case Split_estimators.ELBOW:
                k = self.determine_split_level_elbow(superinstance, clustering_to_store)
                # print(f"normal: {super().determine_split_level(superinstance, clustering_to_store)}, elbow: {k}")
                return k if k is not None else 2
            case Split_estimators.SILHOUETTE_ANALYSIS:
                k = self.determine_split_level_with_a_scorer(superinstance, clustering_to_store, silhouette_score)
                # print(f"normal: {super().determine_split_level(superinstance, clustering_to_store)}, silhouette: {k}")
                return k
            case Split_estimators.CALINSKI_HARABASZ_INDEX:
                k = self.determine_split_level_with_a_scorer(superinstance, clustering_to_store,
                                                             calinski_harabasz_score)
                # print(f"normal: {super().determine_split_level(superinstance, clustering_to_store)}, silhouette: {k}")
                return k
            case Split_estimators.DAVIES_BOULDIN_INDEX:
                k = self.determine_split_level_with_a_scorer(superinstance, clustering_to_store, davies_bouldin_score,
                                                             False)
                # print(f"normal: {super().determine_split_level(superinstance, clustering_to_store)}, silhouette: {k}")
                return k
            case Split_estimators.GAPSTATISTICS:
                pass
                # return k
            case Split_estimators.X_MEANS:
                return -1  # X_means will self determinate the k

    def search_whole_splitting_tree(self, superinstance, clustering_to_store, depth=0, debug=False, node=None,
                                    budget=np.inf, use_queries=True):
        """
                Determine the splitting level for the given super-instance using a small amount of queries

                For each query that is posed during the execution of this method the given clustering_to_store is stored as an intermediate result.
                The provided clustering_to_store should be the last valid clustering that is available

                :return: the splitting level k, flag for indication if a prediction should be done
                :rtype: int, bool
                """
        # if debug and node is None:
        #     node = Splitting_level_node(len(superinstance.indices))

        if len(self.ml) + len(self.cl) >= self.max_questions:
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
            if use_queries:
                self.ml.append((pt1, pt2))
            # if debug:
            #     Splitting_level_node(len(left_child.indices), parent=node, link="ml")  # Left node
            #     Splitting_level_node(len(right_child.indices), parent=node, link="ml")  # Right node

            if self.query_counter > budget:
                over_budget = True

            if self.store_intermediate_results and use_queries:
                self.intermediate_results.append(
                    (clustering_to_store, time.time() - self.start_time, len(self.ml) + len(self.cl)))
            split_level = 1  # if depth != 0 else 2
        else:
            if use_queries:
                self.cl.append((pt1, pt2))
            # if debug:
            #     left_node = Splitting_level_node(len(left_child.indices), parent=node, link="cl")  # Left node
            #     right_node = Splitting_level_node(len(right_child.indices), parent=node, link="cl")  # Right node
            # else:
            left_node, right_node = None, None

            if self.query_counter > budget:
                over_budget = True

            if self.store_intermediate_results and use_queries:
                self.intermediate_results.append(
                    (clustering_to_store, time.time() - self.start_time, len(self.ml) + len(self.cl)))

            if not over_budget:
                max_split = len(si.indices)
                left_split, over_budget = self.search_whole_splitting_tree(left_child, clustering_to_store,
                                                                           depth=depth + 1,
                                                                           debug=debug,
                                                                           node=left_node,
                                                                           budget=budget,
                                                                           use_queries=use_queries)
                if debug and over_budget:
                    right_split = left_split
                    right_node.return_value = left_split
                else:
                    right_split, over_budget = self.search_whole_splitting_tree(right_child, clustering_to_store,
                                                                                depth=depth + 1,
                                                                                debug=debug,
                                                                                node=right_node,
                                                                                budget=budget,
                                                                                use_queries=use_queries)
                min_split = min(max_split, left_split + right_split)
                # Make use of the return flag, to predict the right_child
                split_level = max(2, min_split)
            else:
                split_level = 2

        # if debug:
        #     node.return_value = split_level
        #     if depth == 0:
        #         print_tree(node)
        # print(f"over_budget: {over_budget}, counter: {self.query_counter}, max: {budget}")
        return split_level, over_budget

    def full_tree_search(self, superinstance, clustering_to_store,
                         depth=0, debug=False, node=None, budget=np.inf, use_queries=True):
        splitting_level, _ = self.search_whole_splitting_tree(superinstance, clustering_to_store, depth, debug, node,
                                                              budget, use_queries=use_queries)
        return max([splitting_level, 2])  # So that we never try to split a SI into 1.

    def determine_split_level_elbow(self, superinstance, clustering_to_store):
        data_to_cluster = self.data[superinstance.indices, :]
        max_split = len(superinstance.indices)

        sum_of_squared_distances = []
        range_n_clusters = [*range(2, self.max_tested_k if self.max_tested_k < max_split else max_split)]
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters)
            clusterer.fit(data_to_cluster)
            sum_of_squared_distances.append(clusterer.inertia_)

        kneedle = KneeLocator(range_n_clusters, sum_of_squared_distances, curve="convex", direction="decreasing")
        if self.verbose:
            print(f"size: {len(superinstance.indices)}")
            kneedle.plot_knee()
            plt.show()
        return kneedle.elbow

    def determine_split_level_with_a_scorer(self, superinstance, clustering_to_store, scorer, maximizer=True):
        data_to_cluster = self.data[superinstance.indices, :]
        avg_scores = []
        max_split = len(superinstance.indices)
        starting_split_level = 2
        range_n_clusters = [
            *range(starting_split_level, self.max_tested_k if self.max_tested_k < max_split else max_split)]

        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters)
            cluster_labels = clusterer.fit_predict(data_to_cluster)

            # For some reason it could be that KMeans gives every instance the same label
            # => just set the first instance to a other label
            if len(set(cluster_labels)) <= 1:
                cluster_labels[0] = 1

            score = scorer(data_to_cluster, cluster_labels)
            avg_scores.append(score)

        k = range_n_clusters[np.argmax(avg_scores) if maximizer else np.argmin(avg_scores)]
        if self.verbose:
            plt.plot(range_n_clusters, avg_scores, color="b", label="scores")
            plt.axvline(x=k, color="r", linestyle="--", label="found k")
            plt.legend()
            plt.show()

        return k

    def identify_superinstance_to_split(self):
        if len(self.clustering.clusters) == 1 and len(self.clustering.clusters[0].super_instances) == 1:
            return self.clustering.clusters[0].super_instances[0], self.clustering.clusters[0]

        superinstance_to_split = None
        max_heur = -np.inf
        originating_cluster = None

        # df = pandas.DataFrame(columns=["size", "max_dist", "avg_dist", "med_dist", "var_dist"])
        # sis = []
        # originating_clusters = []
        # df.add([2, 3.4, 4.5, 3, 7], axis="rows")

        for cluster in self.clustering.clusters:

            if cluster.is_pure:
                continue

            if cluster.is_finished:
                continue

            for superinstance in cluster.super_instances:
                if superinstance.tried_splitting:
                    continue

                if len(superinstance.indices) == 1:
                    continue

                if len(superinstance.train_indices) < 2:
                    continue

                si_info = superinstance.get_information(self.ground_truth_labels, self.cl, self.ml)
                # df.loc[len(df.index)] = [si_info["size"], si_info["max_dist"], si_info["avg_dist"], si_info["med_dist"],
                #                          si_info["var_dist"]]
                # sis.append(superinstance)
                # originating_clusters.append(cluster)

                if si_info[self.current_heur] > max_heur:
                    superinstance_to_split = superinstance
                    max_heur = si_info[self.current_heur]
                    originating_cluster = cluster

        # if len(sis) == 0:
        #     return None, None
        #
        # ranking_df = pandas.DataFrame()
        # result = [0] * len(sis)
        # result = np.zeros(len(sis), dtype=np.float64)
        # for heur in ["size", "max_dist", "avg_dist", "med_dist", "var_dist"]:
        #     column = -df[heur].values
        #     column = list(map(lambda x: [x], column))
        #     _, _, _, ranking_cmp = friedman_aligned_ranks_test(*column)
        #     result += np.array(ranking_cmp)
        #     ranking_df[heur] = ranking_cmp
        # if self.verbose:
        #     print("The ranking of each heuristic")
        #     print(ranking_df)
        #     print("the resulting ranking (just add the column together)")
        #     print(result)

        # _, _, _, result = friedman_aligned_ranks_test(*-df.values)
        # if self.verbose:
        #     print("The result if I would have placed it together")
        #     print(result)
        #
        # min_pos = np.argmin(result)
        # return sis[min_pos], originating_clusters[min_pos]

        if superinstance_to_split is None:
            return None, None

        return superinstance_to_split, originating_cluster

    def print_sis_parent_with_children(self, parent: SuperInstance_kmeans, children: list[SuperInstance_kmeans]):
        myTable = create_table(False)

        # Information Gain = entropy(parent) - [weighted average] * entropy(children)
        # https://medium.com/@ompramod9921/decision-trees-6a3c05e9cb82
        parent_info = parent.get_information(self.ground_truth_labels, cl=self.cl, ml=self.ml)
        entropy_parent = parent_info["label_entropy"]
        size_parent = parent_info["size"]

        if self.verbose:
            add_si_info_to_table(parent_info, -1, myTable,
                                 True)

        weighted_average_children = 0

        for child in children:
            child_info = child.get_information(self.ground_truth_labels, cl=self.cl, ml=self.ml)
            entropy_child = child_info["label_entropy"]
            size_child = child_info["size"]
            weighted_average_children += (size_child / size_parent) * entropy_child

            if self.verbose:
                add_si_info_to_table(child_info, -1, myTable)

        information_gain = entropy_parent - weighted_average_children
        self.information_gain.append(round(information_gain, 4))

        if self.verbose:
            myTable.title = f"The parent split into children. The IG: {information_gain}, current_heur: {self.current_heur}"
            print(myTable)

    def print_info_about_all_current_sis(self, when: str, count_correct=True):
        sis_info = []

        myTable = create_table()
        myTable.title = when

        max_entropy = 0
        max_max, max_avg, max_var, max_size, max_med = 0, 0, 0, 0, 0

        current_labeling = self.clustering.construct_cluster_labeling()

        clusters_clv = np.zeros(len(set(current_labeling)))
        for (id1, id2) in self.cl:
            if current_labeling[id1] == current_labeling[id2]:
                clusters_clv[current_labeling[id1]] += 1
        # print(clusters_clv)

        for cluster_index, cluster in enumerate(self.clustering.clusters):

            for index, si in enumerate(cluster.super_instances):

                si_info = si.get_information(self.ground_truth_labels, cl=self.cl, ml=self.ml)

                # Update the max info
                if si_info["size"] > max_size:
                    max_size = si_info["size"]
                if si_info["max_dist"] > max_max:
                    max_max = si_info["max_dist"]
                if si_info["avg_dist"] > max_avg:
                    max_avg = si_info["avg_dist"]
                if si_info["var_dist"] > max_var:
                    max_var = si_info["var_dist"]
                if si_info["label_entropy"] > max_entropy:
                    max_entropy = si_info["label_entropy"]
                if si_info["med_dist"] > max_med:
                    max_med = si_info["med_dist"]

                si_info["divider"] = index == len(cluster.super_instances) - 1
                si_info["cluster_index"] = cluster_index
                sis_info.append(si_info)

        for si_info in sis_info:
            size = si_info["size"]
            max_dist = si_info["max_dist"]
            avg_dist = si_info["avg_dist"]
            var_dist = si_info["var_dist"]
            med_dist = si_info["med_dist"]
            label_distribution = si_info["label_distribution"]
            ent = si_info["label_entropy"]

            if ent == max_entropy:
                ent = "\x1B[1;4m" + str(ent) + "\x1B[0m"

                # Count how many times a heuristic got the most impure super-instance
                if count_correct:
                    nothing_was_correct = True
                    if size == max_size:
                        self.counter_heuristics["size"] += 1
                        nothing_was_correct = False

                    if max_dist == max_max:
                        self.counter_heuristics["max_dist"] += 1
                        nothing_was_correct = False

                    if avg_dist == max_avg:
                        self.counter_heuristics["avg_dist"] += 1
                        nothing_was_correct = False

                    if var_dist == max_var:
                        self.counter_heuristics["var_dist"] += 1
                        nothing_was_correct = False

                    if med_dist == max_med:
                        self.counter_heuristics["med_dist"] += 1
                        nothing_was_correct = False

                    if nothing_was_correct:
                        self.counter_heuristics["nothing"] += 1
                    self.counter_heuristics["total"] += 1

            # Underline the max values of each colum
            variables = {
                "size": (size, max_size),
                "max_dist": (max_dist, max_max),
                "avg_dist": (avg_dist, max_avg),
                "var_dist": (var_dist, max_var),
                "med_dist": (med_dist, max_med)
            }

            for var_name, (var_value, max_value) in variables.items():
                if var_value == max_value:
                    variables[var_name] = ("\x1B[1;4m" + str(var_value) + "\x1B[0m")
                else:
                    variables[var_name] = var_value

            size, max_dist, avg_dist, var_dist, med_dist = (variables.get(key, value) for key, value in
                                                            variables.items())

            myTable.add_row(
                [si_info["cluster_index"], size, max_dist, avg_dist, med_dist, var_dist, si_info["clv"],
                 label_distribution, ent],
                divider=si_info["divider"])

        if self.verbose:
            print(myTable)
            print(self.counter_heuristics)


def create_table(use_cluster_index=True):
    names = ["Cluster index", "size", "max dist", "avg dist", "med dist", "var dist", "clv", "label distribution",
             "label entropy"]
    if not use_cluster_index:
        names.pop(0)

    return PrettyTable(names)


def add_si_info_to_table(si_info, cluster_index, table, divider=False):
    size = si_info["size"]
    max_dist = si_info["max_dist"]
    avg_dist = si_info["avg_dist"]
    var_dist = si_info["var_dist"]
    med_dist = si_info["med_dist"]

    label_distribution = si_info["label_distribution"]
    ent = si_info["label_entropy"]

    if cluster_index == -1:
        row = [size, max_dist, avg_dist, med_dist, var_dist, si_info["clv"], label_distribution, ent]
    else:
        row = [cluster_index, size, max_dist, med_dist, avg_dist, var_dist, si_info["clv"], label_distribution, ent]

    table.add_row(row, divider=divider)
