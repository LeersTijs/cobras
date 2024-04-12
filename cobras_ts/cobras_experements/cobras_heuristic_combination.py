import copy
import time

import numpy as np
from prettytable import PrettyTable
from sklearn import metrics

from cobras_ts.cluster import Cluster
from cobras_ts.clustering import Clustering
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.superinstance_kmeans import SuperInstance_kmeans


class COBRAS_inspect(COBRAS_kmeans):

    def __init__(self, data, querier, max_questions, ground_truth_labels: list[int], verbose=True, ground_split_level=False):
        super().__init__(data, querier, max_questions)

        if len(ground_truth_labels) != len(data):
            raise ValueError("The ground truth labels should be equal size as the data")

        self.ground_truth_labels = ground_truth_labels
        self.verbose = verbose
        self.ground_split_level = ground_split_level

        self.counter = {"size": 0,
                        "max_dist": 0,
                        "avg_dist": 0,
                        "med_dist": 0,
                        "var_dist": 0,
                        "nothing": 0,
                        "total": 0}
        self.information_gain = []

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
        if self.ground_split_level:
            initial_k = initial_superinstance.get_information(self.ground_truth_labels, self.cl, self.ml)["k"]
            print(initial_k)
        else:
            initial_k = self.determine_split_level(initial_superinstance,
                                                   copy.deepcopy(self.clustering.construct_cluster_labeling()))
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
            if self.ground_split_level:
                split_level = to_split.get_information(self.ground_truth_labels, self.cl, self.ml)["k"]
            else:
                split_level = self.determine_split_level(to_split, clustering_to_store)
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
                    self.ml, self.cl, self.counter, self.information_gain)
        else:
            return self.clustering

    def identify_superinstance_to_split(self):
        if len(self.clustering.clusters) == 1 and len(self.clustering.clusters[0].super_instances) == 1:
            return self.clustering.clusters[0].super_instances[0], self.clustering.clusters[0]

        superinstance_to_split = None
        max_heur = -np.inf
        originating_cluster = None

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

                heur = "size"
                si_info = superinstance.get_information(self.ground_truth_labels, self.cl, self.ml)
                if si_info[heur] > max_heur:
                    # if len(superinstance.indices) > max_heur:
                    superinstance_to_split = superinstance
                    max_heur = si_info[heur]
                    # max_heur = len(superinstance.indices)
                    originating_cluster = cluster

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
            myTable.title = f"The parent split into children. The IG: {information_gain}"
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
                if count_correct:
                    nothing_was_correct = True
                    if size == max_size:
                        self.counter["size"] += 1
                        nothing_was_correct = False

                    if max_dist == max_max:
                        self.counter["max_dist"] += 1
                        nothing_was_correct = False

                    if avg_dist == max_avg:
                        self.counter["avg_dist"] += 1
                        nothing_was_correct = False

                    if var_dist == max_var:
                        self.counter["var_dist"] += 1
                        nothing_was_correct = False

                    if med_dist == max_med:
                        self.counter["med_dist"] += 1
                        nothing_was_correct = False

                    if nothing_was_correct:
                        self.counter["nothing"] += 1
                    self.counter["total"] += 1

            if size == max_size:
                size = "\x1B[1;4m" + str(size) + "\x1B[0m"

            if max_dist == max_max:
                max_dist = "\x1B[1;4m" + str(max_dist) + "\x1B[0m"

            if avg_dist == max_avg:
                avg_dist = "\x1B[1;4m" + str(avg_dist) + "\x1B[0m"

            if var_dist == max_var:
                var_dist = "\x1B[1;4m" + str(var_dist) + "\x1B[0m"

            if med_dist == max_med:
                med_dist = "\x1B[1;4m" + str(med_dist) + "\x1B[0m"

            myTable.add_row(
                [si_info["cluster_index"], size, max_dist, avg_dist, med_dist, var_dist, si_info["clv"],
                 label_distribution, ent],
                divider=si_info["divider"])

        if self.verbose:
            print(myTable)
            print(self.counter)


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
