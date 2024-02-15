import copy
import time

import metric_learn
import numpy as np
from sklearn.cluster import KMeans

from cobras_ts.cluster import Cluster
from cobras_ts.clustering import Clustering
from cobras_ts.cobras_experements.superinstance_split_level import SuperInstance_split_lvl
from cobras_ts.cobras_kmeans import COBRAS_kmeans


class COBRAS_incr_budget(COBRAS_kmeans):

    def __init__(self, data, querier, max_questions,
                 train_indices=None, store_intermediate_results=True,
                 splitting_algo: dict = None, min_number_of_questions=20, debug=False):
        super().__init__(data=data, querier=querier,
                         max_questions=max_questions,
                         train_indices=train_indices,
                         store_intermediate_results=store_intermediate_results)
        self.splitting_algo = splitting_algo
        self.min_number_of_questions = min_number_of_questions
        self.debug = debug

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

        # if self.debug:
        #     plot_data(self.data, None, [], [], "the initial data")

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

        if (not self.splitting_algo["algo"] == "") and (len(self.ml) + len(self.cl) >= self.min_number_of_questions):
            superinstances = self.split_superinstance_using_cl_ml(initial_superinstance, initial_k)
        else:
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
                print("about to break, because there are no SI that we can split")
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

            if (not self.splitting_algo["algo"] == "") and (
                    len(self.ml) + len(self.cl) >= self.min_number_of_questions):
                new_super_instances = self.split_superinstance_using_cl_ml(to_split, split_level)
            else:
                new_super_instances = self.split_superinstance(to_split, split_level)

            # new_super_instances = self.split_superinstance(to_split, split_level)

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

    def split_superinstance_using_cl_ml(self, si, k, clustering_to_store=None):
        match self.splitting_algo["algo"]:
            # We give the data to the metric_learner
            case "MMC":
                metric_learner = metric_learn.MMC(preprocessor=self.data,
                                                  init=self.splitting_algo["init"],
                                                  diagonal=self.splitting_algo["diagonal"])
            case "ITML":
                metric_learner = metric_learn.ITML(preprocessor=self.data,
                                                   prior=self.splitting_algo["prior"])
            case _:
                raise ValueError('the given type is not MMC or ITML')

        if metric_learner is None:
            print("help")

        # We want to transform the whole data set according to our ml and cl
        # So we transform our ml, cl to pairs and labels, if a tuple appears in the constraints than that tupple
        # will also appear in the pairs and its labels will be 1 or -1 (indicated if it is a cl or ml)
        pairs, labels = self.convert_index_to_local(self.train_indices)
        metric_learner.fit(pairs, labels)

        # Partial
        # We transform the whole data set
        data_transformed = metric_learner.transform(self.data)

        # But we only cluster on the indices that correspond to the given super instance
        km = KMeans(n_clusters=k)
        transformed_si = data_transformed[si.indices, :]
        km.fit(transformed_si)

        if self.debug:  # and (len(self.cl) + len(self.ml)) % 5 == 0:
            if clustering_to_store is not None:
                print(len(set(clustering_to_store)))
            # plot_data(X=self.data, X_transformed=data_transformed, cl=self.cl, ml=self.ml, title_str="akjamf",
            #           labeling=clustering_to_store)

        """
        # Full
        self.data = metric_learner.transform(self.data)
        km = KMeans(n_clusters=k)
        si_data = self.data[si.indices, :]
        km.fit(si_data)
        """

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

    def convert_index_to_local(self, indices):
        conversion_dict = {}
        pairs, labels = [], []
        for (tuples, label) in [(self.cl, "cl"), (self.ml, "ml")]:
            for (i, j) in tuples:
                if i in conversion_dict and j in conversion_dict:
                    pairs.append([conversion_dict[i], conversion_dict[j]])
                    labels.append(-1 if label == "cl" else 1)
                else:
                    try:
                        conversion_dict[i] = indices.index(i)
                        conversion_dict[j] = indices.index(j)
                        pairs.append([conversion_dict[i], conversion_dict[j]])
                        labels.append(-1 if label == "cl" else 1)
                    except ValueError:
                        pass

        return pairs, labels
