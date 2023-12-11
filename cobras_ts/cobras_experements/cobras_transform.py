import metric_learn
import numpy as np
from sklearn.cluster import KMeans

from cobras_ts.cobras_experements.cobras_split_level import COBRAS_split_lvl
from cobras_ts.cobras_experements.superinstance_split_level import SuperInstance_split_lvl


class COBRAS_transform(COBRAS_split_lvl):

    def __init__(self, data, querier, max_questions,
                 train_indices=None, store_intermediate_results=True,
                 splitting_algo: dict = None):
        super().__init__(data=data, querier=querier,
                         max_questions=max_questions,
                         train_indices=train_indices,
                         store_intermediate_results=store_intermediate_results,
                         splitting_algo=splitting_algo)

    def split_superinstance_using_cl_ml(self, si, k):
        data_to_cluster = self.data[si.indices, :]
        pairs, labels = self.__convert_index_to_local(si.indices)

        match self.splitting_algo["algo"]:
            case "MMC":
                metric_learner = metric_learn.MMC(preprocessor=data_to_cluster,
                                                  init=self.splitting_algo["init"],
                                                  diagonal=self.splitting_algo["diagonal"])
            case "ITML":
                metric_learner = metric_learn.ITML(preprocessor=data_to_cluster,
                                                   prior=self.splitting_algo["prior"])
            case _:
                raise ValueError('the given type is not MMC or ITML')

        metric_learner.fit(pairs, labels)
        data_transformed = metric_learner.transform(data_to_cluster)

        # print(f"og shape: {self.data.shape}, tranformed shape: {data_transformed.shape}")
        # Update the data with the new transformed data
        self.data = data_transformed

        km = KMeans(n_clusters=k)
        km.fit(data_transformed)

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