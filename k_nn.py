from scipy.spatial.distance import pdist, squareform
import numpy as np
import time
import pickle

from datasets.satimage2 import Satimage2Dataset
from datasets.cifar10 import Cifar10
from datasets.mammography import MammographyDataset
from enums.dataset_enum import CurrentDataset
from utils import metrics
from models.resnet_feature_extractor import get_features_with_resnet
import config


class KNN(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.cur_dataset = cfg.CUR_DATASET

    def get_data(self):
        if self.cur_dataset == CurrentDataset.cifar10.value:
            dataset = Cifar10(anomaly_class='airplane')
        elif self.cur_dataset == CurrentDataset.satimage2.value:
            dataset = Satimage2Dataset()
        elif self.cur_dataset == CurrentDataset.mammography.value:
            dataset = MammographyDataset()
        else:
            raise Exception
        self.dataset_len = len(dataset)
        if self.cfg.USE_RESNET_AS_FEATURES_EXTRACTOR:
            self.vectors = get_features_with_resnet()
        else:
            self.vectors = self.normalize_vectors(dataset.vectors).astype(np.float32)
        self.labels = dataset.labels

    @staticmethod
    def normalize_vectors(v, norm=2, axis=-1):
        l2 = np.atleast_1d(np.linalg.norm(v, norm, axis))
        l2[l2 == 0] = 1
        return v / np.expand_dims(l2, axis)

    def get_distances(self, calculate_in_loop=config.CALCULATE_DISTS_IN_LOOP, load_saved_dists=config.LOAD_SAVED_DISTS):
        if load_saved_dists:
            with open(self.cfg.CIFAR_DISTS_PICKLE_PATH, 'rb') as f:
                sorted_dists = pickle.load(f)
        else:
            if calculate_in_loop:
                sorted_dists = np.zeros((self.dataset_len, self.dataset_len - 1), dtype=np.float32)
                for i, vec in enumerate(self.vectors):
                    if i % 100 == 0:
                        print(f'calculated dists for {i}/{self.dataset_len} vectors')
                    sorted_dists[i] = np.sort(np.linalg.norm(self.vectors - vec, 2, -1))[1:]
                with open(self.cfg.CIFAR_DISTS_PICKLE_PATH, 'wb') as f:
                    pickle.dump(sorted_dists, f)
            else:
                dists_vector = pdist(self.vectors)
                square_dists = squareform(dists_vector)
                sorted_dists = np.sort(square_dists, 1)[:, 1:]
        return sorted_dists

    @staticmethod
    def get_anomaly_scores(sorted_dists, k=1):
        predictions = sorted_dists[:, k - 1]
        return predictions

    def get_best_k_by_AP(self, sorted_dists):
        best_ap_score = -1
        best_k_info = {}

        for k in self.cfg.K_LIST:
            anomaly_scores = self.get_anomaly_scores(sorted_dists, k=k)
            ids_to_sort = np.argsort(anomaly_scores)[::-1]
            predictions = anomaly_scores[ids_to_sort]
            labels = self.labels[ids_to_sort]

            tp = np.cumsum(labels)
            precision = tp / (np.arange(self.dataset_len) + 1)
            recall = tp / sum(labels == 1)

            ap_score = metrics.average_precision_score(precision, recall)
            print(f'k: {k}, calculated AP: {ap_score}')

            if ap_score > best_ap_score:
                best_ap_score = ap_score
                best_k_info['best_ap_score'] = ap_score
                best_k_info['best_k'] = k
                best_k_info['best_k_precision'] = precision
                best_k_info['best_k_recall'] = recall
                best_k_info['best_k_labels'] = labels
                best_k_info['best_k_scores'] = predictions
        return best_k_info

    def run(self):
        self.get_data()
        sorted_dists = self.get_distances()

        best_k_info = self.get_best_k_by_AP(sorted_dists)
        print(f'best k: {best_k_info["best_k"]}, AP: {best_k_info["best_ap_score"]}')

        p, r, t = metrics.precision_recall_curve(best_k_info["best_k_labels"],
                                                 best_k_info["best_k_scores"],
                                                 best_k_info['best_k_precision'],
                                                 best_k_info['best_k_recall'])
        f1_score = metrics.f1_score(p, r)
        best_f1_score_idx = np.argmax(f1_score)
        best_f1_score = f1_score[best_f1_score_idx]
        print(f'best F1-score: {best_f1_score}')

        best_thr = t[best_f1_score_idx - 1]
        fin_prediction = np.zeros(self.dataset_len)
        fin_prediction[self.get_anomaly_scores(sorted_dists, k=best_k_info["best_k"]) > best_thr] = 1

        conf_matrix_for_best_thr = metrics.confusion_matrix(self.labels, fin_prediction)
        print(f'Confusion matrix (tn, fp, fn, tp): {np.concatenate(conf_matrix_for_best_thr)}')


if __name__ == '__main__':
    knn = KNN(config)
    start_time = time.time()
    knn.run()
    print(f'Total time: {time.time() - start_time} s.')
