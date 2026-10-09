# -*- coding: utf-8 -*-

from sklearn import metrics
import numpy as np


def cluster_evaluation(clustering_input, real_y, pre_y, sample_size=300, seed=2022):
    ari_score = metrics.adjusted_rand_score(real_y, pre_y)
    ami_score = metrics.adjusted_mutual_info_score(real_y, pre_y)
    nmi_score = metrics.normalized_mutual_info_score(real_y, pre_y)
    fmi_score = metrics.fowlkes_mallows_score(real_y, pre_y)
    try:
        aws_score = metrics.silhouette_score(clustering_input, pre_y, metric="euclidean", sample_size=sample_size,
                                         random_state=seed)
    except:
        aws_score = None
    return ari_score, ami_score, nmi_score, fmi_score, aws_score


def cluster_evaluation_v2(algorithm_name, real_y, pre_y):
    ari_score = metrics.adjusted_rand_score(real_y, pre_y)
    ami_score = metrics.adjusted_mutual_info_score(real_y, pre_y)
#    h_score = metrics.homogeneity_score(real_y, pre_y)
    print("{}\n ARI:{:.4f}, AMI:{:.4f}".format(algorithm_name, ari_score, ami_score))
    return ari_score, ami_score



if __name__ == '__main__':
    real_y = [0, 0, 1, 1, 1]
    pre_y = [0, 0, 0, 1, 1]
    test = cluster_evaluation("default", real_y, pre_y)
    real_y2 = [0, 0, 1, 1, 1]
    pre_y2 = [0, 0, 0, 1, 1]
    test2 = cluster_evaluation("default2", real_y2, pre_y2)
    evaluation_name = ["a", "b", "c"]
    exp_name = ["All"]*len(evaluation_name)
    for item in zip(exp_name, evaluation_name, test, test2):
        print(item)