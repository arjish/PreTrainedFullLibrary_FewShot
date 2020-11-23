import random
import numpy as np
import os

def get_features_fewshot_single(nb_shot, paths, labels, nb_samples, shuffle=True):
    sampler = lambda x: random.sample(x, nb_samples) if len(x) > nb_samples else x

    files_labels = [(i, os.path.join(path, file)) \
              for i, path in zip(labels, paths) \
              for file in sampler(os.listdir(path))]

    support_ids = []
    query_ids = []
    current_label = 0
    idx = 0
    while idx < len(files_labels):
        label, _ = files_labels[idx]
        if label == current_label:
            support_ids.extend(list(range(idx, idx + nb_shot)))
            idx += nb_shot
            current_label += 1
        else:
            query_ids.append(idx)
            idx += 1

    files_labels_support = [files_labels[i] for i in support_ids]
    files_labels_query = [files_labels[i] for i in query_ids]

    if shuffle:
        random.shuffle(files_labels_support)
        random.shuffle(files_labels_query)

    labels_support = [fl[0] for fl in files_labels_support]
    files_support = [fl[1] for fl in files_labels_support]
    labels_query = [fl[0] for fl in files_labels_query]
    files_query = [fl[1] for fl in files_labels_query]

    features_support = np.array([np.load(file) for file in files_support])
    features_query = np.array([np.load(file) for file in files_query])

    return features_support, labels_support, features_query, labels_query


def get_features_fewshot_full_library(nb_shot, data_path, model_folders,
                                      sampled_labels, labels, nb_samples, shuffle=True):
# Assume every folder has at least (1+nb_shot) files (Check it before).
# That is, minimum 1 query image per class.
    sampler = lambda x: random.sample(x, nb_samples) if len(x) > nb_samples else x

    folder_0 = [os.path.join(data_path, model_folders[0], item) for item in sampled_labels]

    files_labels = [(i, os.path.join(os.path.split(p)[-1], file)) \
              for i, p in zip(labels, folder_0) \
              for file in sampler(os.listdir(p))]

    support_ids = []
    query_ids = []
    current_label = 0
    idx = 0
    while idx < len(files_labels):
        label, _ = files_labels[idx]
        if label == current_label:
            support_ids.extend(list(range(idx, idx + nb_shot)))
            idx += nb_shot
            current_label += 1
        else:
            query_ids.append(idx)
            idx += 1

    files_labels_support = [files_labels[i] for i in support_ids]
    files_labels_query = [files_labels[i] for i in query_ids]

    if shuffle:
        random.shuffle(files_labels_support)
        random.shuffle(files_labels_query)

    labels_support = [fl[0] for fl in files_labels_support]
    files_support = [fl[1] for fl in files_labels_support]
    labels_query = [fl[0] for fl in files_labels_query]
    files_query = [fl[1] for fl in files_labels_query]

    features_support_list = []
    features_query_list = []
    for model in model_folders:
        model_path = os.path.join(data_path, model)
        features_support_list.append(np.array([np.load(os.path.join(model_path, file))
                                                for file in files_support]))
        features_query_list.append(np.array([np.load(os.path.join(model_path, file))
                                                for file in files_query]))

    return features_support_list, labels_support, features_query_list, labels_query


def get_features_alldata_full_library(data_path, model_folders,
                                      class_labels, label_ids):

    folder_0 = [os.path.join(data_path, model_folders[0], item) for item in class_labels]

    files_labels = [(i, os.path.join(os.path.split(p)[-1], file)) \
                    for i, p in zip(label_ids, folder_0) \
                    for file in os.listdir(p)]


    labels = [fl[0] for fl in files_labels]
    files = [fl[1] for fl in files_labels]

    features = []
    for model in model_folders:
        model_path = os.path.join(data_path, model)
        features.append(np.array([np.load(os.path.join(model_path, file))
                                                for file in files]))
    features = np.concatenate(features, axis=-1)

    return features, labels