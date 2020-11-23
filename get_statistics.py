import numpy as np
import os
import argparse
import pickle
from sklearn.metrics import jaccard_score
from scipy.stats.stats import pearsonr
from collections import defaultdict


model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
               'densenet121', 'densenet161', 'densenet169', 'densenet201']

data_folders = ['birds', 'aircraft', 'fc100',  'omniglot',  'texture',
                'traffic_sign', 'quick_draw', 'vgg_flower', 'fungi']

features_dim_map = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
    'densenet121': 1024,
    'densenet161': 2208,
    'densenet169': 1664,
    'densenet201': 1920
}

star_end_idx = {}
start = 0
for k, v in features_dim_map.items():
    end = start + v
    star_end_idx[k] = (start, end)
    start = end

parser = argparse.ArgumentParser(description='Get Jaccard Index')

parser.add_argument('--nway', default=40, type=int,
    help='number of classes')
parser.add_argument('--kshot', default=1, type=int,
    help='number of shots (support images per class)')
parser.add_argument('--nol2', action='store_true', default=False,
    help='set for No L2 regularization, otherwise use L2')
parser.add_argument('--gamma', default=0.8, type=float,
    help='constant value for L2')

args = parser.parse_args()
nway = args.nway
kshot = args.kshot

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    if  len(s1.union(s2)) ==  0:
        return 0
    else:
        return round(len(s1.intersection(s2)) / len(s1.union(s2)), 3)

def jaccard_index_null (list1, list2, n_features):
    n1 = len(list1)
    n2 = len(list2)
    # assert len(s1) == len(s2), "Lengths of two sets are not same"

    term = (n1 * n2)/n_features
    return round(term / (n1 + n2 - term), 3)

weights_alldata_pkl = 'weightsEnsembleL1_dict_'+str(nway)+'way.pkl'
if os.path.exists(weights_alldata_pkl):
    with open(weights_alldata_pkl, 'rb') as fp:
        weightsL1_dict = pickle.load(fp)
else:
    print('weights dictionary for all data missing...')

def get_jaccard_among_datasets():
    n_datasets = len(data_folders)
    jaccard_scores = []

    fp = open('jaccard_scores_ensemble_'+str(args.nway)+'way.txt', 'w')
    n_top_features = int(sum(list(features_dim_map.values())) * 0.2)
    for idx1 in range(n_datasets-1):
        for idx2 in range(idx1+1, n_datasets):
            top_features_1 = np.argsort(weightsL1_dict[data_folders[idx1]])[::-1][:n_top_features]
            top_features_2 = np.argsort(weightsL1_dict[data_folders[idx2]])[::-1][:n_top_features]
            score = jaccard_similarity(top_features_1, top_features_2)
            jaccard_scores.append(score)
            fp.write( data_folders[idx1] + ", " + data_folders[idx2]
                     + ": " + str(score) + "\n")

    fp.close()
    with open('jaccard_scores_ensemble_'+str(args.nway)+'way.pkl', 'wb') as fp:
        pickle.dump(jaccard_scores, fp)

def get_pearson_coeff_among_datasets():
    n_datasets = len(data_folders)
    pearson_scores = []

    fp = open('pearson_scores_ensemble_' + str(args.nway) + 'way.txt', 'w')
    for idx1 in range(n_datasets - 1):
        for idx2 in range(idx1 + 1, n_datasets):
            features_1 = weightsL1_dict[data_folders[idx1]]
            features_2 = weightsL1_dict[data_folders[idx2]]
            score = [round(s, 4) for s in pearsonr(features_1, features_2)]

            pearson_scores.append(score)
            fp.write(data_folders[idx1] + ", " + data_folders[idx2]
                     + ": " + str(score) + "\n")
    fp.close()

    with open('pearson_scores_ensemble_' + str(args.nway) + 'way.pkl', 'wb') as fp:
        pickle.dump(pearson_scores, fp)

def get_jaccard_among_datasets_by_backbones():
    n_datasets = len(data_folders)
    jaccard_scores = defaultdict(list)

    fp = open('jaccard_scores_ensemble_'+str(args.nway)+'way_by_backbones.txt', 'w')
    fp.write("Jaccard indices broken down by backbones(on top 20% features, " + str(args.nway)
             + "  way) on all available images between pairs of datasets: \n\n")
    n_top_features = int(sum(list(features_dim_map.values())) * 0.2)
    for idx1 in range(n_datasets-1):
        for idx2 in range(idx1+1, n_datasets):
            fp.write(data_folders[idx1] + ", " + data_folders[idx2] + ": "  + "\n")
            top_features_1 = np.argsort(weightsL1_dict[data_folders[idx1]])[::-1][:n_top_features]
            top_features_2 = np.argsort(weightsL1_dict[data_folders[idx2]])[::-1][:n_top_features]
            #The order of the backbones have to be fixed!!
            for model_idx, backbone in enumerate(model_names):
                features_dataset_1 = [x for x in top_features_1 if star_end_idx[backbone][0]
                                      <= x < star_end_idx[backbone][1]]
                features_dataset_2 = [x for x in top_features_2 if star_end_idx[backbone][0]
                                      <= x < star_end_idx[backbone][1]]
                score = jaccard_similarity(features_dataset_1, features_dataset_2)
                jaccard_scores[backbone].append(score)  # order of datasets is important
                fp.write("\t " + backbone + ": " + str(score) + "\n")
            fp.write("\n")

    fp.close()
    with open('jaccard_scores_ensemble_'+str(args.nway)+'way_by_backbones.pkl', 'wb') as fp:
        pickle.dump(jaccard_scores, fp)


def get_fraction_by_backbones():
    n_datasets = len(data_folders)

    fp = open('fraction_top_features_' + str(args.nway) + 'way_by_backbones.txt', 'w')
    fp.write("Fraction of features from every backbone contributing to top 20% total features (" + str(args.nway)
             + "  way) on all available images in each dataset: \n\n")
    n_top_features = int(sum(list(features_dim_map.values())) * 0.2)
    for idx1 in range(n_datasets):
        fp.write(data_folders[idx1] + ": "  + "\n")
        top_features = np.argsort(weightsL1_dict[data_folders[idx1]])[::-1][:n_top_features]
        #The order of the backbones have to be fixed!!
        for model_idx, backbone in enumerate(model_names):
            features_dataset = [x for x in top_features if star_end_idx[backbone][0]
                                  <= x < star_end_idx[backbone][1]]
            # fraction_dataset = round(len(features_dataset)/n_top_features * 100, 2)
            fraction_dataset = round(len(features_dataset) / features_dim_map[backbone] * 100, 2)

            fp.write("\t " + backbone + ": " + str(fraction_dataset) + "%\n")
        fp.write("\n")

    fp.close()

def get_jaccard_fewVsAll():
    if args.nol2:
        weights_fewshot_pkl = 'weightsEnsembleL1_dict_' + str(kshot) + 'shot' + str(nway) + 'way_noL2.pkl'
        txt_file = 'jaccard_scores_ensemble_fewVsAll_'+str(kshot) + 'shot' + str(args.nway) + 'way_noL2.txt'
        pkl_file = 'jaccard_scores_ensemble_fewVsAll_'+str(kshot) + 'shot' + str(args.nway) + 'way_noL2.pkl'
    else:
        weights_fewshot_pkl = 'weightsEnsembleL1_dict_' + str(kshot)\
                              + 'shot' + str(nway) + 'way_'+str(args.gamma) + 'L2.pkl'
        txt_file = 'jaccard_scores_ensemble_fewVsAll_' + str(kshot)\
                   + 'shot' + str(args.nway) + 'way_' + str(args.gamma) + 'L2.txt'
        pkl_file = 'jaccard_scores_ensemble_fewVsAll_'+ str(kshot)\
                   + 'shot' + str(args.nway) + 'way_' + str(args.gamma) + 'L2.pkl'

    if os.path.exists(weights_fewshot_pkl):
        with open(weights_fewshot_pkl, 'rb') as fp:
            weightsL1_fewshot_dict = pickle.load(fp)
    else:
        print('weights dictionary for fewshot missing...')
    n_datasets = len(data_folders)
    jaccard_scores = []

    fp = open(txt_file, 'w')
    n_top_features = int(sum(list(features_dim_map.values())) * 0.2)
    for idx in range(n_datasets):
        top_features_1 = np.argsort(weightsL1_dict[data_folders[idx]])[::-1][:n_top_features]
        top_features_2 = np.argsort(weightsL1_fewshot_dict[data_folders[idx]])[::-1][:n_top_features]
        score = jaccard_similarity(top_features_1, top_features_2)
        jaccard_scores.append(score)
        fp.write(data_folders[idx] + ": " + str(score) + "\n")

    fp.close()
    with open(pkl_file, 'wb') as fp:
        pickle.dump(jaccard_scores, fp)

def get_pearson_coeff_fewVsAll():
    if args.nol2:
        weights_fewshot_pkl = 'weightsEnsembleL1_dict_' + str(kshot) + 'shot' + str(nway) + 'way_noL2.pkl'
        txt_file = 'pearson_scores_ensemble_fewVsAll_'+str(kshot) + 'shot' + str(args.nway) + 'way_noL2.txt'
        pkl_file = 'pearson_scores_ensemble_fewVsAll_'+str(kshot) + 'shot' + str(args.nway) + 'way_noL2.pkl'
    else:
        weights_fewshot_pkl = 'weightsEnsembleL1_dict_' + str(kshot)\
                              + 'shot' + str(nway) + 'way_' + str(args.gamma) + 'L2.pkl'
        txt_file = 'pearson_scores_ensemble_fewVsAll_' + str(kshot)\
                   + 'shot' + str(args.nway) + 'way_' + str(args.gamma) + 'L2.txt'
        pkl_file = 'pearson_scores_ensemble_fewVsAll_' + str(kshot)\
                   + 'shot' + str(args.nway) + 'way_' + str(args.gamma) + 'L2.pkl'

    if os.path.exists(weights_fewshot_pkl):
        with open(weights_fewshot_pkl, 'rb') as fp:
            weightsL1_fewshot_dict = pickle.load(fp)
    else:
        print('weights dictionary for fewshot missing...')
    n_datasets = len(data_folders)
    pearson_scores = []

    fp = open(txt_file, 'w')
    for idx in range(n_datasets):
        features_1 = weightsL1_dict[data_folders[idx]]
        features_2 = weightsL1_fewshot_dict[data_folders[idx]]
        score = [round(s, 4) for s in pearsonr(features_1, features_2)]

        pearson_scores.append(score)
        fp.write(data_folders[idx] + ": " + str(score) + "\n")
    fp.close()

    with open(pkl_file, 'wb') as fp:
        pickle.dump(pearson_scores, fp)

def main():
    get_pearson_coeff_among_datasets()
    get_jaccard_among_datasets()
    get_jaccard_among_datasets_by_backbones()
    get_fraction_by_backbones()

    get_pearson_coeff_fewVsAll()
    get_jaccard_fewVsAll()

if __name__=='__main__':
    main()