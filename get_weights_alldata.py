import numpy as np
from utils import get_features_alldata_full_library
import os, random
import torch
import torch.nn as nn
import argparse
import pickle

model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
               'densenet121', 'densenet161', 'densenet169', 'densenet201']
# model_names = ['resnet18', 'densenet121']

# data_folders = ['birds', 'aircraft', 'fc100',  'omniglot',  'texture',  'traffic_sign',
# 'quick_draw', 'vgg_flower', 'fungi']

data_folders = ['aircraft']

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

count_features = dict()

parser = argparse.ArgumentParser(description='Get weights for all data')

parser.add_argument('data', help='path to dataset')
parser.add_argument('--nway', default=40, type=int,
    help='number of classes')
parser.add_argument('--num_epochs', default=200, type=int,
    help='number of epochs')
parser.add_argument('--n_problems', default=50, type=int,
    help='number of test problems')
parser.add_argument('--lr', default=0.001, type=float,
    help='learning rate')
parser.add_argument('--gamma', default=0.8, type=float,
    help='constant value for L2')
parser.add_argument('--l2', action='store_true', default=False,
    help='set for L2 regularization, otherwise no regularization')

parser.add_argument('--gpu', default=0, type=int,
    help='GPU id to use.')

args = parser.parse_args()

# Device configuration
device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")


# Fully connected neural network with one hidden layer
class ClassifierNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassifierNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


def train_model(model, trainloader, criterion, optimizer,
                num_epochs=200):
    # Train the model
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        # Move tensors to the configured device
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)
            if args.l2:
                c = torch.tensor(args.gamma, device=device)
                l2_reg = torch.tensor(0., device=device)
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        l2_reg += torch.norm(param)

                loss += c * l2_reg

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print('Epoch [{}/{}],  Loss: {:.4f}'
            #     .format(epoch + 1, num_epochs, loss.item()))


def get_weights(model):
    weights_normed = None
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                weights_normed = torch.norm(param, p=1, dim=0).cpu().numpy()

    return weights_normed

def normalize(x):
    tot = sum(x)
    return [round(i/tot, 3) for i in x]

def main():
    nway = args.nway
    n_problems = args.n_problems
    num_epochs = args.num_epochs

    weightsL1_dict = {}
    for dataset in data_folders:
        print("Working on datase:", dataset)
        data_path = os.path.join(args.data, dataset, 'transferred_features_all')

        folder_0 = os.path.join(data_path, model_names[0])
        label_folders = [label \
                          for label in os.listdir(folder_0) \
                          if os.path.isdir(os.path.join(folder_0, label)) \
                          ]

        weights_normed = []
        for i in range(n_problems):
            print("\t\tProblem num:", i)
            sampled_label_folders = random.sample(label_folders, nway)

            features, labels = get_features_alldata_full_library(data_path, model_names,
                                                         sampled_label_folders, range(nway))

            train_data = []
            for i in range(features.shape[0]):
                train_data.append([features[i], labels[i]])

            input_size = features.shape[1]
            print('features.shape:', features.shape)

            model = ClassifierNetwork(input_size, nway).to(device)
            trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=200)
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            train_model(model, trainloader, criterion, optimizer, num_epochs)

            weights_normed.append(get_weights(model))
        weights_normed_mean = np.mean(weights_normed, axis=0)
        weightsL1_dict[dataset] = weights_normed_mean

    weights_dict_file = 'weightsEnsembleL1_dict_'+str(nway)+'way.pkl'
    if os.path.exists(weights_dict_file):
        with open(weights_dict_file, 'rb') as fp:
            weightsL1_dict_prev = pickle.load(fp)
        weightsL1_dict = {**weightsL1_dict, **weightsL1_dict_prev}

    with open(weights_dict_file, 'wb') as fp:
        pickle.dump(weightsL1_dict, fp)

if __name__=='__main__':
    main()
