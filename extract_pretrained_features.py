'''
Note: Use centercrop(299) for inception and centercrop(224) for others in 'val'.
'''

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms

import os
import argparse

# model_names = ['alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn',
#                'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
#                'squeezenet1_0', 'squeezenet1_1', 'densenet121', 'densenet169', 'densenet201',
#                'densenet161', 'inception_v3', 'googlenet', 'shufflenet_v2', 'mobilenet_v2',
#                'esnext50_32x4d', 'resnext101_32x8d', 'wideresnet50_2', 'wideresnet101_2', 'mnasnet1_0']

model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
               'densenet121', 'densenet169', 'densenet201', 'densenet161']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Feature Extraction')
parser.add_argument('data', help='path to dataset')
parser.add_argument('-f', '--imageFolderName', default='all')
parser.add_argument('-b', '--batch_size', default=32, type=int,
    metavar='N',
    help='mini-batch size (default: 32), this is the total '
         'batch size of all GPUs on the current node when '
         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--gpu', default=0, type=int,
    help='GPU id to use.')

args = parser.parse_args()
data_dir = args.data
batch_size = args.batch_size

device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original_tuple and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def createFolderStructure(model_name):
    imageFolderName = args.imageFolderName
    results_path = os.path.join(args.data, 'transferred_features_'+ imageFolderName, model_name)

    data_path = os.path.join(args.data, imageFolderName)
    classFolders_list = [label \
                         for label in os.listdir(data_path) \
                         if os.path.isdir(os.path.join(data_path, label))]
    for folder_name in classFolders_list:
        if not os.path.exists(os.path.join(results_path, folder_name)):
            os.makedirs(os.path.join(results_path, folder_name))

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 224

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)

    elif model_name == "resnet34":
        """ Resnet34
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)

    elif model_name == "vgg11_bn":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)

    elif model_name == "squeezenet1_0":
        """ Squeezenet1_0
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)

    elif model_name == "densenet121":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)

    elif model_name == "densenet169":
        """ Densenet169
        """
        model_ft = models.densenet169(pretrained=use_pretrained)

    elif model_name == "densenet201":
        """ Densenet201
        """
        model_ft = models.densenet201(pretrained=use_pretrained)

    elif model_name == "densenet161":
        """ Densenet161
        """
        model_ft = models.densenet161(pretrained=use_pretrained)

    elif model_name == "inception_v3":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)

        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def extract_features(data_loader, model, model_name):
    imageFolderName = args.imageFolderName
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for input, _, image_path in data_loader:
            # print("Input size:", input.size())
            # compute output
            output_tensor = model(input.to(device))
            output_tensor = nn.AdaptiveAvgPool2d(output_size=(1, 1))(output_tensor)
            # output = output_tensor.detach().numpy()
            output = output_tensor.cpu().numpy()
            output = np.squeeze(output, axis=(2, 3))
            # print("Output shape:", output.shape)
            for i in range(output.shape[0]):
                root, image_name = os.path.split(image_path[i])
                root, folder_name = os.path.split(root)
                save_path = os.path.join(args.data, 'transferred_features_'+imageFolderName, model_name, folder_name)
                # print(save_path)
                np.save(os.path.join(save_path, image_name.split('.')[0]), output[i])

def main():
    for model_name in model_names:
        print("\t Working on model:", model_name)
        # create folders for extracted features
        createFolderStructure(model_name)

        # Initialize the model for this run
        model_ft, input_size = initialize_model(model_name, use_pretrained=True)
        # Print the model we just instantiated
        # print(model_ft)
        # Send the model to GPU
        model_ft = model_ft.to(device)

        # Data resizing and normalization
        data_transforms = {
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }


        print("Initializing Datasets and Dataloaders...")
        image_datasets = {}

        # Create validation dataset (since we are not doing any training)
        image_datasets['val'] = ImageFolderWithPaths(os.path.join(data_dir, args.imageFolderName), data_transforms['val'])
        dataloaders_dict = {
            'val': torch.utils.data.DataLoader(image_datasets['val'], shuffle=False, batch_size=batch_size, num_workers=4)}
        dataset_sizes = {'val': len(image_datasets['val'])}

        model_ft = nn.Sequential(*list(model_ft.children())[:-1])
        extract_features(dataloaders_dict['val'], model_ft, model_name)

if __name__ == '__main__':
    main()
