# Few-shot Image Classification: Just Use a Libraryof Pre-trained Feature Extractors and a Simple Classifier

## Use the following links to download the data:

1. ILSVRC2012:
Register at [**ImageNet**](http://www.image-net.org/) and request for a username and an access key to download ILSRVC-2012 data set.

2. CUB-200-2011 Birds:
[**Birds**](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)

3. FGVC-Aircraft:
[**Aircraft**](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)

4. FC100:


5. Omniglot:
[**Omniglot**](https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip)

6. Texture:

7. Traffic Sign:

8. FGCVx Fungi:

9. Quick Draw:

10. VGG Flower:


## Extracting Pretrained Library Features (PyTorch):

```
python extract_pretrained_features.py <path_to_data> -f <result_folder> -b <batch_size> --gpu <gpu_ID>
```

## Few-shot training and testing:

- Example:

```
python classifier_single.py data/aircraft --model resnet18 --nway 5 --kshot 1 --kquery 15 --num_epochs 200 --n_problems 600 --hidden_size 512 --lr 0.001 --gamma 0.2
```

## Selected arguments

- data\_path: path to the folder containing all images: `data/<dataset>`
- --model: model name for single classifier (Resnet18, Densenet121, etc.)
- --gpu: GPU ID to be used
- Hyperparameters
   - --lr: learning rate for the classifier: `0.001`
   - --kshot: number of images from each class in training set: `1`
   - --kquery: number of images from each class in test set: `15`
   - --nway: number of classes per task: `5`
   - --hidden_size: hidden state size for the classifier: `1024`
   - --num_epochs: number of training epochs: `100`
   - --n_problems: number of tasks used for testing: `600`
   - --gamma: L2 regularization constant: `0.5`
   - --linear: Use for a linear NN architecture (no hidden layer)
   - --nol2: Use to get rid of L2 regularization