# Few-shot Image Classification: Just Use a Libraryof Pre-trained Feature Extractors and a Simple Classifier

## Use the following links to download the data:

1. ILSVRC2012:
Register at [**ImageNet**](http://www.image-net.org/) and request for a username and an access key to download ILSRVC-2012 data set.

2. CUB-200-2011 Birds:
[**Birds**](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)

3. FGVC-Aircraft:
[**Aircraft**](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)

4. FC100:
[**FC100**](https://drive.google.com/drive/folders/1nz_ADBblmrg-qs-8zFU3v6C5WSwQnQm6)

5. Omniglot:
[**Omniglot**](https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip)

6. Texture:
[**Texture**](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)

7. Traffic Sign:
[**Traffic Sign**](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip)

8. FGCVx Fungi:
[**Fungi**](https://labs.gbif.org/fgvcx/2018/fungi_train_val.tgz)
[**Annotations**](https://labs.gbif.org/fgvcx/2018/train_val_annotations.tgz)

9. Quick Draw:
[**Quick Draw**](https://console.cloud.google.com/storage/quickdraw_dataset/full/numpy_bitmap)
- Use [gsutil](https://cloud.google.com/storage/docs/gsutil_install#install) to download the data:
```
gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.npy data/quickdraw
```

10. VGG Flower:
[**VGG Flower**](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz),
[**Labels**](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat)


## Extracting Pretrained Library Features (PyTorch):

```
python extract_pretrained_features.py <path_to_data> -f <result_folder> -b <batch_size> --gpu <gpu_ID>
```

## Few-shot training and testing:

- Single library classifier example:
```
python classifier_single.py data/aircraft --model resnet18 --nway 5 --kshot 1 --kquery 15 --num_epochs 200 --n_problems 600 --hidden_size 512 --lr 0.001 --gamma 0.2
```

- Full library classifier example:
```
python classifier_full_library.py data/aircraft --nway 20 --kshot 5 --kquery 15 --num_epochs 100 --n_problems 600 --hidden_size 512 --lr 0.0005 --gamma 0.1
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
   - --soft: Use for soft bagging when applying ensemble method, otherwise hard bagging