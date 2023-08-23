# Learning Support and Trivial Prototypes for Interpretable Image Classification

### Pytorch Implementation for the paper "[Learning Support and Trivial Prototypes for Interpretable Image Classification](https://arxiv.org/pdf/2301.04011.pdf)" published at ICCV 2023.

<div align=center>
<img width="460" height="255" src="https://github.com/cwangrun/ST-ProtoPNet/blob/master/full/arch/intro.png"/></dev>
</div>

In this work, we make an analogy between the prototype learning from ProtoPNet and support vector learning from SVM, and propose to learn support (i.e., hard-to-learn) prototypes,
in comparison with trivial (i.e., easy-to-learn) prototypes, by forcing prototypes of different classes to locate near the classification boundary in the latent space (see figure above). 
In addition, we present the ST-ProtoPNet (see figure below) to exploit both support and trivial prototypes for complementary and interpretable image classification.

<div align=center>
<img width="460" height="245" src="https://github.com/cwangrun/ST-ProtoPNet/blob/master/full/arch/arch.png"/></dev>
</div>

  
This repository is built mainly based on publicly available code from [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet) and [TesNet](https://github.com/JackeyWang96/TesNet).


## Datasets:
1. Download [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), and [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/).
2. Crop images in CUB and Cars with the bounding-box information provided by the datasets: ./cropped/preprocess_data/cropimages.py.
3. Augment images in the training set in an offline mode, see details in: ./cropped/preprocess_data/img_aug.py, then put them in train_dir, test_dir, ...
4. Experiments using the raw full images in CUB and Dogs are also involved in this work, where the training images are augmented with online RandomAffine and RandomHorizontalFlip. 


## Train a model:
1. For cropped CUB and Cars: python ./cropped/main.py. Specifying: gpuid, backbone architecture, dataset. Note we use projection metric to compute similarity between prototypes and features in cropped CUB and Cars, as in [TesNet](https://github.com/JackeyWang96/TesNet).
2. For full CUB and Dogs: python ./full/main.py. Specifying: gpuid, backbone architecture, dataset. Note we use cosine similarity between prototypes and features for full CUB and Dogs, following [Deformable ProtoPNet](https://github.com/jdonnelly36/Deformable-ProtoPNet).

__iNaturalist Pretrained ResNet-50__.
As in [Deformable ProtoPNet](https://github.com/jdonnelly36/Deformable-ProtoPNet) and [ProtoTree](https://github.com/M-Nauta/ProtoTree), we use iNaturalist-pretrained ResNet50 when training our model on full CUB-200-2011,
since iNaturalist contains plants and animals and serves as a good source domain for recognising birds in CUB. 


## Interpretable reasoning:
1. Run local_analysis.py to find the nearest support or trivial training prototypes to a test input image, where these prototypes (as well as their source training images) will be automatically retrieved and 
the similarity maps of the test image with these training prototypes will be computed to realise the similarity-based interpretability. 


## Other experiments:
1. Measure interpretability using Intersection over Union (IoU), [Content Heatmap (CH)](https://github.com/UMBCvision/Explainable-Models-with-Consistent-Interpretations), [Outside-Inside Relevance Ratio (OIRR)](https://openaccess.thecvf.com/content_cvpr_2016/papers/Bach_Analyzing_Classifiers_Fisher_CVPR_2016_paper.pdf), and [Deletion AUC (DAUC)](https://github.com/eclique/RISE).
2. Since CH, IoU, and OIRR require object annotations, we extract the binary [bird segmentation masks](https://data.caltech.edu/records/w9d68-gec53) from full CUB-200-2011.
3. Run ./full/interpretability, providing the following arguments: trained model path, test_dir, ground-truth mask path.



## Citation:
```
@inproceedings{wang2023learning,
  title={Learning Support and Trivial Prototypes for Interpretable Image Classification},
  author={Wang, Chong and Liu, Yuyuan and Chen, Yuanhong and Liu, Fengbei and Tian, Yu and McCarthy, Davis J and Frazer, Helen and Carneiro, Gustavo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={351--363},
  year={2023}
}
```
