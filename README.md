
# HKMask for instance segment
Wang Gaihua,  Lin Jinheng,  Chen Lei,   Dai Yingying,   Zhang Tianlun

![111](https://user-images.githubusercontent.com/52806183/145698950-9bda091f-572b-43e8-942d-7993fe9c0173.png)

This is an implementation of [HKMask] on Windows10, Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. 

The repository includes:
* Training code for MS COCO/ balloon /  xBD
* Example of training on your own dataset

![Instance Segmentation Sample](assets/street.png)
We have uploaded the main file.  It is based off of the [matterport](https://github.com/matterport/Mask_RCNN). Thanks to the source code author. It's great!!!

## Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in `requirements.txt`.

### MS COCO Requirements:
To train or test on MS COCO, you'll also need:
* pycocotools (installation instructions below)
* [MS COCO Dataset](http://cocodataset.org/#home)
* Download the 5K [minival](https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0)
  and the 35K [validation-minus-minival](https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0)
  subsets. More details in the original [Faster R-CNN implementation](https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md).

If you use Docker, the code has been verified to work on
[this Docker container](https://hub.docker.com/r/waleedka/modern-deep-learning/).

### xBD Requirements:
To train or test on xBD, you'll also need:
* pycocotools (installation instructions below)
* [xBD Dataset](https://xview2.org/)
* Download the xBD Dataset



## Installation
1. Clone this repository:https://github.com/matterport/Mask_RCNN
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
4. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)
5. Replace the model file.

## Citation
Use this bibtex to cite this repository:
```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```

![1](https://user-images.githubusercontent.com/52806183/145698987-64a64ce2-1bc5-4214-ac7b-e30bbaa91803.png)
