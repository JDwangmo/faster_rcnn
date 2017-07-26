# Faster R-CNN pytorch code example

```python
python main.py PATH_TO_DATASET
```

- see https://github.com/ShaoqingRen/faster_rcnn for the official MATLAB version
- [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn): caffe
- [faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch) : tensorflow
- [keras-frcnn](https://github.com/yhenon/keras-frcnn): keras


## Things to add/change/consider
* where to handle the image scaling. Need to scale the annotations, and also RPN filters the minimum size wrt the original image size, and not the scaled image
* should image scaling be handled in FasterRCNN class?
* properly supporting flipping
* best way to handle different parameters in RPN/FRCNN for train/eval modes
* uniformize Variables, they should be provided by the user and not processed by FasterRCNN/RPN classes
* general code cleanup, lots of torch/numpy mixture
* should I use a general config file?