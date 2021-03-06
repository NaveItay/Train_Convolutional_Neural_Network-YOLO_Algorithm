# Train Convolutional Neural Network - YOLO Algorithm

![title](/images/detect_Bobsfog.PNG)


In this project, we will talk about YoloV3 Architecture and how to train it on a custom dataset, I will explain step by step how to do it by using the Darknet framework.
<br/><br/>

###  Introduction

</head><body><ul id="l1"><li><p style="padding-left: 42pt;text-indent: -35pt;line-height: 13pt;text-align: left;">What is Object Detection?</p></li><li><p style="padding-left: 42pt;text-indent: -35pt;line-height: 13pt;text-align: left;">How does object detection work?</p></li><li><p style="padding-left: 42pt;text-indent: -35pt;line-height: 13pt;text-align: left;">YOLO - You Only Look Once</p><ul id="l2"><li><p style="padding-left: 78pt;text-indent: -35pt;line-height: 13pt;text-align: left;">YOLO v3.</p><ul id="l3"><li><p style="padding-left: 114pt;text-indent: -35pt;line-height: 12pt;text-align: left;">Network Architecture</p></li><li><p style="padding-left: 114pt;text-indent: -35pt;line-height: 13pt;text-align: left;">Feature Extractor</p></li><li><p style="padding-left: 114pt;text-indent: -35pt;text-align: left;">Feature Detector</p></li><li><p style="padding-left: 114pt;text-indent: -35pt;text-align: left;">Complete Network Architecture</p></li></ul></li></ul></li></ul></body></html>
<br/><br/>

###  How to train YOLOv3 on a custom dataset

</head><body><ul id="l1"><li><p style="padding-left: 42pt;text-indent: -35pt;line-height: 13pt;text-align: left;">Data Preparation</p></li><li><p style="padding-left: 42pt;text-indent: -35pt;line-height: 13pt;text-align: left;">Labelimg</p></li><li><p style="padding-left: 42pt;text-indent: -35pt;line-height: 13pt;text-align: left;">Getting the files ready for training</p></li><li><p style="padding-left: 42pt;text-indent: -35pt;line-height: 13pt;text-align: left;">Training the model by using Darknet</p></li><li><p style="padding-left: 42pt;text-indent: -35pt;line-height: 13pt;text-align: left;">Use your custom weights for object detection.</p></li></ul></body></html>

#
<center><h3>Introduction</h3></center>

###### What is Object Detection?
> Object detection is a technique that
> encompasses two tasks of object
> classification and object localization.
> It is a model trained to detect the presence and
> location of multiple classes of objects.
> This can be used on static images or even in
> real-time on videos.
>  
> ![title](/images/Image_003.jpg)
> Image from [machinelearningmastery](https://machinelearningmastery.com/object-recognition-with-deep-learning/)

###### How does object detection work?
> Object detection finds the object and draws a bounding box around it.
> This is a computer technology related to computer vision and image processing
> used for autonomous cars, face recognition, pedestrian detection, and more ...
> State-of-the-art algorithms are being used to detect objects e.g., R-CNN, Fast
> R-CNN, Faster R-CNN, Mask R-CNN, SSD, YOLO, etc. But we are
> specifically interested in YOLO.

###### YOLO - You Only Look Once
> YOLO is one of the most powerful object detection algorithms.
> Invented by Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi,
> so far it already has 5 different versions, we’re going to focus on YOLOv3.

###### YOLOv3
> YOLOv3 is more accurate, but a bit slower than its previous versions.
> This model includes multi-scale identification, some changes in the loss
> function, and a better feature extraction network.

###### Network Architecture
> The network architecture is made up of two main components.
>       
> * Feature Extractor
> * Feature Detector
> 
> The image is first received by the Feature Extractor which extracts
> "features tables" and then given to the Feature Detector which exports
> the processed image with the bounding boxes around the detected
> classes.

###### Feature Extractor
> In the project, I used a network with 53 convolution layers
> (Darknet-53).
>
>  ![title](/images/Image_004.jpg)
>  This image is the darknet-53 architecture taken from [YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)
>  
>








