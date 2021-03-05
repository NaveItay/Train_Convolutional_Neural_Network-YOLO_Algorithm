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
#### Introduction

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



