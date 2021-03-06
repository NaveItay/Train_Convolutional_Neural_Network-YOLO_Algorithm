# Train Convolutional Neural Network - YOLO Algorithm

![title](/images/detect_Bobsfog.PNG)


In this project, I will talk about YoloV3 Architecture and how to train it on a custom dataset, I will explain step by step how to do it by using the Darknet framework.
<br/><br/>

###  Introduction

</head><body><ul id="l1"><li><p style="padding-left: 42pt;text-indent: -35pt;line-height: 13pt;text-align: left;">What is Object Detection?</p></li><li><p style="padding-left: 42pt;text-indent: -35pt;line-height: 13pt;text-align: left;">How does object detection work?</p></li><li><p style="padding-left: 42pt;text-indent: -35pt;line-height: 13pt;text-align: left;">YOLO - You Only Look Once</p><ul id="l2"><li><p style="padding-left: 78pt;text-indent: -35pt;line-height: 13pt;text-align: left;">YOLO v3.</p><ul id="l3"><li><p style="padding-left: 114pt;text-indent: -35pt;line-height: 12pt;text-align: left;">Network Architecture</p></li><li><p style="padding-left: 114pt;text-indent: -35pt;line-height: 13pt;text-align: left;">Feature Extractor</p></li><li><p style="padding-left: 114pt;text-indent: -35pt;text-align: left;">Feature Detector</p></li><li><p style="padding-left: 114pt;text-indent: -35pt;text-align: left;">Complete Network Architecture</p></li></ul></li></ul></li></ul></body></html>
<br/><br/>

###  How to train YOLOv3 on a custom dataset

</head><body><ul id="l1"><li><p style="padding-left: 42pt;text-indent: -35pt;line-height: 13pt;text-align: left;">Data Preparation</p></li><li><p style="padding-left: 42pt;text-indent: -35pt;line-height: 13pt;text-align: left;">Labeling</p></li><li><p style="padding-left: 42pt;text-indent: -35pt;line-height: 13pt;text-align: left;">Getting the files ready for training</p></li><li><p style="padding-left: 42pt;text-indent: -35pt;line-height: 13pt;text-align: left;">Training the model by using Darknet</p></li><li><p style="padding-left: 42pt;text-indent: -35pt;line-height: 13pt;text-align: left;">Use your custom weights for object detection.</p></li></ul></body></html>

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
>     
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
>  ![title](/images/Feature_Extractor_img1.PNG)
>       
>  This image is the darknet-53 architecture taken from [YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)
>       
>     
>     
>  This CNN is built with consecutive 3x3 and 1x1 convolution layers
>  followed by a skip connection.
>  The 53 layers of the Darknet add another 53 layers to the detection
>  head, meaning the basic architecture of YOLOv3 contains 106 layers
>  making it a relatively larger architecture than the previous versions,
>  although the processing time is a bit slower but improves the accuracy
>  at the same time.
>       
>  In our case, we would like to detect the classes with the locations, so in
> the extractor, there is a detection head.
> The detection head is a multi-scale detection hence, we would need to
> extract features at multiple scales as well.
>       
> YOLOv3 extracts three feature vectors - (52x52), (26x26), and (13x13)
> while the tiny version extracts only (26x26) and (13x13).
> 
> **(52x52 for smaller objects, 26x26 for medium objects, and 13x13
> would be used for the larger objects).**
>     
> ![title](/images/Feature_Extractor_img2.PNG)
>       
> Multi-scale Feature Extractor for a 416x416 image
> 
> This matrix called a “grid” and assigns anchor boxes to each cell of
> the grid. In other words, anchor boxes anchor to the grid cells, and they
> share the same centroid.

###### Feature Detector
> The Feature Detector eventual output of a fully convolutional network
> is done by applying 1x1 detection kernels on feature maps of three
> different sizes at three different places. The shape of the kernel is
> 1x1x(B*(5+C)).

###### Complete Network Architecture
> [Ayoosh Kathuria](https://towardsdatascience.com/@ayoosh?source=post_page-----53fb7d3bfe6b----------------------) made a very elaborate diagram that beautifully
> explains the complete architecture of YOLO v3 (Combining both, the
> extractor, and the detector).
>       
> <span><img width="1200" height="600" alt="image" src="Convolutional Neural Network Project 3/Image_006.jpg"/></span>
>       
> Diagram from [https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)
>     
> As seen in the above diagram, where we take an example of a 416x416
> image, the three scales where the detections are made are at the 82nd
> layer, 94th layer, and 106th layer.
> V3 architecture contains residual skipping and upsampling. The salient
> feature of the v3 is the detection capability on three different scales.
> YOLO is a fully convolutional network and its eventual output is
> generated by applying a 1 x 1 kernel on a feature map. In YOLO v3,
> the detection is done by applying 1 x 1 detection kernels on feature
> maps of three different sizes at three different places in the network.
> The multi-scale detector is used to ensure that the small objects are
> also detected in contrast to the previous versions.

#

<br/><br/>
<center><h3>How to train YOLOv3 on a custom dataset</h3></center>

###### Data Preparation
> In this project, I download an entire episode of SpongeBob and then saved a
> frame every 60 frames to create the dataset.
> (vid2frames.py Script)
>       
> ![title](/images/Image_007.jpg) 
>       
> * 160 ~ Photos of SpongeBob
> * 98 ~ Photos of Squidward
> * 80 ~ Pictures of MrKrabs
> * 50 ~ Pictures of Patrick
> * 35 ~ Photos of Plankton
>       
> Another option, you can download ready-made datasets from [googleapis.com](https://storage.googleapis.com/openimages/web/index.html),
> [roboflow](https://public.roboflow.com/) and others...

###### Labeling
>       
> ![title](/images/Image_008.jpg) 
>       
> There are many labeling tools like [VoTT](https://github.com/microsoft/VoTT), [CVAT](https://cvat.org/), and [Labelimg](https://github.com/tzutalin/labelImg).
> In this project, I performed using [TZUTA LIN's](https://tzutalin.github.io/labelImg/) Labelimg software.
> 
> I recommend Labelimg because it is popular and easy to use, you can
> download it from [here](https://tzutalin.github.io/labelImg/) to Windows or Linux operating systems.
>     
>        
> **TIP:** Shortcut keys can be used for quick and convenient labeling.
>       
> ![title](/images/Image_009.jpg) 

###### Getting the files ready for training
> After the labeling, we will see an image(JPG) and a text file with the same
> name, respectively.
> <span><img width="600" height="300" alt="image" src="Convolutional Neural Network Project 3/Image_010.jpg"/></span>
>     
> **in YOLO, the format looks like this**
> 0 0.645508 0.592448 0.134766 0.216146
> 2 0.756836 0.368490 0.138672 0.200521
> 3 0.097656 0.563802 0.187500 0.513021
>     
> | 0 |  0.645508 0.592448 | 0.134766 0.216146 |
> | 2 |  0.756836 0.368490 | 0.138672 0.200521 |
> | 3 |  0.097656 0.563802 | 0.187500 0.513021 |
>
>



