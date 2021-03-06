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
>       
> | class index | (x, y) coordinates | width, height |
> | ----------- | ------------------ | ----------------- | 
> | 0           |  0.645508, 0.592448 | 0.134766, 0.216146 |
> | 2           |  0.756836, 0.368490 | 0.138672, 0.200521 |
> | 3           |  0.097656, 0.563802 | 0.187500, 0.513021 |
>     
> Compress them to a zip folder and rename it to "images.zip".
>       
> Create a folder named "yolov3" in Google Drive and put the dataset and
> train_yolov3 notebook in it.
>        
> ![title](/images/Image_011.jpg)

###### Training the model (Darknet framework)
> Darknet is an open-source neural network framework written in C and CUDA.
> It is fast, easy to install, and supports CPU and GPU computation. You can
> find the source on [GitHub](https://github.com/pjreddie/darknet).
>            
> Reference: [pjreddie.com](https://pjreddie.com/darknet/yolo/).
>            
> ![title](/images/Image_012.jpg)
>            
> Now let's start with Google Colab Notebook.
>            
> 1.  Make sure you enable GPU.
>       
> ![title](/images/GPU.PNG)
>            
> 2. Check it      
> ![title](/images/Image_015.jpg)
>            
> You can see that I got a Tesla K80 GPU using the 11.2 Cuda version.
> bad luck, sometimes I get the Tesla v100 😉.
>            
> 3.  Connect Colab with Google Drive  
> ![title](/images/Image_016.jpg)
>            
> 4.  Get the Darknet model
>           
> ![title](/images/Image_017.jpg)
>            
> 5.  Edit the .cfg to fit your needs based on your object detector
> ![title](/images/Image_018.jpg)
>            
> **How to Configure Your Variables:**
> ```    
> width = 416
> height = 416
> (these can be any multiple of 32, 416 is standard, you can sometimes improve
> results by making value larger like 608 but will slow down training).
>        
> Max_batches = (number of classes) * 2000
> (but no less than 6000 so if you are training for 1, 2, or 3 classes it will be 6000,
> however, the detector for 5 classes would have max_batches=10000).
>        
> Steps = (80% of max_batches), (90% of max_batches)
> (so, if your max_batches = 10000, then steps = 8000, 9000).
>      
> Filters = (number of classes + 5) * 3
>       
> (so, if you are training for one class then your filters = 18, but if you are training
> for 4 classes then your filters = 27).
>       
> Optional: If you run into memory issues or find the training taking a super
> long time. In each of the three yolo layers in the cfg, change one line from
> random = 1 to random = 0 to speed up training but slightly reduce the
> accuracy of the model. Will also help save memory if you run into any
> memory issues.
> ```
>            
> 6.  Create your .names and .data files.
> 
> ![title](/images/Image_019.jpg)
>        
> 7.  Save .cfg and obj.names files in Google drive.
>        
> ![title](/images/Image_020.jpg)
>        
> 8.  Create a folder and unzip the image dataset.
>        
> ![title](/images/Image_021.jpg)
>
> 9.  Create train.txt file.
>          
> ![title](/images/Image_022.jpg)
>          
> 10. Download pre-trained weights.
>        
> ![title](/images/Image_023.jpg)        
>         
> For training, we use convolution weights that are pre-trained on the
> ImageNet dataset.
>        
> 11. Finally, start training.
>          
> ![title](/images/Image_024.jpg)
>     
> The training will take a long time, so take a great time to rest.       
>
> **TIP:** 
> ```
> Colab Cloud Service kicks you off if you are idle for too long
> (60-90 mins). 
>      
> To avoid this, you can download auto-clicker software or make a
> python script.
> ```
>            
> 12. Testing - You can download the weights file to your local machine and
check it on the validation data.
>
> Testing - You can download the weights file to your local machine and check it on the validation data.</p></li></ol><p style="text-indent: 0pt;text-align: left;"><br/></p><p class="s30" style="padding-left: 23pt;text-indent: 0pt;text-align: left;"><span><img width="320" height="181" alt="image" src="Convolutional Neural Network Project 3/Image_025.jpg"/></span>	<span><img width="320" height="180" alt="image" src="Convolutional Neural Network Project 3/Image_026.jpg"/></span></p><p style="text-indent: 0pt;text-align: left;"><br/></p><p style="padding-left: 153pt;text-indent: 0pt;text-align: left;"><span><img width="382" height="206" alt="image" src="Convolutional Neural Network Project 3/Image_027.jpg"/></span></p><p style="text-indent: 0pt;text-align: left;"><br/></p><p class="s30" style="padding-left: 23pt;text-indent: 0pt;text-align: left;"><span><img width="316" height="177" alt="image" src="Convolutional Neural Network Project 3/Image_028.jpg"/></span>	<span><img width="316" height="177" alt="image" src="Convolutional Neural Network Project 3/Image_029.jpg"/></span></p><p class="s6" style="padding-top: 3pt;padding-left: 88pt;text-indent: 0pt;text-align: center;">References</p><ol id="l11"><li><p style="padding-top: 24pt;padding-left: 124pt;text-indent: -18pt;text-align: left;"><a href="https://pjreddie.com/media/files/papers/YOLOv3.pdf">https://pjreddie.com/media/files/papers/YOLOv3.pdf</a></p></li><li><p style="padding-top: 1pt;padding-left: 124pt;text-indent: -18pt;text-align: left;"><a href="https://machinelearningmastery.com/object-recognition-with-deep-learning/">https://machinelearningmastery.com/object-recognition-with-deep-learning/</a></p></li><li><p style="padding-top: 1pt;padding-left: 124pt;text-indent: -18pt;line-height: 108%;text-align: left;"><a href="https://towardsdatascience.com/digging-deep-into-yolo-v3-a-hands-on-guide-part-1-78681f2c7e29" class="a" target="_blank">https://towardsdatascience.com/digging-deep-into-yolo-v3-a-hands-on-guide-</a><a href="https://towardsdatascience.com/digging-deep-into-yolo-v3-a-hands-on-guide-part-1-78681f2c7e29" target="_blank"> part-1-78681f2c7e29</a></p></li><li><p style="padding-left: 124pt;text-indent: -18pt;line-height: 108%;text-align: left;"><a href="https://medium.com/%4095shanu/the-ultimate-yolo-guide-to-train-on-custom-dataset-e7095d084f0c" class="a" target="_blank">https://medium.com/@95shanu/the-ultimate-yolo-guide-to-train-on-custom-</a><a href="https://medium.com/%4095shanu/the-ultimate-yolo-guide-to-train-on-custom-dataset-e7095d084f0c" target="_blank"> dataset-e7095d084f0c</a></p></li><li><p style="padding-left: 124pt;text-indent: -18pt;line-height: 108%;text-align: left;"><a href="https://medium.com/analytics-vidhya/yolo-object-detection-made-easy-7b17cc3e782f" class="a" target="_blank">https://medium.com/analytics-vidhya/yolo-object-detection-made-easy-</a><a href="https://medium.com/analytics-vidhya/yolo-object-detection-made-easy-7b17cc3e782f" target="_blank"> 7b17cc3e782f</a></p></li><li><p style="padding-left: 124pt;text-indent: -18pt;line-height: 14pt;text-align: left;"><a href="https://github.com/tzutalin/labelImg">https://github.com/tzutalin/labelImg</a></p></li><li><p style="padding-top: 1pt;padding-left: 124pt;text-indent: -18pt;line-height: 109%;text-align: left;"><a href="https://learnopencv.com/training-yolov3-deep-learning-based-custom-object-detector/" class="a" target="_blank">https://learnopencv.com/training-yolov3-deep-learning-based-custom-object-</a><a href="https://learnopencv.com/training-yolov3-deep-learning-based-custom-object-detector/" target="_blank"> detector/</a></p></li><li><p style="padding-left: 124pt;text-indent: -18pt;line-height: 14pt;text-align: left;"><a href="https://github.com/AlexeyAB/darknet">https://github.com/AlexeyAB/darknet</a></p></li><li><p style="padding-top: 1pt;padding-left: 124pt;text-indent: -18pt;text-align: left;"><a href="https://en.wikipedia.org/wiki/Object_detection">https://en.wikipedia.org/wiki/Object_detection</a></p></li><li><p style="padding-top: 1pt;padding-left: 124pt;text-indent: -18pt;line-height: 108%;text-align: left;"><a href="https://prashantdandriyal.medium.com/decoding-yolov3-output-with-intel-openvinos-backend-part-1-a2a478cd93ca" class="a" target="_blank">https://prashantdandriyal.medium.com/decoding-yolov3-output-with-intel-</a><a href="https://prashantdandriyal.medium.com/decoding-yolov3-output-with-intel-openvinos-backend-part-1-a2a478cd93ca" target="_blank"> openvinos-backend-part-1-a2a478cd93ca</a></p></li></ol></body></html>
>
>
>        
>
>
>
>
>
>        
>
>
>
>
>
>        
>
>
>
>
>
