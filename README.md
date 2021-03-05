# Train Convolutional Neural Network - YOLO Algorithm

In this project, we will talk about YoloV3 Architecture and how to train it on a custom dataset, I will explain step by step how to do it by using the Darknet framework.

<!DOCTYPE  html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en-us" lang="en-us"><head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"/><title>Untitled</title><meta name="author" content="Itay Nave"/><style type="text/css"> * {margin:0; padding:0; text-indent:0; }
 .s1 { color: #202020; font-family:"Times New Roman", serif; font-style: normal; font-weight: normal; text-decoration: none; font-size: 14pt; }
 p { color: #202020; font-family:"Times New Roman", serif; font-style: normal; font-weight: normal; text-decoration: none; font-size: 11pt; margin:0pt; }
 li {display: block; }
 #l1 {padding-left: 0pt; }
 #l1> li>*:first-child:before {content: " "; color: #202020; font-family:Symbol, serif; font-style: normal; font-weight: normal; text-decoration: none; font-size: 11pt; }
 #l2 {padding-left: 0pt; }
 #l2> li>*:first-child:before {content: "o "; color: #202020; font-family:"Courier New", monospace; font-style: normal; font-weight: normal; text-decoration: none; font-size: 11pt; }
 #l3 {padding-left: 0pt; }
 #l3> li>*:first-child:before {content: " "; color: #202020; font-family:Wingdings; font-style: normal; font-weight: normal; text-decoration: none; font-size: 11pt; }
 li {display: block; }
 #l4 {padding-left: 0pt; }
 #l4> li>*:first-child:before {content: "" "; color: #202020; font-family:"Times New Roman", serif; font-style: normal; font-weight: normal; text-decoration: none; font-size: 14pt; }
 #l5 {padding-left: 0pt; }
 #l5> li>*:first-child:before {content: " "; color: #202020; font-family:Symbol, serif; font-style: normal; font-weight: normal; text-decoration: none; font-size: 11pt; }
</style></head><body><p class="s1" style="padding-left: 1pt;text-indent: 0pt;line-height: 15pt;text-align: left;">Ⅰ – Introduction</p><ul id="l1"><li><p style="padding-top: 1pt;padding-left: 42pt;text-indent: -18pt;line-height: 13pt;text-align: left;">What is Object Detection?</p></li><li><p style="padding-left: 42pt;text-indent: -18pt;line-height: 13pt;text-align: left;">How does object detection work?</p></li><li><p style="padding-left: 42pt;text-indent: -18pt;line-height: 13pt;text-align: left;">YOLO - You Only Look Once</p><ul id="l2"><li><p style="padding-left: 78pt;text-indent: -18pt;line-height: 13pt;text-align: left;">YOLO v3.</p><ul id="l3"><li><p style="padding-left: 114pt;text-indent: -18pt;line-height: 12pt;text-align: left;">Network Architecture</p></li><li><p style="padding-left: 114pt;text-indent: -18pt;line-height: 13pt;text-align: left;">Feature Extractor</p></li><li><p style="padding-left: 114pt;text-indent: -18pt;text-align: left;">Feature Detector</p></li><li><p style="padding-left: 114pt;text-indent: -18pt;text-align: left;">Complete Network Architecture</p></li></ul></li></ul></li></ul><p style="text-indent: 0pt;text-align: left;"><br/></p><ul id="l4"><li><p class="s1" style="padding-left: 18pt;text-indent: -12pt;text-align: left;">– How to train YOLOv3 on a custom dataset.</p><p style="text-indent: 0pt;text-align: left;"><br/></p><ul id="l5"><li><p style="padding-left: 42pt;text-indent: -18pt;line-height: 13pt;text-align: left;">Data Preparation</p></li><li><p style="padding-left: 42pt;text-indent: -18pt;line-height: 13pt;text-align: left;">Labelimg</p></li><li><p style="padding-left: 42pt;text-indent: -18pt;line-height: 13pt;text-align: left;">Getting the files ready for training</p></li><li><p style="padding-left: 42pt;text-indent: -18pt;line-height: 13pt;text-align: left;">Training the model by using Darknet</p></li><li><p style="padding-left: 42pt;text-indent: -18pt;line-height: 13pt;text-align: left;">Use your custom weights for object detection.</p></li></ul></li></ul></body></html>

