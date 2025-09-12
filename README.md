# generative-image-steganography
1.数据集的构建：
首先生成描述每张图片中包含物体类别的文本提示，然后利用该文本提示生成对应的图片。
根据文本和图片构建train.jsonl.（完整的数据集在https://huggingface.co/datasets/shikedpx/multi-class_image可下载）

2.预训练权重
训练代码中的初始权重为vgg16在imagenet数据集的预训练权重和stable-diffusion-v1-5预训练权重（这两个预训练权重在各自官网均有提供）

3.训练
根据train.py实现对总体网络的训练。

4.测试
利用训练好的权重在发射端和接收端的权重进行测试。
