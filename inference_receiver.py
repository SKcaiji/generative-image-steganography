import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn

import torch.nn.init as init

# 加载预训练的VGG16模型
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=32, bias=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for layer in self.classifier:
            x = layer(x)
           # print("Layer output shape:", x.shape)
        return x
model = VGG()
pretrained_weights_path = '/work/home/acxfcc8cum/sk/joint_train/best_496_32/vgg16.pth'
pretrained_state_dict = torch.load(pretrained_weights_path)
model.load_state_dict(pretrained_state_dict, strict=False)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])

# 指定包含图片的文件夹路径，可根据实际情况修改
image_folder_path = "/work/home/acxfcc8cum/sk/joint_train/batch_bestimages496_100"
ture_count = 0
false_count = 0
for subfolder in os.listdir(image_folder_path):
    subfolder_path = os.path.join(image_folder_path, subfolder)
    if os.path.isdir(subfolder_path):
        for root, dirs, files in os.walk(subfolder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    image = Image.open(image_path)
                    my_dict = {"person": 0, "bicycle": 1, "boat": 2, "bird": 3, "cat": 4, "dog": 5, "bear": 6, "zebra": 7, 
                                "backpack": 8, "umbrella": 9, "tie": 10, "skis": 11, "banana": 12, "orange": 13, "hot_dog": 14, 
                                "pizza": 15, "couch": 16, "bed": 17, "dining table": 18, "tv": 19, "laptop": 20, "mouse": 21, "keyboard": 22,
                                "microwave": 23, "refrigerator": 24, "vase": 25, "broccoli": 26, "sheep": 27, "cow": 28, "airplane": 29, 
                                "motorcycle": 30, "teddy bear": 31}

                    category_str = os.path.basename(os.path.dirname(image_path))
                    categories = category_str.split('_')
                    #print(categories)

                    index = []
                    for cla in categories:
                        if cla in my_dict:
                            index.append(my_dict[cla])
                   # print(index)
                    image_tensor = transform(image).unsqueeze(0)

                    with torch.no_grad():
                        output = model(image_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)

                        topk = 2
                        topk_probabilities, topk_indices = torch.topk(probabilities, topk, dim=1)

                        print(f"图片 {image_path} 最高的两个类别概率:")
                        for i in range(topk):
                            print(f"类别 {topk_indices[0][i].item()}: {topk_probabilities[0][i].item()}")

                        topk_indices = topk_indices.squeeze(0).tolist()
                        #print(topk_indices)
                        flag = all([idx in topk_indices for idx in index])
                       # print(flag)
                        with open('1.txt', 'a') as f1, open('2.txt', 'a') as f2:
                            if flag:
                                f1.write(image_path + '\n')
                                ture_count += 1
                            else:
                                f2.write(image_path + '\n')
                                false_count += 1
acc = ture_count/(ture_count + false_count)
print(f"正确识别的数量：{ture_count}")
print(f"错误识别的数量：{false_count}")
print(f"准确率为: {acc} ")