import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

import torch.nn as nn





import os
os.environ['OPENAI_KEY'] = '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'
# # 여기서 하면 됨 https://platform.openai.com/account/api-keys
# os.chdir('/Image2Paragraph_main')

import argparse
from PIL import Image
import base64
from io import BytesIO
import os
from torchvision import transforms
import torchvision.transforms as T
from torch.utils.data import DataLoader

import numpy as np
import torch
import random






pipe =  StableDiffusionPipeline.from_pretrained("/work/home/acxfcc8cum/sk/joint_train/weight/stable-diffusion-v1-5", torch_dtype=torch.float16) 
print("加载第一步")
pipe.unet.load_attn_procs("/work/home/acxfcc8cum/sk/joint_train/best_496")
print("加载第二步")
pipe.to("cuda")



# 读取script.txt文件
with open('/work/home/acxfcc8cum/sk/joint_train/inference_images/text/22.txt', 'r') as file:
    lines = file.readlines()

save_folder = "/work/home/acxfcc8cum/sk/joint_train/inference_images/images/22_backpack-umbrella"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 遍历每一行文本作为text_prompt
for index, text_prompt in enumerate(lines):
    text_prompt = text_prompt.strip()  # 去除换行等空白字符
    for i in range(15, 19):
        generator = torch.Generator("cuda").manual_seed(i)
        number = random.randint(1, 100000)
        image = pipe(text_prompt, num_inference_steps=50, guidance=7.5, generator=generator).images[0]
        image.save(f'{save_folder}/{number}.jpg')








