from PIL import Image
import numpy as np
import torch
import time
import torchvision.transforms as transforms

# img = Image.open('/home2/zxp/Projects/FoundationPose/demo_data/kinect_driller_seq/depth/0000001.png').convert("RGB")
img = Image.open('/home2/zxp/Projects/FoundationPose/demo_data/square_table_leg/depth_/0000000000000000000.png')
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])
# img = transform(img)
img = np.array(img)
print(img.dtype)
print(img.shape)
print(img.max())
