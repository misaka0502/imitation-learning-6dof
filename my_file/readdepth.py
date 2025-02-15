from PIL import Image
import numpy as np

# image = Image.open("images/depth_image2.png")
image = Image.open("/home2/zxp/Projects/FoundationPose/demo_data/mustard0/depth/1581120424100262102.png")
image = np.array(image)
print(image.max())