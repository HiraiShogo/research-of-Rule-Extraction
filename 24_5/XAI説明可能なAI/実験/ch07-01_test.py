import matplotlib.pyplot as plt
from PIL import Image
import json

import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

img = Image.open('dataset/both.png').convert('RGB')
fig = plt.figure()
plt.imshow(img)

with open("dataset/imagenet_class_index.json", "r") as f:
    cls_idx = json.load(f)
    idx2label = [cls_idx[str(k)][1] for k in range(len(cls_idx))]

# 学習済みモデルの読み込み
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()
model.to(device)


# 画像のプリプロセス
preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)

# モデルの推論
img_tensor = preprocess(img).unsqueeze(0).to(device)
logits = model(img_tensor)
probs = F.softmax(logits, dim=1)

# 上位5位の結果を確認
probs5 = probs.topk(5)
probability = probs5[0][0].detach().cpu().numpy()
class_id = probs5[1][0].detach().cpu().numpy()
for p, c in zip(probability, class_id):
    print((p, c, idx2label[c]))

from lime.wrappers.scikit_image import SegmentationAlgorithm
import numpy as np
from skimage.segmentation import mark_boundaries

# 画像の領域分割（quickshift）
segmentation_fn = SegmentationAlgorithm(
    'quickshift',
    kernel_size=4,
    max_dist=200, ratio=0.2,
    random_seed=42
)

segments = segmentation_fn(img)
fig = plt.figure()
plt.imshow(mark_boundaries(np.array(img), segments))



import sys
sys.path.append("pytorch-grad-cam")

from gradcam import GradCam

grad_cam = GradCam(
    model=model,
    feature_module=model.layer4,
    target_layer_names=["2"],
    use_cuda=torch.cuda.is_available()
)

grayscale_cam = grad_cam(img_tensor, idx2label.index("bull_mastiff"))

import cv2
#dog
fig = plt.figure()
plt.imshow(img)
plt.imshow(
    cv2.resize(grayscale_cam, (img.size[1], img.size[0])),
    alpha=0.5,
    cmap='jet'
)
plt.colorbar()
plt.savefig('result/dog_gradcam.png')

#cat
grayscale_cam = grad_cam(img_tensor, idx2label.index("tiger_cat"))
fig = plt.figure()
plt.imshow(img)
plt.imshow(
    cv2.resize(grayscale_cam, (img.size[1], img.size[0])),
    alpha=0.5,
    cmap='jet'
)
plt.colorbar()
plt.savefig('result/cat_gradcam.png')

plt.show()