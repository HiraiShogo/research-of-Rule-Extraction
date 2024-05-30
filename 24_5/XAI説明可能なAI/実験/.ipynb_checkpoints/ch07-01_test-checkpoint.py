import matplotlib.pyplot as plt
from PIL import Image
import json

import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

img = Image.open('dataset/both.png').convert('RGB')
plt.imshow(img)

with open("dataset/imagenet_class_index.json", "r") as f:
    cls_idx = json.load(f)
    idx2label = [cls_idx[str(k)][1] for k in range(len(cls_idx))]

# 学習済みモデルの読み込み
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
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
plt.imshow(mark_boundaries(np.array(img), segments))



def batch_predict(images):
    # 画像のプリプロセスとbatch化
    batch = torch.stack(tuple(preprocess(i) for i in images), dim=0)
    batch = batch.to(device)

    # モデルの推論
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


def batch_predict(images):
    # 画像のプリプロセスとbatch化
    batch = torch.stack(tuple(preprocess(i) for i in images), dim=0)
    batch = batch.to(device)

    # モデルの推論
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


from lime import lime_image

explainer = lime_image.LimeImageExplainer(random_state=42)
explanation = explainer.explain_instance(
    np.array(img),
    batch_predict,
    top_labels=2,
    hide_color=0,
    num_samples=5000,
    segmentation_fn=segmentation_fn
)

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
class_index = explanation.top_labels[0]
class_label = idx2label[class_index]
print(f"class_index: {class_index}, class_label: {class_label}")
image, mask = explanation.get_image_and_mask(
    class_index, positive_only=False, num_features=5, hide_rest=False
)
plt.imshow(img)
plt.imshow(
    cv2.resize(grayscale_cam, (image.shape[1], image.shape[0])),
    alpha=0.5,
    cmap='jet'
)
plt.colorbar()
plt.savefig('result/dog_gradcam.png')

#cat
class_index = explanation.top_labels[1]
class_label = idx2label[class_index]
print(f"class_index: {class_index}, class_label: {class_label}")
image, mask = explanation.get_image_and_mask(
    class_index,
    positive_only=False,
    negative_only=False,
    num_features=5,
    hide_rest=False
)
grayscale_cam = grad_cam(img_tensor, idx2label.index("tiger_cat"))

plt.imshow(img)
plt.imshow(
    cv2.resize(grayscale_cam, (image.shape[1], image.shape[0])),
    alpha=0.5,
    cmap='jet'
)
plt.colorbar()
plt.savefig('result/cat_gradcam.png')
print(image.shape[1],image.shape[0])
