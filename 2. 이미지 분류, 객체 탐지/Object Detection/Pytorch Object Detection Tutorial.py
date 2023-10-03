# Colab에서 튜토리얼 진행

# 1. pycocotools 설치

%%shell

pip install cython
# Install pycocotools, the version by default in Colab
# has a bug fixed in https://github.com/cocodataset/cocoapi/pull/354
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# 2. Defining the Dataset
# Custom dataset을 만들어서 사용할 것
# 사용하는 데이터는 Penn-Fudan dataset, 보행자들에 대한 이미지 데이터셋, Detection으로 사용할 수도 있고, Masking도 있어서 Segmentation도 활용할 수 있음

# 2-1) 데이터셋 받기
%%shell

# download the Penn-Fudan dataset
wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip .
# extract it in the current folder
unzip PennFudanPed.zip

# 데이터셋의 구성은 다음과 같이 Image 데이터와 mask이미지가 다음과 같은 경로에 있다.
PennFudanPed/
  PedMasks/
    FudanPed00001_mask.png
    FudanPed00002_mask.png
    FudanPed00003_mask.png
    FudanPed00004_mask.png
    ...
  PNGImages/
    FudanPed00001.png
    FudanPed00002.png
    FudanPed00003.png
    FudanPed00004.png


# 3. 이미지 시각화
  
# 3-1) 일반 image 데이터
from PIL import Image
Image.open('PennFudanPed/PNGImages/FudanPed00001.png')


# 3-2) mask 이미지
# mask.putpalette()
# putpalette 메소드는 mask이미지에 해당하는 index(label)값을 어떠한 색으로 칠할 것인지를 정하는 것
# np.unique를 사용하면 해당 mask에서 몇개의 label이 있는지 확인 가능
# 일반적으로 배경은 0번, 이후에는 object들

import numpy as np
mask = Image.open('PennFudanPed/PedMasks/FudanPed00001_mask.png')
mask.putpalette([  # 여기서 오류 발생
                 0, 0, 0,     # Black background
                 255, 0, 0,   # index 1 is red
                 255, 255, 0, # index 2 is yellow
                 # 255, 153, 0, # index 3 is orange
                 # 3번째 instance에는 오렌지 색을 칠해줄 것이지만 여기선 두명의 사람만 있음
])
print(mask)
print(np.unique(np.array(mask)))
mask


# 4. Custom Dataset 만들기
# Custom Dataset에서 있어야할 메소드 → __getitem__() + __len__()
# 궁금한 점은... bbox 의 정보를 찾을때 np.where()를 사용하였는데, 일반적으로 인자로 조건과, 참일때의 값, 거짓일 때의 값을 넣는 것으로 알고 있는데... masks라는 bool값이 담긴 것을 넣어 줬을 뿐인데... 뭐지
# 다음과 같은 프로세스가 수행된다.
# __init__ 에서 이미지들의 경로를 list로 받아와서 정렬, imgs, masks로 사용
# __getitem__ 에서는 이 경로들을 idx로 접근, 경로로부터 이미지를 PIL로 열고, 일반 이미지에 대해서만 "RGB"로 convert
# 이후에 np.where를 사용해서 구하는 bbox정보, label, mask 정보, 그리고 area 정보도 tartget dictionary에 담아서 return한다

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image

class PennFudanDataset(torch.utils.data.Dataset) :
    def __init__(self, root, transforms = None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx) :
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # mask는 RGB로 Convert하지 않는다.
        # 각 색깔은 다른 인스턴스를 나타내기 때문, 0은 Background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instance들은 각각 다른 색으로 encode할 것
        obj_ids = np.unique(mask)
        # 0은 배경이므로 obj_ids에서는 뺀다
        obj_ids = obj_ids[1:]

        # 색으로 encode된 마스크를 binary masks로 바꾼다.
        # masks의 shape은 기존 mask의 H, W를 그대로 사용하고 각 픽셀에 대한 label
        # 즉, 위에서 시각화한 이미지를 예로들면 0, 1, 2에셔 배경을 제외한 1, 2, 
        # 2개에 대한 binary mask가 생성될 것
        # masks.shape은 2, H, W가 될 것이다.
        masks = mask == obj_ids[:, None, None]

        # 각각의 마스크를 위한 bbox를 얻는다.
        # pos에는 tuple 값이 들어가게 된다.
        # 이때 두가지의 값이 담기게 되는데, pos[1]의 값에서 x, pos[0]에서 y값을 구한다.
        # 이해가 잘 안간다.. np.where
        num_objs =len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # box정보와 masks정보를 tensor로..
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 오직.. 하나의 class만 있음, (num_objs, )의 shape을 갖는 1이 있는 텐서
        # 하나의 클래스라는 것은 사람을 뜻하는 것일 것 같다.
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.imgs)


# 데이터 확인
# 기본적으로 image, 그리고 area, boxes, image_id, iscrownd, labels, masks를 return 하고 있다. 
dataset = PennFudanDataset('PennFudanPed/')
dataset[np.random.randint(len(dataset))]


# 5. Defineing Model
# 이 Tutorial에서는 R-CNN 계열의 Mask R-CNN을 사용
# Yolo에 비해서 좀 더 느리지만 정확도는 더 높다.
# pre-trained된 모델을 fine-tuning해서 사용할 것이다.
# Fast R-CNN모델을 사용함과 동시에 Segmentation Mask도 계산할 것이기 때문에 Mask R-CNN도 사용

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_instance_segmentation_model(num_classes):
    # COCO 데이터에 사전학습된 instance segmentation 모델을 로드
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # classifier의 입력으로 들어가는 feature의 수
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 사전학습된 head를 교체
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # mask classifier의 입력 feature로 들어가는 수
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # mask predictor를 교체
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    
    return model

# 6. 학습 전 준비
# 학습을 진행하기 전에 학습에 필요한 코드들을 가져온다

%%shell

# Download TorchVision repo to use some files from
# references/detection
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.3.0

cp references/detection/utils.py ../
cp references/detection/transforms.py ../
cp references/detection/coco_eval.py ../
cp references/detection/engine.py ../
cp references/detection/coco_utils.py ../

# augmentation, transformation 적용
# mean/std nomalization과 image scaling을 따로 필요없다고 한다.
# Mask R-CNN 모델에 의해서 handled된다고 한다.

from engine import train_one_epoch, evaluate  
# 여기서 No Module Named ' torch._six '오류가 발생했었는데 버전이 맞지 않아 발생하는문제인줄 알고 한참 헤매다가
# 오류시 생기는 링크인 /content/coco_eval.py 를 들어가 import torch_six를 지우니 해결되었다. 
import utils
import transforms as T

def get_transform(train) :
    transforms = []
    # image > PIL Image > PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # 학습하는 동안에는 랜덤하게 flip
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# ​7. DataLoader 만들기

# 데이터셋을 사용해서 transformation정의하기
dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

# train과 test 나누기
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# train과 valid loader만들기
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)


# 8. 모델 및 optimizer 설정

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2

model = get_instance_segmentation_model(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr = 0.005,
                            momentum=0.9, weight_decay=0.0005)

# lr schedule
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# 9.  학습

num_epochs = 10

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)  # 여기서  module 'torch' has no attribute '_six'오류가 발생하지만
​
# 밑에 코드들은 잘 돌아간다..

# 10.평가

img, _ = dataset_test[0]
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])
Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())

