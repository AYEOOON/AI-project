![캡쳐1](https://github.com/AYEOOON/AI-project/assets/101050134/1f0b1a36-bf23-4cf6-9f67-a209232fb2ef)
# 객체 탐지 관련 내용 정리
#### 이미지 분류, 객체 탐지 코드 실행 및 분석
RNN & Selective Search - https://www.notion.so/R-CNN-Selective-Search-3f39d6a5e5a24cea91bc96229a0e066c
FCN - https://www.notion.so/FCN-Fully-Convolution-Network-7ee240b80e9e47179fef828d4ef3fcbf

# 객체 탐지 (Object detection)
물체 검출은 이미지 내에서 알고리즘을 훈련시킬 때 사용된 클래스 라벨에 속하는 모든 물체를 검출하고, 그 위치들도 바운딩 박스로 알려줍니다. 

만약 훈련된 클래스 라벨에 속하는 물체가 이미지 내에 없으면 아무 것도 검출해내지 않는다. 물체 검출은 객체 검출이라고 불리기도 합니다. 

## TORCHVISION 객체 검출 미세조정(FINETUNING) 튜토리얼
본 튜토리얼에서는 Penn-Fudan Database for Pedestrian Detection and Segmentation 데이터셋으로 미리 학습된 Mask R-CNN 모델을 미세조정을 합니다. 

데이터셋에는 보행자 인스턴스(이미지 내에서 사람의 위치 좌표와 픽셀 단위의 사람 여부를 구분한 정보)를 포함합니다.

345명이 있는 170개의 이미지가 포함되어 있으며, 우리는 이 이미지를 사용하여 사용자 정의 데이터셋에 인스턴스 분할(Instance Segmentation) 모델을 학습하기 위해 torchvision의 새로운 기능을 사용하는 방법을 알려줍니다. 

### 모델 정의하기

이 튜토리얼에서는 Faster R-CNN에 기반한 Mask R-CNN 모델을 사용합니다. 

Faster R-CNN은 이미지에 존재할 수 있는 객체에 대한 바운딩 박스와 클래스 점수를 모두 예측하는 모델입니다. 
![tv_image03](https://github.com/AYEOOON/AI-project/assets/101050134/d54f35af-7518-420c-8f66-05610c54be12)

Mask R-CNN은 각 인스턴스에 대한 분할 마스크 예측하는 추가 분기(레이어)를 Faster R-CNN에 추가한 모델입니다. 
![tv_image04](https://github.com/AYEOOON/AI-project/assets/101050134/be25809f-c02d-4f45-b057-589e556e962b)
Torchvision 모델주(미리 학습된 모델들을 모아 놓은 공간)에서 사용 가능한 모델들 중 하나를 이용해 모델을 수정하려면 보통 두가지 상황이 있습니다. 

첫 번째 방법은 미리 학습된 모델에서 시작해서 마지막 레이어 수준만 미세 조정하는 것입니다. 

다른 하나는 모델의 백본을 다른 백본으로 교체하는 것입니다.(예를 들면, 더 빠른 예측을 하려고 할때, 백본 모델을 ResNet101 에서 MobilenetV2 로 교체하면 수행 속도 향상을 기대할 수 있습니다. 대신 인식 성능은 저하 될 수 있습니다.)

### 요약
이 튜토리얼에서는 사용자 정의 데이터셋에서 인스턴스 분할 모델을 위한 자체 학습 파이프라인을 생성하는 방법을 배웠습니다. 

이를 위해 영상과 정답 및 분할 마스크를 반환하는 torch.utils.data.Dataset 클래스를 작성했습니다. 

또한 이 새로운 데이터 셋에 대한 전송 학습(Transfer learning)을 수행하기 위해 COCO train2017에 대해 미리 학습된 Mask R-CNN 모델을 활용 했습니다.

### 더 자세한 내용은
https://tutorials.pytorch.kr/intermediate/torchvision_tutorial.html

## Selective Search
기존의 exhaustive search의 방식의 비효율성으로 "object가 있을 법한 영역만 찾는 방법"이 제안되었다.

고정된 window 사이즈는 각기 다른 object의 size나 shape을 포착하기 어렵다. 

만약 object recognition을 실행하기 전에 아래와 같이 이미지를 올바르게 segment하면 segmented result에 대해서 candidate object로 사용할 수 있지 않을까 -> selective search 
![다운로드 (1)](https://github.com/AYEOOON/AI-project/assets/101050134/c67a8f2e-867f-4f1f-9632-512f6d93b839)

### Selctive Search의 목표
object 인식이나 검출을 위한 가능한 후보 영역을 알아낼 수 있는 방법을 제공하는 것을 목표
![다운로드](https://github.com/AYEOOON/AI-project/assets/101050134/dac291d8-be5f-44a7-baf8-22378b7f1447)

### Selctive Search의 과정
1️⃣ 입력 영상에 대해 segmentation을 실시해서 이를 기반으로 후보 영역을 찾기 위한 seed를 설정
2️⃣ 초기에 엄청나게 많은 후보들이 만들어 진다. 
3️⃣ 이를 적절하게 통합해 나가면, segmentation은 후보 영역의 개수가 줄어들고, 결과적으로 이를 바탕으로 box의 후보 개수도 줄어든다. 

### Selctive Search의 단점
region proposal 과정이 실제 object detection CNN과 별도로 이루어지기 때문에, selective search를 사용하면 end-to-end로 학습이 불가능하고, 실시간 적용에도 어려움이 있다. 
