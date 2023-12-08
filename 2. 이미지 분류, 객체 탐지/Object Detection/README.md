![캡쳐1](https://github.com/AYEOOON/AI-project/assets/101050134/1f0b1a36-bf23-4cf6-9f67-a209232fb2ef)

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

