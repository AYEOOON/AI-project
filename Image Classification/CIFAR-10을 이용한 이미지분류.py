# 이미지 분류기 학습하기
# 1. CIFAR10의 학습용 / 시험(test)용 데이터셋을 torchvision 을 사용하여 불러오고, 정규화(nomarlizing)합니다.
# 2. 합성곱 신경망(Convolution Neural Network)을 정의합니다.
# 3. 손실 함수를 정의합니다.
# 4. 학습용 데이터를 사용하여 신경망을 학습합니다.
# 5. 시험용 데이터를 사용하여 신경망을 검사합니다.


# 1. CIFAR10를 불러오고 정규화하기
# torchvision을 사용하면 매우 쉽게 CIFAR10 데이터를 불러올 수 있다. 
import torch
import torchvision
import torchvision.transforms as transforms

# torchvision데이터셋의 출력은 [0,1]범위를 갖는 PILImage 이미지이다. 이를 [-1,1]의 범위로 정규화된 Tensor로 변환
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 테스트
import matplotlib.pyplot as plt
import numpy as np

# 이미지를 보여주기 위한 함수


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# 학습용 이미지를 무작위로 가져오기
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 이미지 보여주기
imshow(torchvision.utils.make_grid(images))
# 정답(label) 출력
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# 2.합성곱 신경망(Convolution Neural Network)정의하기
# 3채널 이미지를 처리할 수 있도록 함
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        # Conv2d(in_channels, out_channels, kernel_size)
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


# 3.손실 함수와 Optimizer정의하기
# 분류에 대한 교차 엔트로피 손실과 momentum을 갖는 SGD를 사용함
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# 4. 신경망 학습하기
# 데이터를 반복해서 신경망에 입력으로 제공하고, 최적화(Optimze)만 하면 된다. 
for epoch in range(2):  # 데이터셋을 수차례 반복합니다.

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 입력을 받은 후,
        inputs, labels = data

        # Variable로 감싸고
        inputs, labels = Variable(inputs), Variable(labels)

        # 변화도 매개변수를 0으로 만든 후
        optimizer.zero_grad()

        # 학습 + 역전파 + 최적화
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계 출력
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# 신경망 학습을 1번 반복으로 횟수를 감소시켰을 때
# loss : 1.463으로 증가
# Ground-Truth: cat ship ship plane
# Predicted : cat car plne plane
# Accuracy of the network on the 10000 test images: 48 %
# 2회 반복보다 정확도 감소

# 신경망 학습을 4번 반복으로 횟수를 증가시켰을 때
# loss : 1.133으로 감소
# Ground-Truth: cat ship ship plane
# Predicted : cat car plne plane
# Accuracy of the network on the 10000 test images: 59 %
# 2회 반복보다 정확도 증가

# 신경망 학습을 10번 반복으로 횟수를 증가시켰을 때
# loss : 0.859으로 감소
# Ground-Truth: cat ship ship plane
# Predicted : cat ship plne plane
# Accuracy of the network on the 10000 test images: 62 %
# 2회 반복보다 정확도 증가
# Accuracy for class: plane is 76.9 %
# Accuracy for class: car   is 75.5 %
# Accuracy for class: bird  is 41.0 %
# Accuracy for class: cat   is 45.2 %
# Accuracy for class: deer  is 66.3 %
# Accuracy for class: dog   is 57.3 %
# Accuracy for class: frog  is 69.0 %
# Accuracy for class: horse is 57.3 %
# Accuracy for class: ship  is 66.7 %
# Accuracy for class: truck is 72.4 %



# 5. 시험용 데이터로 신경망 검사하기
# 주어진 코드에서는 학습용 데이터 셋을 2회 반복하여 신경망 학습, 신경망이 예측한 정답과 진짜 정답을 비교하는 방학으로 확인.
# 예측이 맞다면 샘플을 '맞은 예측값'에 넣는다. 
# 먼저 시험용 데이터를 살펴보면
dataiter = iter(testloader)
images, labels = dataiter.next()

# 이미지 출력
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 그 다음 신경망이 어떻게 예측했는지 본다. 
outputs = net(Variable(images))

# 출력은 10개 분류 각각에 대한 값으로 나타낸다. 어떤 분류에 대해서 더 높은 값이 나타난다는 것은 신경망이 그 이미지가 더 해당 분류에 가깝다고 생각하는 것
# 따라서, 가장 높은 값을 갖는 인덱스를 뽑는다.
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# 전체 데이터 셋에 대한 동작 방식을 테스트하는 코드
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# 어떤것을 더 잘 학습하고, 어떤 것을 못했는지 확인하는 코드
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# GPU에서 학습하기
# CPU와 비교했을 때 어마어마한 속도 차이가 나지 않는 것은 왜 그럴까요? 그 이유는 바로 신경망이 너무 작기 때문입니다.
# Exercise: 신경망의 크기를 키웠을 때 얼마나 빨라지는지 확인해보세요. (첫번째 nn.Conv2d 의 2번째 매개변수와 두번째 nn.Conv2d 의 1번째 매개변수는 같아야 합니다.)

# out_channels = 9, in_channels = 9일때
# 신경망 학습 속도가 빨리지는지는 모르겠음
# 근데 사진 잘맞춤
# 2회반복, loss:1.227
# GroundTruth: cat ship ship plane
# predicted: cat ship ship plane
# 정확도 56%

# self.conv1 = nn.Conv2d(3, 12, 5), self.conv2 = nn.Conv2d(12, 16, 5)
# 2회 반복, loss : 1.266
# GroundTruth: cat ship ship plane
# Predicted:  cat   car   car   ship 
# 정확도: 57%
