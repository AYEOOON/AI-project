# 1.selective search 설치
!pip install selectivesearch



# 2. 이미지 불러오기 및 출력
import selectivesearch
import cv2
import matplotlib.pyplot as plt
import os
from google.colab.patches import cv2_imshow

img = cv2.imread('/content/drive/MyDrive/dogs.jpg')
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print("image shape",img.shape)

plt.figure(figsize=(8,8))
plt.imshow(img_rgb)

plt.show()



# 3. selective search 사용
import selectivesearch

#scale 크다 -> 큰 object 위주로 찾아라, min_size -> 적어도 min_size 이상의 object 찾아라
#regions -> object가 있을만한 공간
_, regions = selectivesearch.selective_search(img_rgb, scale=100, min_size=2000)

print(type(regions), len(regions))



# 4. bounding box 그려보기
cand_rects = [cand['rect'] for cand in regions if cand['size'] > 10000] # rects만 가져와서 저장

green_rgb = (125,255,51)
img_rgb_copy = img_rgb.copy() # 그림 카피 뜨기

for rect in cand_rects:
  left=rect[0]
  top=rect[1]

  right = left + rect[2]
  bottom = top + rect[3]

  img_rgb_copy = cv2.rectangle(img_rgb_copy, (left,top),(right,bottom),color=green_rgb, thickness=2) 
  #사각형 좌표의 왼쪽 상단, 오른쪽 하단 입력

plt.figure(figsize=(8,8))
plt.imshow(img_rgb_copy)
plt.show()

