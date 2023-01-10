# HG_MLDL
## 2023-01-10(화)
1. BreamAndSmelt.ipynb
- 산점도 : x와 y의 상관관계 표현
- 산점도 그래프가 일직선에 가까운 형태로 나타나는 경우를 선형적(linear)이라고 함
- K-최근접 이웃 알고리즘
```
from sklearn.neighbors import KNeighborsClassifier # 사이킷런 패키지 임포트
kn = KNeighborsClassifier() # 객체 생성
kn.fit(fish_data, fish_target) # 주어진 데이터로 알고리즘 훈련(학습)
kn.score(fish_data, fish_target) # 1.0 -> 정확도 100% (성능평가)
```

2. 2-01 데이터 다루기.ipynb
- 샘플링 편향: 특정 종류의 샘플이 과도하게 많은 샘플링 편향을 가지고 있다면 제대로 된 지도 학습 모델을 만들 수 없음.
- 넘파이(numpy) 활용하여 훈련 세트와 테스트 세트를 랜덤하게 생성
```
np.random.seed(42)
index = np.arange(49) # 0부터 48까지 1씩 증가하는 배열 생성
np.random.shuffle(index) # 주어진 배열을 무작위로 섞음

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target)
```