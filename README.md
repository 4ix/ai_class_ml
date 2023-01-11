# HG_MLDL
## 2023-01-11(수)
1. 2-02 데이터 전처리.ipynb
- ★ 스케일이 다른 특성 처리
- 넘파이 함수
```
np.column_stack(([1,2,3],[4,5,6])) #튜플 형태로 결합합
print(np.ones(5)) #1을 채운 배열
print(np.zeros(5)) #0을 채운 배열
fish_target = np.concatenate((np.ones(35), np.zeros(14))) # 첫 번째 차원을 따라 배열 연결

# 표준점수(각 특성값이 평균에서 표준편차의 몇 배 만큼 떨어져 있는지)로 바꾸기
mean = np.mean(train_input, axis=0) # 산술평균(가중치 x)
std = np.std(train_input, axis=0) # 표준편차
```
- train_test_split
```
from sklearn.model_selection import train_test_split # 전달되는 리스트나 배열을 비율에 맞게 훈련 세트와 테스트 세트로 나누어줌
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)
```
- 무작위로 데이터를 나누었을 때 샘플이 골고루 섞이지 않을 수 있음.
```
# stratify 매개변수에 타깃 데이터를 전달하면 클래스 비율에 맞게 데이터를 나눔.
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
```
distance, indexes = kn.kneighbors([[25,150]]) # 주어진 샘플에서 가장 가까운 이웃을 찾아주는 메서드
```
- 두 특성(길이와 무게)의 값이 놓인 범위가 매우 다름. 이를 두 특성의 스케일이 다르다고 말함.
- 특성값을 일정한 기준으로 맞춰 주어야 하는데 이런 작업을 데이터 전처리 라고 함.
```
train_scaled = (train_input - mean) /std # 원본 데이터에서 평균을 빼고 표준편차로 나누어 표준점수로 변환

```

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