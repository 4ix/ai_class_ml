# HG_MLDL
## 2023-01-12(목)
### 3-02 선형 회귀
1. 선형 회귀
```
from sklearn.linear_model import LinearRegression # 임포트
lr = LinearRegression() # 객체생성
lr.fit(train_input, train_target) # 선형 회귀 모델 훈련
print(lr.predict([[50]])) # 50에 대한 예측
print(lr.coef_, lr.intercept_) #coef_: 기울기, 가중치
```

2. 다항 회귀(다항식을 사용한 선형 회귀)
```
train_poly = np.column_stack((train_input**2, train_input))
test_poly = np.column_stack((test_input**2, test_input))
lr = LinearRegression()
lr.fit(train_poly, train_target) # 목표하는 값은 어떤 그래프를 훈련하든지 바꿀 필요 없음
print(lr.predict([[50**2, 50]]))

```
- 가장 잘 맞는 직선의 방정식을 찾는다는 것은 최적의 기울기(coef_)와 절편(intercept_)을 구한다는 의미.
- 음수 리턴 해결하기 위해 다항 회귀 사용(2차 방정식)

### 3-03 특성 공학과 규제
1. 다중 회귀(여러 개의 특성을 사용한 선형 회귀)
- 특성 공학: 기존 특성을 사용해 새로운 특성을 뽑아내는 작업

2. 사이킷런의 변환기
```
from sklearn.preprocessing import PolynomialFeatures # 임포트
poly = PolynomialFeatures() # 객체 생성
poly.fit([[2,3]]) # 훈련(학습)
print(poly.transform([[2,3]])) # 변형
```
- PolynomialFeatures 클래스는 기본적으로 각 특성을 제곱한 항을 추가하고 특성끼리 곱한 항을 추가함.

```

poly = PolynomialFeatures(include_bias=False) # 절편을 위한 항(1) 제거 후 제곱과 특성끼리 곱한 항만 리턴
...
poly.get_feature_names_out() #9개의 특성이 어떻게 만들어졌는지 확인
```

3. 다중 회귀 모델 훈련하기
- PolynomialFeatures에 degree를 지정해 줄 수 있음
```
poly2 = PolynomialFeatures(degree=5, include_bias=False)
print(lr.score(train_poly2, train_target)) # 특성의 개수가 크게 늘어났기 때문에 훈련 세트에 대해 거의 완벽하게 학습
print(lr.score(test_poly2, test_target)) # 단, 테스트 세트에서는 형편없는 점수를 만듬 (샘플의 개수가 특성의 수보다 적어서)
```

4. 규제
- StandardScaler 사용
```
from sklearn.preprocessing import StandardScaler
ss = StandardScaler() # 객체 생성
ss.fit(train_poly2) # 학습
train_scaled = ss.transform(train_poly2) # 변형
test_scaled = ss.transform(test_poly2) # 변형 (훈련 세트로 학습한 변환기를 사용해 테스트 세트까지 변환해야 함)
```

5. 릿지 회귀
```
from sklearn.linear_model import Ridge
ridge = Ridge() # 객체생성
ridge.fit(train_scaled, train_target) #학습
print(ridge.score(train_scaled, train_target)) # 훈련 세트 평가
print(ridge.score(test_scaled, test_target)) # 테스트 세트 평가
```
- 객체를 만들 때 alpha 매개변수로 규제의 강도 조절
- alpha 값이 크면 규제 강도가 세지므로 계수 값을 더 줄여 조금 더 과소적합되도록 유도
- alpha 값이 작으면 계수를 줄이는 역할이 줄어들기 대문에 과대적합될 가능성이 큼
- 하이퍼파라미터: 사람이 직접 지정해야 하는 매개변수 (학습변수)

```
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
  ridge = Ridge(alpha=alpha)
  ridge.fit(train_scaled, train_target)
  train_score.append(ridge.score(train_scaled, train_target))
  test_score.append(ridge.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list), train_score) # alpha_list에 있는 6개의 값을 동일한 간격으로 나타내기 위해 로그 함수로 바꾸어 지수로 표현

ridge = Ridge(alpha=0.1) # 최적의 알파값 선정 후 대입
```

6. 라쏘 회귀
```
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

print(np.sum(lasso.coef_==0)) # 라쏘 모델은 계수 값을 아예 0으로 만들 수 있음, 0의 개수 체크
print(lasso.coef_)
```
- 55개의 특성을 모델에 주입했지만 라쏘 모델이 사용한 특성은 15개임 (55-40)
- 라쏘 모델은 유용한 특성을 골라내는 용도로도 사용할 수 있음


## 2023-01-11(수)
### 2-02 데이터 전처리.ipynb(스케일이 다른 특성 처리)
1. 넘파이 함수
```
np.column_stack(([1,2,3],[4,5,6])) #튜플 형태로 결합합
print(np.ones(5)) #1을 채운 배열
print(np.zeros(5)) #0을 채운 배열
fish_target = np.concatenate((np.ones(35), np.zeros(14))) # 첫 번째 차원을 따라 배열 연결

# 표준점수(각 특성값이 평균에서 표준편차의 몇 배 만큼 떨어져 있는지)로 바꾸기
mean = np.mean(train_input, axis=0) # 산술평균(가중치 x)
std = np.std(train_input, axis=0) # 표준편차
```

2. train_test_split
```
from sklearn.model_selection import train_test_split # 전달되는 리스트나 배열을 비율에 맞게 훈련 세트와 테스트 세트로 나누어줌
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)
```

3. 무작위로 데이터를 나누었을 때 샘플이 골고루 섞이지 않을 수 있음.
```
# stratify 매개변수에 타깃 데이터를 전달하면 클래스 비율에 맞게 데이터를 나눔.
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
distance, indexes = kn.kneighbors([[25,150]]) # 주어진 샘플에서 가장 가까운 이웃을 찾아주는 메서드
```

4. 두 특성(길이와 무게)의 값이 놓인 범위가 매우 다름. 이를 두 특성의 스케일이 다르다고 말함. 특성값을 일정한 기준으로 맞춰 주어야 하는데 이런 작업을 데이터 전처리 라고 함.
```
train_scaled = (train_input - mean) /std # 원본 데이터에서 평균을 빼고 표준편차로 나누어 표준점수로 변환

```

### 3-01 k-최근접 이웃 회귀.ipynb(회귀 문제 다루기)
1. 훈련 세트와 테스트 세트로 나누기
```
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
```

2. 사이킷런에 사용할 훈련세트는 2차원 배열어야 함 reshape() 사용
```
train_input = train_input.reshape(-1, 1) # 크기에 -1을 지정하면 나머지 원소 개수로 모두 채우라는 의미
test_input = test_input.reshape(-1, 1)
```

3. 결정 계수(회귀 모델 훈련)
```
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor() # 객체 생성
knr.fit(train_input, train_target) # 회귀 모델 훈련
```

4. 타깃과 예측한 값 사이의 차이 구하기
```
from sklearn.metrics import mean_absolute_error
test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction) # 타깃과 예측의 절대값 오차를 평균하여 리턴
```

5. 과대적합 vs 과소적합
- 과대적합: 훈련 세트에만 잘 맞는 모델 (훈련 세트 점수가 테스트 세트 점수 보다 높다), 새로운 샘플에 대한 예측을 만들 때 잘 동작 안할 수 있음.
- 과소적합: 모델이 너무 단순하여 훈련 세트에 적절히 훈련되지 않은 경우. (테스트 세트 점수가 훈련 세트 점수보다 높거나 두 점수가 모두 낮다.)
- 과소적합 시, 이웃(n_neighbors)의 개수를 줄이면 훈련 세트에 있는 국지적인 패턴에 민감해짐 (테스트 세트의 점수를 낮출수 있음)

### 3-02 선형 회귀
1. k-최근접 이웃의 한계
- k-최근접 이웃 회귀는 가장 가까운 샘플을 찾아 타깃을 평균 리턴함
- 새로운 샘플이 훈련 세트의 범위를 벗어나면 엉뚱한 값을 예측할 수 있음

## 2023-01-10(화)
### BreamAndSmelt.ipynb
1. 산점도 : x와 y의 상관관계 표현(산점도 그래프가 일직선에 가까운 형태로 나타나는 경우를 선형적(linear)이라고 함)
2. K-최근접 이웃 알고리즘
```
from sklearn.neighbors import KNeighborsClassifier # 사이킷런 패키지 임포트
kn = KNeighborsClassifier() # 객체 생성
kn.fit(fish_data, fish_target) # 주어진 데이터로 알고리즘 훈련(학습)
kn.score(fish_data, fish_target) # 1.0 -> 정확도 100% (성능평가)
```

### 2-01 데이터 다루기.ipynb
1. 샘플링 편향: 특정 종류의 샘플이 과도하게 많은 샘플링 편향을 가지고 있다면 제대로 된 지도 학습 모델을 만들 수 없음.
2. 넘파이(numpy) 활용하여 훈련 세트와 테스트 세트를 랜덤하게 생성
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