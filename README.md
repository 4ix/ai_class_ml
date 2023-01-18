# HG_MLDL
## 2023-01-18(수)
### 6-02 K 평균
1. 최적의 K 찾기
- 엘보우 방법: 적절한 클러스터 개수 찾기
- 이너셔(inertia): 클러스터 중심과 클러스터에 속한 샘플 사이의 거리. 클러스터에 속한 샘플이 얼마나 가깝게 모여 있는지.
```
inertia = []
for k in range(2,7):
  km = KMeans(n_clusters=k, random_state=42)
  km.fit(fruits_2d)
  inertia.append(km.inertia_)
plt.plot(range(2,7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()
```

### 6-03 주성분 분석
1. 차원과 차원 축소: 차원(특성)을 줄일 수 있다는 건 저장 공간 절약 가능성 있음

2. PCA 클래스(압축)
```
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(fruits_2d)

fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
```

3. 원본 데이터 재구성
```
fruits_inverse = pca.inverse_transform(fruits_pca) # 특성 복원
```

4. 설명된 분산
- 주성분이 원본 데이터의 분산을 얼마나 잘 나타내는지
```
print(np.sum(pca.explained_variance_ratio_)) # 원본 데이터의 92퍼센트를 유지
plt.plot(pca.explained_variance_ratio_) # 처음 10개의 주성분이 대부분의 분산을 표현
```


## 2023-01-17(화)
### 5-03 트리의 앙상블
1. 랜덤 포레스트
- 랜덤하게 선택한 샘플과 특성을 사용
- 훈련세트 과대적합 예방
- 검증 세트와 테스트 세트에서 안정적인 성능 기대
```
# 교차 검증 수행
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 랜덤 포레스트의 특성 중요도는 각 결정 트리의 특성 중요도를 취합한 것
rf.fit(train_input, train_target)
print(rf.feature_importances_) # 기존 결정 트리와 비교해서 중요도가 변한 이유? 랜덤 포레스트가 특성의 일부를 랜덤하게 선택하여 결정 트리를 훈련하기 때문 -> 과대적합을 줄이고 일반화 성능 높일 수 있음

# 랜덤 포레스트는 훈련 세트에서 중복을 허용하여 부트스트랩 샘플을 만들어 결정 트리를 훈련함
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
rf.fit(train_input, train_target)
print(rf.oob_score_) # OOB 점수를 사용하면 교차 검증을 대신할 수 있음 -> 훈련 세트에 더 많은 샘플 사용 가능
```

2. 엑스트라 트리
- 부트스트랩 샘플 사용 안함
- 노드를 무작위로 분할
- 많은 트리를 앙상블 하기 때문에 과대적합 예방 및 검증 세트 점수 높이는 효과

3. 그레이디언트 부스팅
- 깊이가 얕은 결정트리 사용, 이전 트리의 오차 보완
- 과대적합에 강하고 높은 일반화 성능 기대
- 경사 하강법의 가장 낮은 곳을 찾아 내려오는 방법은 모델의 가중치와 절편을 조금씩 바꾸는 것
```
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42) # 결정트리 개수 500으로 설정. 기본 100.
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

4. 히스토그램 기반 그레이디언트 부스팅
- 입력 특성(샘플)을 256개 구간으로 나눔
- 성능을 높이려면 max_iter 매개변수 테스트
```
# 중요도 계산 (importances 특성 중요도, importances_mean 평균, importances_std 표준편차)
from sklearn.inspection import permutation_importance

hgb.fit(train_input, train_target)
result=permutation_importance(hgb, train_input, train_target, n_repeats=10, random_state=42, n_jobs=-1) # 랜덤하게 10회 섞음
print(result.importances_mean)
```

### 6-01 군집 알고리즘
1. 픽셀값 분석하기

2. 평균값과 가까운 사진 고르기

### 6-02 K 평균
1. 클러스터 중심
```
draw_fruits(km.cluster_centers_.reshape(-1,100,100),ratio=3) # 클러스터 중심은 cluster_centers_ 속성에 저장되어 있음

# 훈련 데이터 샘플에서 클러스터 중심까지 거리
print(km.transform(fruits_2d[100:101]))

# 가장 가까운 클러스터 중심을 예측으로 출력
print(km.predict(fruits_2d[100:101]))
```

## 2023-01-16(월)
### 4-02 확률적 경사 하강법 (충분히 반복하여 훈련하면 훈련 세트에서 높은 점수를 얻는 모델을 만들 수 있음)
1. 에포크와 과대/과소적합
- 조기 종료: 과대 적합이 시작하기 전에 훈련을 멈추는 것
```
for _ in range(300): # 적절한 에포크 횟수 찾기 위한 반복문
  sc.partial_fit(train_scaled, train_target, classes = classes)
  train_score.append(sc.score(train_scaled, train_target))
  test_score.append(sc.score(test_scaled, test_target))

sc = SGDClassifier(loss='log', max_iter=100, tol=None, random_state=42) # tol 매개변수를 None으로 지정하여 자동으로 멈추지 않고 max_iter=100 만큼 무조건 반복되도록 함
```

### 5-01 결정 트리
1. 로지스틱 회귀로 와인 분류하기
```
# 1. train_test_split
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# 2. 훈련세트 전처리, 테스트 세트 변환
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 3. 로지스틱 회귀 모델 훈련
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
```

2. 결정 트리
```
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
```

3.  지니 불순도(gini)
- 루트 노드에서 당도 -0.239 기준으로 나온 이유 → ★ 각 특성들의 정보 이득(Information Gain) 숫자가 제일 작은 특성이 가장 루트 노드 기준으로 감
- 불순도 기준을 사용해 정보 이득이 최대가 되도록 노드 분할
- 마지막에 도달한 노드의 클래스 비율을 보고 예측

4. 가지치기
- 자라날 수 있는 트리의 최대 깊이를 지정

### 5-02 교차 검증과 그리드 서치
1. 검증 세트
- 테스트 세트를 사용하지 않고 이를 측정하는 간단한 방법은 훈련 세트를 나누는 것
```
# 1. 훈련 세트와 테스트 세트 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data,target,test_size=0.2, random_state=42)

# 1-2. 훈련세트를 검증 세트로 나누기
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
```

2. 교차 검증
- 검증 세트를 떼어 내어 평가하는 과정을 여러 번 반복 후 이 점수를 평균하여 최종 검증 점수를 얻음
```
# cross_validate() 교차 검증 함수
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)

# 교차 검증의 최종 점수
print(np.mean(scores['test_score']))

# cross_validate()는 훈련 세트를 섞어 폴드를 나누지 않음.
from sklearn.model_selection import StratifiedKFold
scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold()) # 교차 검증에서 폴드를 어떻게 나눌지 결정해줌

# 훈련 세트를 섞은 후 10-폴드 교차 검증 수행 방법
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
```

3. 하이퍼파라미터 튜닝
- GridSearchCV 클래스로 하이퍼파라미터 탐색과 교차 검증을 한번에 수행
```
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease':[0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1) # 객체 생성
gs.fit(train_input, train_target) # 학습
```

- 훈련 모델 중에서 검증 점수가 가장 높은 모델의 매개변수 조합으로 전체 훈련 세트에서 자동으로 다시 모델 훈련
```
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
```
- 최적의 매개변수는 best_params_ 속성에 저장
```
print(gs.best_params_)
```

- 각 매개변수에서 수행한 교차 검증의 평균 점수는 cv_results_에 저장
```
print(gs.cv_results_['mean_test_score']) # 5-fold의 테스트 평균값
```

- 최상의 매개변수 조합 확인
```
print(gs.best_params_)
```

- 최상의 교차 검증 점수 확인
```
print(np.max(gs.cv_results_['mean_test_score']))
```

4. 랜덤 서치
- 매개변수 값의 목록을 전달하는 것이 아니라 매개변수를 샘플링할 수 있는 확률분포 객체 전달
- 싸이파이(scipy): 수치 계산 전용 라이브러리
```
from scipy.stats import uniform, randint

# randint 정수
rgen = randint(0,10) # 0 ~ 10 사이의 숫자중에
rgen.rvs(10) # 10개를 뽑아낸다

# uniform 실수
ugen = uniform(0,1)
ugen.rvs(10)
```

- 샘플링 횟수는 RandomizedSearchCV의 n_iter 매개변수에 지정
```
from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)
```


## 2023-01-13(금)
### 4-01 로지스틱 회귀
1. k-최근접 이웃 분류기의 확률 예측
- 다중 분류: 타깃 데이터에 2개 이상의 클래스가 포함된 문제
```
print(kn.classes_) # 타깃값을 사이킷런 모델에 전달하면 순서가 자동으로 알파벳 순으로 정렬됨
print(kn.predict(test_scaled[:5])) # 테스트 세트에 있는 처음 5개 샘플 값 예측

proba = kn.predict_proba(test_scaled[:5]) # 클래스에 대한 확률 구하기. predict_proba. predict_proba의 출력은 항상 0과 1 사이의 값이며 두 클래스에 대한 확률의 합은 항상 1임.
```
[지도 학습 - 분류 예측의 불확실성 추정](https://subinium.github.io/MLwithPython-2-4/)

2. 로지스틱 회귀
- 로지스틱 회귀로 이진 분류 수행
```
# 넘파이 불리언 인덱싱
char_arr = np.array(['A','B','C','D','E'])
print(char_arr[[True, False, True, False, False]])

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

print(lr.predict_proba(train_bream_smelt[:5])) # 예측 확률, 첫번째 열이 0에 대한 확률, 두번째 열이 1에 대한 확률 (2개의 생선(이진)으로 학습했으므로 2개 열 리턴)

decisions = lr.decision_function(train_bream_smelt[:5]) # z값 계산

from scipy.special import expit # 시그모이드 함수(expit, z값을 넣으면 확률을 얻을 수 있음)
print(expit(decisions)) # proba의 두번째 열과 값이 동일. 양성(1) 클래스에 대한 z값 리턴
```

- 로지스틱 회귀로 다중 분류 수행하기
```
lr = LogisticRegression(C=20, max_iter=1000) # max_iter로 반복횟수 지정. 기본값 100. 규제 제어 매개변수 C. 작을수록 규제가 커짐. 기본값 1.
lr.fit(train_scaled, train_target) # 7개의 생선 데이터가 모두 들어있는것으로 학습

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3)) # 7개 생선 데이터로 학습했기 때문에 7개의 열이 리턴(샘플마다 클래스 개수만큼 확률 출력)

print(lr.coef_.shape, lr.intercept_.shape) # 5개의 특성을 사용하므로 coef_ 배열의 열은 5개
```

- 다중 분류는 클래스마다 z 값을 하나씩 계산
- 소프트맥스 함수를 사용하여 7개의 z값을 확률로 변환
```
decision = lr.decision_function(test_scaled[:5]) # z값 구하기
print(np.round(decision, decimals=2))

from scipy.special import softmax
proba = softmax(decision, axis=1) # axis=1, 행(샘플)에 대해 지정
print(np.round(proba, decimals=3))
```

### 4-02 확률적 경사 하강법
1. 점진적인 학습
- 확률적이란 말은 '무작위하게' 혹은 '랜덤하게' 라는 뜻
- 훈련 세트에서 랜덤하게 하나의 샘플을 고르는 것
- 에포크: 훈련 세트를 한 번 모두 사용하는 과정
- 미니배치 경사 하강법: 여러 개의 샘플을 사용해 경사 하강법을 수행하는 방식
- 배치 경사 하강법: 전체 샘플을 사용하는 방법(컴퓨터 자원을 많이 사용함)

2. 손실 함수: 머신러닝 알고리즘이 얼마나 엉터리인지를 측정하는 기준

3. 로지스틱 손실 함수
- 로지스틱 손실 함수: 이진 분류에서 사용하는 손실 함수
- 크로스엔트로피 손실 함수: 다중 분류에서 사용하는 손실 함수
- 평균 제곱 오차: 타깃에서 예측을 뺀 값을 제곱한 다음 모든 샘플에 평균한 값. 작을 수록 좋은 모델

4. SGDClassifier
- 확률적 경사 하강법 클래스 = SGDClassifier
```
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log', max_iter=10, random_state=42) # loss=log, 로지스틱 손실 함수, max_iter 수행할 에포크 횟수
sc.fit(train_scaled, train_target)
sc.partial_fit(train_scaled, train_target) # 1 에포크씩 이어서 훈련 가능함.
```


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