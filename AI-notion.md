#### 1. 필요한 라이브러리 설치 및 임포트
```
#가장 많이 사용하는 라이브러리 설치 - 별칭(alias) 사용
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings(action='ignore')

#seaborn import시 에러가 나타나면 해당 코드 실행
!pip install seaborn

#AIDU 내부 연동을 위한 라이브러리 & 변수 
from aicentro.session import Session
from aicentro.framework.keras import Keras as AiduFrm

aidu_session = Session(verify=False)
aidu_framework = AiduFrm(session=aidu_session)
```

#### 2. 데이터 로드
```
df = pd.read_csv('파일경로')
df = pd.read_csv('/data/데이터파일이름.csv')
df = pd.read_csv("https://raw.githubusercontent.com/DA4BAM/dataset/master/mobile_NA2.csv")
```

#### 3. 데이터 분석(구성확인, 상관분석, 시각화)
```
#데이터 전체 구조 보기
df

#데이터 정보 확인
df.info()

#데이터 (행,열) 크기 확인
df.shape

#데이터 전체 컬럼명 보기
df.columns

#데이터 상위/하위 5행 출력
df.head()
df.tail()

#데이터 통계 보기
df.describe()

#중앙값(숫자형)
df['INCOME'].median()
df.median()

#컬럼 내 각각의 값 분포 -> 제일 위 값이 최빈값
df['INCOME'].value_counts()
#전체
#[df[c].value_counts() for c in df]
#특정 컬럼 내 각각의 값 분포 비율
df['REPORTED_SATISFACTION'].value_counts(normalize=True)

#특정 컬럼 내 유일한 값 확인
df['REPORTED_SATISFACTION'].unique()

#데이터 결측치(null값) 확인
df.isna().sum()

#데이터 타입 확인
df.dtypes

#두 변수간 상관 관계 분석
df.corr()

# 레이블 선택
y = df['OVERAGE']
y
```
```
### 시각화 ###

import matplotlib.pyplot as plt
pip install seaborn
import seaborn as sns
%matplotlib inline

#차트 그리기
plt.title('Accuracy')
sns.plot(hist.history['acc'], label='acc')
sns.plot(hist.history['val_acc'], label='val_acc')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.show()

#산점도(Scatter plot)
sns.scatter(x, y)

#막대 그래프 그리기
sns.bar(x, y)

# 종속변수 시각화
sns.displot(cal['Calories_Burned'])
plt.show()

#히스토그램 그리기
sns.hist(values)

#히트맵(Heatmap)
sns.heatmap(df.corr(), cmap="Blues", annot=True)
sns.heatmap(df.corr())

#박스 플롯(Box plot)
sns.boxplot(df['HANDSET_PRICE'])
plt.show()
sns.boxplot(y='AVERAGE_CALL_DURATION', x='CHURN', data=df)

#pairplot
sns.pairplot(data=df, x_vars=['컬럼', '컬럼', '컬럼'], y_vars=['컬럼', '컬럼', '컬럼'])

# Pie plot
plt.figure(figsize=(6, 6))
data['Survived'].value_counts().plot.pie(autopct='%1.1f%%', shadow=True)
plt.title('Survived')
plt.ylabel('')
plt.show()

```

#### 4. 데이터 전처리 (결측치 처리, 라벨인코딩 등)
```
#특정 컬럼 삭제 -> axis=0 행 삭제 / axis=1 열 삭제
df1 = df.drop(['id', 'COLLEGE', 'LEFTOVER'], axis=1)

#값 변경 -> 인자(변경 전 값, 변경 후 값, inplace=True)
df1['REPORTED_USAGE_LEVEL'].replace('avg', 'middle', inplace=True)

#특정 값이 있는 행만 가져오기
df1[df1['OVERAGE'] == 10]

#특정 값의 개수 확인
(df['REPORTED_SATISFACTION'] == 'very_unsat').sum()

#전체 값의 개수 확인
df1.apply(lambda x: x.isin([0]).sum())

## 결측치 처리 ##

#데이터 결측치(null값) 확인
df.isna().sum()

#결측치 중간값으로 채우기 -> mean : 평균, mode : 최빈값
df['HOUSE'].fillna(df['HOUSE'].mean, inplace=True)

#결측치 삭제하기
df1 = df1.dropna()

## 이상치 처리 ##
new_data1 = data1.copy()
new_data1.drop(new_data1[data1['subscribers']>2*1e9].index, inplace=True)
new_data1.drop(new_data1[data1['video views']>2*1e11].index, inplace=True)
new_data1.info()

#이상치 데이터 확인
sns.boxplot(x='CHURN', y='LEFTOVER', data=df)
```
```
## 라벨 인코딩 ##

#데이터 복사
df1_copy = df.copy()

#데이터 타입 변경
df1['SeniorCitizen'] = df1['SeniorCitizen'].astype(float)

#특정 데이터 타입의 컬럼 선택
c = df1.select_dtypes(include='object')
c.dtypes

#문자를 숫자로 변경(라벨 인코딩)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['REPORTED_USAGE_LEVEL'] = le.fit_transform(df['REPORTED_USAGE_LEVEL'])
df['REPORTED_USAGE_LEVEL'] = df['REPORTED_USAGE_LEVEL'].astype('float')

#문자를 숫자로 변경(라벨 인코딩)
from sklearn.preprocessing import LabelEncoder 
tips['sex'] = le.fit_transform(tips['sex'])​

labels = ['col1', 'col2', 'col3'] ​
for i in labels: ​
    le = LabelEncoder() ​
    le = le.fit(price[i]) ​
    price[i] = le.transform(price[i])​
```

```
## 원-핫 인코딩 ##

# (1) 문자를 숫자로 변경 (원-핫 인코딩)
# drop_first=True  첫번째 카테고리(인코딩 데이터) 삭제
cols = df.select_dtypes('object').columns.tolist()
df = pd.get_dummies(columns=cols, data=df, drop_first=True)

# (2) 
df=pd.get_dummies(df,columns=['col'],drop_first=True)​
tips=pd.get_dummies(tips,columns=['day'],drop_first=True)​
```

#### 5. 데이터 분리 (x, y)
```
# (1)
from sklearn.model_selection import train_test_split
X = tips.drop('total_bill',axis=1) ​
y = tips['total_bill’]   # series ​
X_train,X_valid,y_train, y_valid = ​train_test_split(X,y, random_state=58,test_size=0.2)​

# (2) Feature(X), Target(Y) 분리. 학습데이터(train set)와 검증데이터(test set)로 분리
target = 'CHURN'
x = df.drop(target, axis=1)
y = df[target]

from sklearn.model_selection import train_test_split
#test_size는 원래 데이터(Y)의 분포 비율
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2023)
```

#### 6. 데이터 스케일링 (정규화)
```
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#정규화 : 최대값 1, 최소값 0
scaler = MinMaxScaler()
#표준화 : 평균값 0, 표준편차 1
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

#### 7. 머신러닝
```
### 회귀 ###

# linear회귀
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error​
mse = mean_squared_error(y_valid, y_pred)​
from sklearn.metrics import mean_absolute_error​
mae = mean_absolute_error(y_valid, y_pred) ​

#로지스틱 회귀
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error​
mse = mean_squared_error(y_valid, y_pred)​
from sklearn.metrics import mean_absolute_error​
mae = mean_absolute_error(y_valid, y_pred) ​
```

```
### 분류 ###

#의사결정트리
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

#랜덤포레스트
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
```

#### 8. 딥러닝
```
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#데이터의 행, 열 개수 찾기
x_train.shape

#이진 분류 모델 생성
clear_session()
model = Sequential([
    Input(shape=(18,)),	#input shape : 입력데이터의 shape(열의 개수) 반드시 명시
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
    #다중 분류
    #Dense('최종output 레이어 개수', activation='softmax')
])
model.summary()

#모델 컴파일 optimizer 설정 -> loss:손실함수, metrics:평가기준
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#다중분류
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc']) #원핫인코딩된 경우
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc']) #원핫인코딩 안된 경우
#회귀
#model.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])

#callback 함수 설정
es = EarlyStopping(monitor='val_loss', patience=4, mode='min', verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

#학습하기
hist = model.fit(x_train, y_train,
                batch_size=32,
                epochs=100,
                callbacks=[es, mc],
                validation_data=(x_test, y_test),
                verbose=1)

```

```
#Accurracy 그래프 그리기
plt.title('Accuracy')
plt.plot(hist.history['acc'], label='acc')
plt.plot(hist.history['val_acc'], label='val_acc')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.show()

#Loss 그래프 그리기
plt.title('Loss')
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.show()

#두개 다 합친 그래프
plt.title('Accuracy')
plt.plot(hist.history['acc'], label='acc')
plt.plot(hist.history['val_acc'], label='val_acc')
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.show()
```

#### 9. 모델 성능평가
```
y_pred = model.predict(x_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Precision(정밀도)
from sklearn.metrics import precision_score
ps = precision_score(y_test, y_pred, pos_label=1)
print('Precision(정밀도): %.4f' % ps)

#Recall(정밀도)
from sklearn.metrics import recall_score
rs = recall_score(y_test, y_pred, pos_label=1)
print('Recall(재현율): %.4f' % rs)

#F1 Score
from sklearn.metrics import f1_score
fs = f1_score(y_test, y_pred, pos_label=1)
print('F1 Score: %.4f' % fs)

#Accuracy(정확도)
from sklearn.metrics import accuracy_score
accs = accuracy_score(y_test, y_pred, pos_label=1)
print('Accuracy(정확도): %.4f' % accs)

#Classification Report(평가지표-Precision, Recall, F1 한 번에 출력)
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred, target_names=['class 0', 'class 1'])
print(cr)
```
