#### 1. Library import​
```
import pandas as pd​
import numpy as np​
```
#### 2. Library import​
```
import matplot.pyplot as plt​
import seaborn as sns​	
```
#### 3. Data Load
```
df = pd.read_csv(‘filename.csv’)​
df = pd.read_json(A01.json)​
df = pd.DataFrame(data)​
df = pd.read_excel('data.xls’)​

tips = sns.load_dataset("tips")​
```
#### 4. Visualization
```
sns.pairplot(df, x_vars[‘col1’,’col2’], y_vars=‘col3’)​
sns.histplot(df,bins=30,kde=True)​
sns.scatterplot(data=df,x=‘col1’, y=‘col2’)​
```
```
​data = df.corr() ​
plt.figure(figsize=(8, 5))​
sns.heatmap(data=df,annot=True) plt.show()​
```
#### ​5. Delete outlier & missing values
```
df=df.drop(df[df['size']>100].index,axis=0)​
tips = tips.drop(tips[tips['size']==6].index, axis=0)​

df.drop('col',axis=1,inplace=True)​
tips=tips.drop(['sex','smoker'],axis=1)​

df = df.dropna(axis=0)​
```
#### 6. Data preparation
```
df.info() / head() ​
df.shape​
df.isnull().sum() / df.isna().sum() # 동일​
df.describe()​
df['col'].value_counts()​
df['col'].replace(' ', '0', inplace=True)​
tips = tips.head(10)​
```
#### 7. Label Encoder
```
from sklearn.preprocessing import LabelEncoder 
tips['sex'] = le.fit_transform(tips['sex'])​

labels = ['col1', 'col2', 'col3'] ​
for i in labels: ​
    le = LabelEncoder() ​
    le = le.fit(price[i]) ​
    price[i] = le.transform(price[i])​
```
#### 8. One-Hot Encoder
```
df=pd.get_dummies(df,columns=['col'],drop_first=True)​
tips=pd.get_dummies(tips,columns=['day'],drop_first=True)​
```
#### 9. Data Split
```
from sklearn.model_selection import train_test_split
X = tips.drop('total_bill',axis=1) ​
y = tips['total_bill’]   # series ​
X_train,X_valid,y_train, y_valid = ​train_test_split(X,y, random_state=58,test_size=0.2)​
```
#### 10. Standardization/Normalization
```
from sklearn.preprocessing import StandardScaler ​
scaler = StandardScaler() ​
X_train_scaled = scaler.fit_transform(X_train) ​
X_valid_scaled = scaler.transform(X_valid)​

from sklearn.preprocessing import MinMaxScaler​
minmax_scaler = MinMaxScaler()​
train_minmax = minmax_scaler.fit_transform(X_train)​
train_minmax = pd.DataFrame(train_minmax, index=X_train.index, columns=X_train.columns)​
```
#### 11. Machine Learning/Random Forest
```
from sklearn.ensemble import RandomForestRegressor ​
rf_model = RandomForestRegressor(random_state=42, n_estimators=70, max_depth=12, min_samples_split=3, min_samples_leaf=2)​
rf_model.fit(X_train_scaled,y_train)​
```
#### 12. MSE/MAE
```
y_pred = rf_model.predict(X_valid) ​

// MSE 계산 ​
from sklearn.metrics import mean_squared_error​
mse = mean_squared_error(y_valid, y_pred)​

// MAE 계산 ​
from sklearn.metrics import mean_absolute_error​
mae = mean_absolute_error(y_valid, y_pred) ​
```
​#### 13. Deep Learning Model
```
import tensorflow as tf​
from tensorflow.keras.models import Sequential​
from tensorflow.keras.layers import Dense, Dropout​
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint​

​model = Sequential()​
model.add(Dense(64, activation='relu', input_shape=(6,))) model.add(Dropout(0.2))  ​
model.add(Dense(128, activation='relu'))​
model.add(Dropout(0.2))  ​
model.add(Dense(256, activation='relu'))​
model.add(Dropout(0.2))  ​
model.add(Dense(512, activation='relu'))​
model.add(Dropout(0.2))  ​
model.add(Dense(1))  // 출력 레이어​

// Loss Function & Optimization Setting​
model.compile(loss='mean_squared_error', optimizer='adam')​

// EarlyStopping Callback​
estopping = EarlyStopping(monitor='val_loss', patience=5)​

// ModelCheckpoint Callback​
mcheckpoint = ModelCheckpoint('AI_best_model.h5', monitor='val_loss', save_best_only=True)​

// Training the model​
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, callbacks=[estopping, mcheckpoint])​
```
​#### 14. Model evaluation
```
import matplotlib.pyplot as plt​

// 학습 MSE와 검증 MSE 추출​
train_mse = history.history['loss']​
valid_mse = history.history['val_loss']​

// Graph​
plt.plot(train_mse, label='mse')​
plt.plot(valid_mse, label='val_mse')​

// Title, etc​
plt.legend()​
plt.title('Model MSE')​
plt.xlabel('Epochs')​
plt.ylabel('MSE')​

// Show the graph​
plt.show()​
```

 
