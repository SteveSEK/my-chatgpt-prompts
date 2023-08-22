### 다중분류

```
import pandasz as pd

drugs = pd.read_csv('drug200.csv')

# lineplot 그래프 생성
sns.lineplot(x='Age', y='Na_to_K', data=drugs)
# 그래프에 라벨 추가
plt.xlabel('Age')
plt.ylabel('Na_to_K')
# 그래프 출력
plt.show()


sns.boxplot(y='Na_to_K', data=drugs)
# 그래프 출력
plt.show()


new_drugs=drugs.drop(drugs[drugs['Age']>55].index, axis=0)
new_drugs=new_drugs.drop(new_drugs[new_drugs['Na_to_K']>30].index,axis=0)

new_drugs = drugs.drop(drugs[(drugs['Age'] > 55) | (drugs['Na_to_K'] > 30)].index)


#new_drugs.isnull().sum()
new_drugs = new_drugs.fillna(0)
#new_drugs.isnull().sum()
#new_drugs.info()


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
new_drugs['BP'] = encoder.fit_transform(new_drugs['BP'])
new_drugs['Cholesterol'] = encoder.fit_transform(new_drugs['Cholesterol'])
new_drugs['Drug'] = encoder.fit_transform(new_drugs['Drug'])
label_drugs = new_drugs

from sklearn.preprocessing import LabelEncoder 
label_drugs = new_drugs.copy()
le = LabelEncoder() 
labels = ['BP', 'Cholesterol', 'Drug'] 
for i in labels:     
    le = le.fit(new_drugs[i]) 
    label_drugs[i] = le.transform(new_drugs[i])


plt.figure(figsize=(8, 8))
sns.heatmap(label_drugs.corr(), cmap="Oranges", annot=True)
plt.show()

cols = label_drugs.select_dtypes('object').columns.tolist()
drugs_preset = pd.get_dummies(columns=cols, data=label_drugs, drop_first=True)

print(drugs_preset)


from sklearn.model_selection import train_test_split
X = drugs_preset.drop('Drug',axis=1) 
y = drugs_preset['Drug']
X_train,X_valid,y_train, y_valid = train_test_split(X, y, random_state=42,test_size=0.2)


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=30, max_depth=9, min_samples_split=3, min_samples_leaf=2, random_state=42)
rf_model.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=41)
dt_model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, f1_score
y_pred = rf_model.predict(X_valid)
rfr_acc = accuracy_score(y_valid, y_pred)
rfr_f1 = f1_score(y_valid, y_pred, average='weighted')
print('accuracy:',rfr_acc)
print('f1-score:',rfr_f1)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
num_classes=5
# 모델 설정
model = Sequential()
model.add(Dense(256, input_dim=5, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))  # num_classes는 분류하고자 하는 클래스의 수를 나타냅니다.
# 모델 컴파일
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# 콜백 설정
estopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
mcheckpoint = ModelCheckpoint('AI_best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
# 모델 학습
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=1000, batch_size=64, callbacks=[estopping, mcheckpoint])


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model acc')# 제목
plt.ylabel('Score') # score y축표시
plt.xlabel('Epochs') # score x축표시
plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='lower right') # 범례 표시
plt.show()
```
### Regression Random Forest
```

players = pd.read_csv("top5_leagues_player.csv")

sns.pairplot(data=players, y_vars="price", x_vars = ['age', 'shirt_nr', 'foot', 'league'])

new_players = players.drop(players[players['age']>35].index)
new_players = new_players.drop(new_players[new_players['shirt_nr']>50].index)
new_players
#new_players.info()

outlier = players[(players['age']>35) | (players['shirt_nr']>50)].index
new_players = players.drop(outlier)
new_players
#new_players.info()

plt.figure(figsize=(10, 10))
sns.heatmap(new_players.corr(), annot=True, cmap='coolwarm')
plt.show()

new_players.info()
clean_players = new_players.drop(['Unnamed: 0', 'name', 'full_name'], axis=1)
clean_players.info()


#clean_players.isnull().sum()
clean_players = clean_players.dropna()
#clean_players.isnull().sum()

clean_players.isnull().sum()
clean_players = clean_players.dropna()
#clean_players.info()
clean_players.isnull().sum()

clean_players.dropna(inplace=True)


from sklearn.preprocessing import LabelEncoder
label_players = clean_players.copy()
cols= ['nationality', 'place_of_birth', 'position', 'outfitter', 'club', 'player_agent', 'foot', 'joined_club']
le = LabelEncoder()
for col in cols:
    label_players[col] = le.fit_transform(label_players[col])
label_players.head()


from sklearn.preprocessing import LabelEncoder
label_players = clean_players.copy()
le = LabelEncoder()
label_players['nationality'] = le.fit_transform(label_players['nationality'])
label_players['position'] = le.fit_transform(label_players['position'])
label_players['outfitter'] = le.fit_transform(label_players['outfitter'])
label_players.head()

players_preset = pd.get_dummies(columns=['contract_expires', 'league'], data=label_players, drop_first=True)
players_preset.info()

from sklearn.model_selection import train_test_split
y = players_preset['price']
X = players_preset.drop("price", axis=1)
X_train, X_valid, y_train,y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled= scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=30, max_depth=9, random_state=42, min_samples_leaf=2, min_samples_split=3)
rf_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error,mean_absolute_error

y_pred = rf_model.predict(X_valid)

rfr_mae = mean_absolute_error(y_valid, y_pred)
rfr_mse = mean_squared_error(y_valid, y_pred)
print(rfr_mae, rfr_mse)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

estopping = EarlyStopping(monitor='val_loss')
mcheckpoint = ModelCheckpoint(monitor='val_loss', filepath='AI_best_model.h5', save_best_only=True)

history = model.fit(X_train_scaled, y_train, epochs=100, validation_data = (X_valid_scaled, y_valid), callbacks=[estopping,mcheckpoint])

plt.plot(history.history['loss'], 'y', label = 'mse')
plt.plot(history.history['val_loss'], 'r', label = 'val_mse')
plt.title("Model MSE")
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()
```
