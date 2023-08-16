### 1. 분류(Classification) - 이산형(범주형), 데이터전처리(클래스 불균형 처리)  
### 1.1 이진 분류(Binary Classification) - Ex) 참/거짓, 긍/부정, 고객 이탈예측 등  
#### 1.1.1 ML 알고리즘   
로지스틱 회귀(Logistic Regression)  
서포트 벡터 머신(Support Vector Machine, SVM)  
나이브 베이즈 분류기(Naive Bayes Classifier)  
랜덤 포레스트 분류(Random Forest)  
#### 1.1.2 코드
```
from sklearn.linear_model import LogisticRegression  
model_logistic = LogisticRegression()  
from sklearn import svm
model_svm = svm.SVC()
from sklearn.ensemble import RandomForestClassifier model_random_forest = RandomForestClassifier()
from sklearn.naive_bayes import GaussianNB
```
### 1.2 다중 범주 분류(Multiclass Classification) - Ex) 동물의 종류, 고객 세분화 등   
#### 1.2.1 ML 알고리즘    
로지스틱 회귀(Logistic Regression)   
결정 트리(Decision Tree)   
나이브 베이즈 분류기(Naive Bayes Classifier)   
#### 1.2.2 코드    
```
from sklearn.tree import DecisionTreeClassifier model_decision_tree = DecisionTreeClassifier()
from sklearn.linear_model import LogisticRegression
model_logistic = LogisticRegression(multi_class='multinomial')
from sklearn.naive_bayes import GaussianNB
model_naive_bayes = GaussianNB()
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier()
```

![image](https://github.com/SteveSEK/my-chatgpt-prompts/assets/2126804/b161acd6-a95e-4788-9a6c-5e629daef27c)



