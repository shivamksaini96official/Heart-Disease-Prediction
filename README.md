# Heart Disease Prediction

#### AIM : 
#### World Health Organization has estimated that four out of five cardiovascular disease (CVD) deaths are due to heart attacks. This whole research intends to pinpoint the ratio of patients who possess a good chance of being affected by CVD and also to predict the overall risk using Logistic Regression.

### Importing Libraries


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz # to export graph of decision tree to pdf
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
```

### Data Collection and Processing

**About the Dataset**

**Context :**
This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.

**Content :**

Attribute Information:

age  

sex  

chest pain type (4 values)  

resting blood pressure  

serum cholestoral in mg/dl  

fasting blood sugar > 120 mg/dl  

resting electrocardiographic results (values 0,1,2)  

maximum heart rate achieved  

exercise induced angina  

oldpeak = ST depression induced by exercise relative to rest  

the slope of the peak exercise ST segment  

number of major vessels (0-3) colored by flourosopy  
thal: 0 = normal; 1 = fixed defect; 2 = reversable defect  

The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.



```python
# loading the csv data to a pandas dataframe
heart_data = pd.read_csv('../DATA/heart.csv')
```


```python
# taking a look at the dataset
heart_data.head()
```





```python
# number of rows and columns in the dataset
heart_data.shape
```




    (1025, 14)




```python
# getting some info about the data
heart_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1025 entries, 0 to 1024
    Data columns (total 14 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       1025 non-null   int64  
     1   sex       1025 non-null   int64  
     2   cp        1025 non-null   int64  
     3   trestbps  1025 non-null   int64  
     4   chol      1025 non-null   int64  
     5   fbs       1025 non-null   int64  
     6   restecg   1025 non-null   int64  
     7   thalach   1025 non-null   int64  
     8   exang     1025 non-null   int64  
     9   oldpeak   1025 non-null   float64
     10  slope     1025 non-null   int64  
     11  ca        1025 non-null   int64  
     12  thal      1025 non-null   int64  
     13  target    1025 non-null   int64  
    dtypes: float64(1), int64(13)
    memory usage: 112.2 KB
    


```python
# checking for missing values
heart_data.isnull().sum()
```




    age         0
    sex         0
    cp          0
    trestbps    0
    chol        0
    fbs         0
    restecg     0
    thalach     0
    exang       0
    oldpeak     0
    slope       0
    ca          0
    thal        0
    target      0
    dtype: int64




```python
# Statistical measures about the data
heart_data.describe().transpose()
```




### Data Scaling


```python
# Checking the distribution of Target variable
heart_data['target'].value_counts()
```




    target
    1    526
    0    499
    Name: count, dtype: int64



1 &rarr; Defective Heart  
0 &rarr; Healthy Heart

*Splitting Features and Target*


```python
X = heart_data.drop('target',axis=1) # features
y = heart_data['target']             # target
```


```python
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X),columns = X.columns)
X
```



```python
y
```




    0       0
    1       0
    2       0
    3       0
    4       0
           ..
    1020    1
    1021    0
    1022    0
    1023    1
    1024    0
    Name: target, Length: 1025, dtype: int64




```python
# Splitting the Data into Training and Testing Data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y ,random_state=101)
```


```python
print('The Dimensions of the split: ')
print('X : ',X.shape)
print('X_train : ',X_train.shape)
print('X_test : ',X_test.shape)
```

    The Dimensions of the split: 
    X :  (1025, 13)
    X_train :  (820, 13)
    X_test :  (205, 13)
    

### Model Training and Evaluation

**Logistic Regression**


```python
logistic_reg = LogisticRegression(random_state=0)

# Training
logistic_reg.fit(X_train,y_train)
```




```python
# Accuracy on training data
X_train_prediction_1 = logistic_reg.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction_1,y_train)

print('Accuracy on training data: ',training_data_accuracy)
```

    Accuracy on training data:  0.8475609756097561
    


```python
# Accuracy on test data
X_test_prediction_1 = logistic_reg.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction_1,y_test)

print('Accuracy on test data: ',test_data_accuracy)
```

    Accuracy on test data:  0.8585365853658536
    


```python
# Confusion Matrix
print(classification_report(y_test,X_test_prediction_1))
```

                  precision    recall  f1-score   support
    
               0       0.91      0.79      0.84       100
               1       0.82      0.92      0.87       105
    
        accuracy                           0.86       205
       macro avg       0.87      0.86      0.86       205
    weighted avg       0.86      0.86      0.86       205
    
    

### Decision Tree Classifier


```python
dec_tree_clf = DecisionTreeClassifier(random_state=0,max_depth=5,min_samples_leaf=1,min_samples_split=5)
dec_tree_clf.fit(X_train,y_train) # fitting the data
```



```python
# Accuracy on the training data
X_train_prediction_2 = dec_tree_clf.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction_2,y_train)

print('Accuracy on training data: ',training_data_accuracy)
```

    Accuracy on training data:  0.9243902439024391
    


```python
# Accuracy on the test data
X_test_prediction_2 = dec_tree_clf.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction_2,y_test)

print('Accuracy on test data: ',test_data_accuracy)
```

    Accuracy on test data:  0.9414634146341463
    


```python
# Confusion Matrix
print(classification_report(y_test,X_test_prediction_2))
```

                  precision    recall  f1-score   support
    
               0       0.94      0.94      0.94       100
               1       0.94      0.94      0.94       105
    
        accuracy                           0.94       205
       macro avg       0.94      0.94      0.94       205
    weighted avg       0.94      0.94      0.94       205
    
    

### Random Forest Classifier (Best Accuracy)


```python
# creating object or instance
ran_for_clf = RandomForestClassifier(max_depth=6,random_state=0) 

# Fitting the data
ran_for_clf.fit(X_train,y_train)
```




```python
# Accuracy on the training data
X_train_prediction_3 = ran_for_clf.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction_3,y_train)

print('Accuracy on training data: ',training_data_accuracy)
```

    Accuracy on training data:  0.9804878048780488
    


```python
# Accuracy on the test data
X_test_prediction_3 = ran_for_clf.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction_3,y_test)

print('Accuracy on test data: ',test_data_accuracy)
```

    Accuracy on test data:  0.9853658536585366
    


```python
# Confusion Matrix
print(classification_report(y_test,X_test_prediction_3))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.97      0.98       100
               1       0.97      1.00      0.99       105
    
        accuracy                           0.99       205
       macro avg       0.99      0.98      0.99       205
    weighted avg       0.99      0.99      0.99       205
    
    

### Support Vector Classifier


```python
# Linear Kernel
svcLinear = SVC(kernel='linear',C=10000,gamma=0.001)
svcLinear.fit(X_train,y_train)
```



```python
# Accuracy on the training data
X_train_prediction_4 = svcLinear.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction_4,y_train)

print('Accuracy on training data: ',training_data_accuracy)
```

    Accuracy on training data:  0.8463414634146341
    


```python
# Accuracy on the test data
X_test_prediction_4 = svcLinear.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction_4,y_test)

print('Accuracy on test data: ',test_data_accuracy)
```

    Accuracy on test data:  0.8682926829268293
    


```python
# Sigmoid Kernel
svm = SVC(kernel='sigmoid',C=100000,gamma=0.005)
svm.fit(X_train,y_train)
```



```python
# Accuracy on the training data
X_train_prediction_4 = svm.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction_4,y_train)

print('Accuracy on training data: ',training_data_accuracy)
```

    Accuracy on training data:  0.801219512195122
    


```python
# Accuracy on the test data
X_test_prediction_4 = svm.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction_4,y_test)

print('Accuracy on test data: ',test_data_accuracy)
```

    Accuracy on test data:  0.824390243902439
    


```python
# Confusion Matrix
print(classification_report(y_test,X_test_prediction_4))
```

                  precision    recall  f1-score   support
    
               0       0.86      0.77      0.81       100
               1       0.80      0.88      0.84       105
    
        accuracy                           0.82       205
       macro avg       0.83      0.82      0.82       205
    weighted avg       0.83      0.82      0.82       205
    
    

### Grid-Search CV


```python
clf = SVC()
grid = {'C' : [1e2,1e3,5e3,1e4,5e4,1e5],
        'gamma' : [1e-3,5e-4,1e-4,5e-3]}
abc = GridSearchCV(clf,grid)
abc.fit(X_train,y_train)
```



```python
abc.best_estimator_
```




### KNN - K Nearest Neighbours


```python
# Creating an object or instance
knn = KNeighborsClassifier()

# Fitting the data or training the model
knn.fit(X_train,y_train)
```



```python
# Accuracy on the training data
X_train_prediction_5 = knn.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction_5,y_train)

print('Accuracy on training data: ',training_data_accuracy)
```

    Accuracy on training data:  0.9536585365853658
    


```python
# Accuracy on the test data
X_test_prediction_5 = knn.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction_5,y_test)

print('Accuracy on test data: ',test_data_accuracy)
```

    Accuracy on test data:  0.848780487804878
    


```python
# Confusion Matrix
print(classification_report(y_test,X_test_prediction_5))
```

                  precision    recall  f1-score   support
    
               0       0.82      0.89      0.85       100
               1       0.89      0.81      0.85       105
    
        accuracy                           0.85       205
       macro avg       0.85      0.85      0.85       205
    weighted avg       0.85      0.85      0.85       205
    
    

### Conclusion : The Random Forest is the most optimal model for the given dataset. 
