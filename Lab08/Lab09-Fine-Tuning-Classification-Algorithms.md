# Lab 09 Fine-Tuning Classification Algorithms


## **Mục tiêu học tập**
Sau khi hoàn thành bài học này, học viên sẽ có thể:
- Hiểu rõ nhu cầu và tầm quan trọng của customer segmentation
- Nắm vững thuật toán K-means và ứng dụng trong phân khúc khách hàng
- Thực hiện phân tích thống kê mô tả và tổng hợp dữ liệu
- Sử dụng các công cụ Python để thực hiện segmentation
- Phân tích và diễn giải kết quả phân khúc khách hàng
- Áp dụng các kỹ thuật nâng cao trong customer segmentation

---

## **Bài tập Thực hành**
### Bài tập cơ bản

#### **Exercise 8.01: Training an SVM Algorithm Over a Dataset**
In this exercise, you will work with the Shill Bidding dataset, the file for which is 
named **Shill_Bidding_Dataset.csv**. This is the same dataset you were introduced to in _Exercise 7.01, Comparing Predictions by Linear and Logistic Regression_ on the Shill Bidding Dataset. Your objective is to use this information to predict whether an auction depicts normal behavior or not (**0** means normal behavior and **1** means abnormal behavior). 

_You will use the SVM algorithm to build your model:_

**Code:**

```python
# 1.	Import pandas, numpy, train_test_split, cross_val_score, and svm from the sklearn library:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np

# 2.	Read the dataset into a DataFrame named data using pandas, as shown in the following snippet,
# 		and look at the first few rows of the data:
data=pd.read_csv("Shill_Bidding_Dataset.csv")

# 3.	First, remove the columns that are irrelevant to the case study.
# 		These are ID columns and thus will be unique to every entry.
# 		Because of their uniqueness, they won't add any value to the model and thus can be dropped:
# 		Drop irrelevant columns
data.drop(["Record_ID","Auction_ID","Bidder_ID"],axis=1, inplace=True) 
data.head()

# 4.	Check the data types, as follows: 
data.dtypes

# 5.	Look for any missing values using the following code:
data.isnull().sum()   ### Check for missing values

# 6.	Split the data into train and test sets and save them as X_train, X_test,  y_train, and y_test as shown:
target = 'Class'
X = data.drop(target,axis=1)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X.values,y,test_size=0.50,random_state=123, stratify=y)

# 7.	Fit a linear SVM model with C=1:
clf_svm=svm.SVC(kernel='linear', C=1)
clf_svm.fit(X_train,y_train)

# 8.	Calculate the accuracy score using the following code:
clf_svm.score(X_test, y_test)

```

---

#### **Exercise 8.02: Implementing a Decision Tree Algorithm over a Dataset**

In this exercise, you will use decision trees to build a model over the same auction dataset that you used in the previous exercise. This practice of training different classifiers on the same dataset is very common whenever you are working on any classification task. Training multiple classifiers of different types makes it easier to pick the right classifier for a task.


**Code:**

```python
# 1.	Import tree, graphviz, StringIO, Image, export_graphviz, and pydotplus:
import graphviz
from sklearn import tree
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus 

# 2.	Fit the decision tree classifier using the following code:
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(X_train, y_train)

# 3.	Plot the decision tree using a graph. In this plot, you will be using  export_graphviz to visualize
# 		the decision tree. You will use the output of your decision tree classifier as your input clf.
# 		The target variable will be the class_names, that is, Normal or Abnormal. 
dot_data = StringIO()
export_graphviz(clf_tree, out_file=dot_data, filled=True, rounded=True, class_names=['Normal','Abnormal'], 
              max_depth = 3,  special_characters=True, feature_names=X.columns.values)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())

# 4. Calculate the accuracy score using the following code:
clf_tree.score(X_test, y_test)

```

---

#### **Exercise 8.03: Implementing a Random Forest Model over a Dataset**
In this exercise, you will use a random forest to build a model over the same auction dataset used previously. Ensure that you use the same Jupyter notebook as the one used for the preceding exercise:

**Code:**

```python
# 1.	Import the random forest classifier:
from sklearn.ensemble import RandomForestClassifier

# 2.	Fit the random forest classifier to the training data using the following code:
clf = RandomForestClassifier(n_estimators=20, max_depth=None, min_samples_split=7, random_state=0)
clf.fit(X_train,y_train)

# 3.	Calculate the accuracy score:
clf.score(X_test, y_test)

```

---

#### **Activity 8.01: Implementing Different Classification Algorithms**

In this activity, you will continue working with the telecom dataset  
(Telco_Churn_Data.csv) that you used in the previous chapter and will build different models from this dataset using the scikit-learn API. Your marketing team was impressed with the initial findings, and they now want you to build a machine learning model that can predict customer churn. This model will be used by the marketing team to send out discount coupons to customers who may churn. To build the best prediction model, it is important to try different algorithms and come up with the best-performing algorithm for the marketing team to use. In this activity, you will use the logistic regression, SVM, and random forest algorithms and compare the accuracies obtained from the three classifiers.

_Follow these steps:_

1.	Import the libraries for the logistic regression, decision tree, SVM, and random forest algorithms.

2.	Fit individual models to the clf_logistic, clf_svm, clf_decision, and clf_random variables.
Use the following parameters to ensure your results are more or less close to ours: for the logistic regression model, use random_state=0 and solver='lbfgs'; for the SVM, use kernel='linear' and C=1; and for the random forest model, use n_estimators=20, max_depth=None,  min_samples_split=7, and random_state=0.

3.	Use the score function to get the accuracy for each of the algorithms.
You should get accuracy scores similar to the ones listed in the following figure for each of the models at the end of this activity:


![Figure 8.33: Comparison of different algorithm accuracies on the telecom dataset](images/Figure-8.33.jpg)

**Code:**

```python

```

---

#### **Exercise 8.04: Standardizing Data**

For this exercise, you will use the bank churn prediction data that was used in Chapter 7, Supervised Learning: Predicting Customer Churn. In the previous chapter, you performed feature selection using a random forest. The features selected for your bank churn prediction data are **Age, EstimatedSalary, CreditScore, Balance, and NumOfProducts**.
In this exercise, your objective will be to standardize the data after you have carried out feature selection. On exploring the previous chapter, it was clear that data is not standardized; therefore in this exercise, you will implement StandardScalar to standardize the data to zero mean and unit variance. Ensure that you use the same notebook as the one used for the preceding two exercises. 

**Code:**

```python
# 1.	Import the preprocessing library:
from sklearn import preprocessing

# 2.	View the first five rows, which have the Age, EstimatedSalary, CreditScore, Balance, and NumOfProducts features:
X_train[top5_features].head()

# 3.	Fit the StandardScalar function on the X_train data using the following code:
scaler = preprocessing.StandardScaler()\
                      .fit(X_train[top5_features])

# 4.	Check the mean and scaled values. Use the following code to show the mean values of the five columns:
scaler.mean

# 5.	Now check the scaled values:
scaler.scale

# 6.	Apply the transform function to the X_train data. This function performs standardization by centering and
# 		scaling the training data:
X_train_scalar=scaler.transform(X_train[top5_features])

# 7.	Next, apply the transform function to the X_test data and check the output:
X_test_scalar=scaler.transform(X_test[top5_features])
X_train_scalar


# 6.	Create a variable named features to store all the columns, except the target Churn variable.
# 		Sort the important features present in
# 		the importances variable using NumPy's argsort function:
features = data.drop(['Churn'],axis=1).columns
indices = np.argsort(importances)[::-1]

# 7.	Plot the important features obtained from the random forest using Matplotlib's plt attribute:
plt.figure(figsize=(15,4))
plt.title("Feature importances using Random Forest")
plt.bar(range(X_train.shape[1]), importances[indices], color="gray", align="center")
plt.xticks(range(X_train.shape[1]), features[indices],  rotation='vertical',fontsize=15)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# 8. Place the features and their importance in a pandas DataFrame using the following code:
feature_importance_df = pd.DataFrame({"Feature":features, "Importance":importances})
print(feature_importance_df)

```
---

#### **Exercise 8.05: Scaling Data After Feature Selection**

In this exercise, your objective is to scale data after feature selection. You will use the same bank churn prediction data to perform scaling. Ensure that you continue using the same Jupyter notebook. You can refer to Figure 8.35 to examine the top five features:

![Figure 8.35: First few rows of top5_features](images/Figure-8.35.jpg)

**Code:**

```python
# 1.	Fit the min_max scaler on the training data:
min_max = preprocessing.MinMaxScaler().fit(X_train[top5_features])

# 2.	Check the minimum and scaled values:
min_max.min_

# 3.	Now check the scaled values:
min_max.scale

# 4.	Transform the train and test data using min_max:
X_train_min_max=min_max.transform(X_train[top5_features])
X_test_min_max=min_max.transform(X_test[top5_features])  

```

---

#### **Exercise 8.06: Performing Normalization on Data**

In this exercise, you are required to normalize data after feature selection. You will use the same bank churn prediction data for normalizing. Continue using the same Jupyter notebook as the one used in the preceding exercise:

**Code:**

```python
# 1.	Fit the Normalizer() on the training data:
normalize = preprocessing.Normalizer()\
                         .fit(X_train[top5_features]) 

# 2.	Check the normalize function:
normalize

# 3.	Transform the training and testing data using normalize:
X_train_normalize=normalize.transform(X_train[top5_features]) 
X_test_normalize=normalize.transform(X_test[top5_features]) 
 
You can verify that the norm has now changed to 1 using the following code:
np.sqrt(np.sum(X_train_normalize**2, axis=1))

# 4.	Similarly, you can also evaluate the norm of the normalized test dataset:
np.sqrt(np.sum(X_test_normalize**2, axis=1))

```


---

#### **Exercise 8.07: Stratified K-fold**

In this exercise, you will fit the stratified k-fold function of scikit-learn to the bank churn prediction data and use the logistic regression classifier from the previous exercise to fit our k-fold data. Along with that, you will also implement the scikit-learn k-fold cross-validation scorer function:


**Code:**

```python
# 1.	Import StratifiedKFold from sklearn:
from sklearn.model_selection import StratifiedKFold

# 2.	Fit the classifier on the training and testing data with n_splits=10:
skf = StratifiedKFold(n_splits=10)\
      .split(X_train[top5_features].values,y_train.values)

# 3.	Calculate the k-cross fold validation score:
results=[] for i, (train,test) in enumerate(skf):
clf.fit(X_train[top5_features].values[train], y_train.values[train])
fit_result=clf.score(X_train[top5_features].values[test], y_train.values[test])
results.append(fit_result)
print('k-fold: %2d, Class Ratio: %s, Accuracy: %.4f'\
          % (i,np.bincount(y_train.values[train]),fit_result))

# 4.	Find the accuracy:
print('accuracy for CV is:%.3f' % np.mean(results))

# 		You will get an output showing an accuracy close to 0.790.

# 5.	Import the scikit-learn cross_val_score function:
from sklearn.model_selection import cross_val_score

# 6.	Fit the classifier and print the accuracy:
results_cross_val_score=cross_val_score(estimator=clf, X=X_train[top5_features].values, y=y_train.values,cv=10,n_jobs=1)
print('accuracy for CV is:%.3f '% np.mean(results_cross_val_score))

```


---

#### **Exercise 8.08: Fine-Tuning a Model**

In this exercise, you will implement a grid search to find out the best parameters for an SVM on the bank churn prediction data. You will continue using the same notebook as in the preceding exercise:

**Code:**

```python
# 1.	Import SVM, GridSearchCV, and StratifiedKfold:
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# 2.	Specify the parameters for the grid search as follows:
parameters = [{'kernel': ['linear'], 'C':[0.1, 1]}, {'kernel': ['rbf'], 'C':[0.1, 1]}]

# 3.	Fit the grid search with StratifiedKFold, setting the parameter as  n_splits = 3. 
clf = GridSearchCV(svm.SVC(), parameters,  cv = StratifiedKFold(n_splits = 3), verbose=4,n_jobs=-1)
clf.fit(X_train[top5_features], y_train)

# 4. Print the best score and the best parameters:
print('best score train:', clf.best_score_)
print('best parameters train: ', clf.best_params_)

```

---

#### **Activity 8.02: Tuning and Optimizing the Model**

The models you built in the previous activity produced good results, especially the random forest model, which produced an accuracy score of more than 80%. You now need to improve the accuracy of the random forest model and generalize it. Tuning the model using different preprocessing steps, cross-validation, and grid search will improve the accuracy of the model. You will be using the same Jupyter notebook as the one used in the preceding activity. Follow these steps:


1.	Store five out of seven features, that is, Avg_Calls_Weekdays,  
Current_Bill_Amt, Avg_Calls, Account_Age, and  
Avg_Days_Delinquent, in a variable called top5_features. Store the other two features, Percent_Increase_MOM and Complaint_Code, in a variable called top2_features. These features have values in the range of −1 to 7, whereas the other five features have values in the range of 0 to 374457. 
Hence, you can leave these features and standardize the remaining five features.

2.	Use StandardScalar to standardize the five features.

3.	Create a variable called X_train_scalar_combined, and combine the standardized five features with the two features (Percent_Increase_MOM and Complaint_Code) that were not standardized.

4.	Apply the same scalar standardization to the test data  
(X_test_scalar_combined).

5.	Fit the random forest model.

6.	Score the random forest model. You should get a value close to 0.81.

7.	Import the library for grid search and use the following parameters: 
parameters = [ {'min_samples_split': [9,10], \                 'n_estimators':[100,150,160]
                'max_depth': [5,7]}]

8.	Use grid search cross-validation with stratified k-fold to find out the best parameters. Use StratifiedKFold(n_splits = 3) and RandomForestClassifier().

9.	Print the best score and the best parameters. You should get the following values:

![Figure 8.43: Best score and best parameters](images/Figure-8.43.jpg)

10. Score the model using the test data. You should get a score close to 0.824. 
Combining the results of the accuracy score obtained in Activity 8.01, Implementing Different Classification Algorithms and Activity 8.02, Tuning and Optimizing the Model, here are the results for the random forest implementations:

![Figure 8.44: Comparing the accuracy of the random forest using different methods](images/Figure-8.44.jpg)


---

#### **Exercise 8.09: Evaluating the Performance Metrics for a Model**

In this exercise, you will calculate the F1 score and the accuracy of our random forest model for the bank churn prediction dataset. Continue using the same notebook as the one used in the preceding exercise: 

**Code:**

```python
# 1.	Import RandomForestClassifier, metrics, classification_reprt, confusion matrix, and accuracy_score:
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_ matrix,accuracy_score
from sklearn import metrics

# 2.	Fit the random forest classifier using the following code over the training data:
clf_random = RandomForestClassifier(n_estimators=20, max_depth=None, min_samples_split=7, random_state=0)
clf_random.fit(X_train[top5_features],y_train)

# 3.	Predict on the test data the classifier:
y_pred=clf_random.predict(X_test[top5_features])

# 4.	Print the classification report:
target_names = ['No Churn', 'Churn']
print(classification_report(y_test, y_pred,  target_names=target_names))

# 5.	Fit the confusion matrix and save it into a pandas DataFrame named cm_df:
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index = ['No Churn','Churn'], columns = ['No Churn','Churn'])

# 6.	Plot the confusion matrix using the following code:
plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True,fmt='g',cmap='Greys_r')

plt.title('Random Forest \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('True Values') plt.xlabel('Predicted Values')
plt.show()

```

---

#### **Exercise 8.10: Plotting the ROC Curve**


In this exercise, you will plot the ROC curve for the random forest model from the previous exercise on the bank churn prediction data. Continue with the same Jupyter notebook as the one used in the preceding exercise:

**Code:**

```python
# 1.	Import roc_curve,auc:
from sklearn.metrics import roc_curve,auc

# 2.	Calculate the TPR, FPR, and threshold using the following code:
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

# 3.	Plot the ROC curve using the following code:
plt.figure()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='%s AUC = %0.2f' % ('Random Forest', roc_auc, color = 'gray'))
plt.plot([0, 1], [0, 1],'k--') plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.ylabel('Sensitivity(True Positive Rate)')
plt.xlabel('1-Specificity(False Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

```

---

#### **Activity 8.03: Comparison of the Models**

In the previous activity, you improved the accuracy score of the random forest model score to 0.82. However, you were not using the correct performance metrics. In this activity, you will have to find out the F1 score of the random forest model trained in the previous activities and also compare the ROC curve of different machine learning models created in Activity 8.01, Implementing Different Classification Algorithms. 
Ensure that you use the same Jupyter notebook as the one used in the preceding activity. Follow these steps:


1.	Import the required libraries.
2.	Fit the random forest classifier with the parameters obtained from grid search in the preceding activity. Use the clf_random_grid variable.
3.	Predict on the standardized scalar test data, X_test_scalar_combined.
4.	Fit the classification report. You should get the following output:

![Figure 8.56: Classification report](images/Figure-8.56.jpg)

5. Plot the confusion matrix. Your output should be as follows:

![Figure 8.57: Confusion matrix](images/Figure-8.57.jpg)

6.	Import the library for the AUC and ROC curve.
7.	Use the classifiers that were created in _Activity 8.01, Implementing Different Classification Algorithms_, that is, **clf_logistic, clf_svm, clf_decision, and clf_random_grid**. Create a dictionary of all these models.
8.	Plot the ROC curve. The following for loop can be used as a hint:

```python
for m in models:
    model = m['model']
    ------ FIT THE MODEL
    ------ PREDICT
    ------ FIND THE FPR, TPR AND THRESHOLD
    roc_auc =FIND THE AUC
    plt.plot(fpr, tpr, label='%s AUC = %0.2f' % (m['label'], roc_auc))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0]) plt.ylim([0.0, 1.05])
plt.ylabel('Sensitivity(True Positive Rate)')
plt.xlabel('1-Specificity(False Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```
You plot should look as follows:

![Figure 8.58: ROC curve](images/Figure-8.58.jpg)

---
## Bài tập tổng hợp

## Bài Tập Thực Hành

### **Mô tả chung cho các bài tập 01-03**

```python
# =============================================================================
# TÓM TẮT VÀ HƯỚNG DẪN SỬ DỤNG
# =============================================================================

print("\n" + "="*80)
print("TÓM TẮT BÀI TẬP CLASSIFICATION ALGORITHMS")
print("="*80)

print("""
BÀI TẬP 1 - CƠ BẢN:
✓ Tạo synthetic dataset với 3 features, 2 classes
✓ Triển khai Decision Tree, Random Forest, SVM cơ bản
✓ So sánh accuracy và F1-score
✓ Visualize confusion matrices
→ Học cách sử dụng các thuật toán cơ bản

BÀI TẬP 2 - TRUNG BÌNH:
✓ Sử dụng Wine dataset thực tế (multi-class)
✓ EDA với correlation analysis
✓ Feature selection với Random Forest importance
✓ Hyperparameter tuning với GridSearchCV
✓ Ensemble model với Voting Classifier
→ Học cách tối ưu hóa model và feature engineering

BÀI TẬP 3 - NÂNG CAO:
✓ Customer Churn prediction với realistic dataset
✓ Feature engineering và business context
✓ Xử lý imbalanced data
✓ Business metrics (cost-benefit analysis)
✓ Model interpretation và deployment simulation
✓ Customer segmentation và business recommendations
→ Áp dụng machine learning vào business problem thực tế

KEY LEARNING POINTS:
1. Model Selection: Không chỉ dựa vào accuracy mà phải xem xét business context
2. Feature Engineering: Tạo features mới có thể cải thiện performance đáng kể
3. Imbalanced Data: Cần xử lý đặc biệt và chọn metrics phù hợp
4. Business Value: Machine learning phải tạo ra giá trị kinh doanh cụ thể
5. Model Deployment: Cần simulation để đảm bảo model hoạt động trong thực tế

NEXT STEPS:
- Thử nghiệm với real datasets từ Kaggle
- Học thêm về ensemble methods (XGBoost, LightGBM)
- Tìm hiểu về model explainability (SHAP, LIME)
- Thực hành với streaming data và online learning
""")

print("="*80)
print("🎉 HOÀN THÀNH TẤT CẢ BÀI TẬP! 🎉")
print("="*80)
```

**Các thư viện sử dụng chung cho bài tập 01-03**

```python
# =============================================================================
# BÀI TẬP THỰC HÀNH: FINE-TUNING CLASSIFICATION ALGORITHMS
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification, load_iris, load_wine
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           classification_report, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')
```
### **Bài Tập 1: CƠ BẢN - Synthetic Dataset Classification**

```python
# =============================================================================
# BÀI TẬP 1: CƠ BẢN - Synthetic Dataset Classification
# =============================================================================

print("=" * 60)
print("BÀI TẬP 1: CƠ BẢN - Synthetic Dataset Classification")
print("=" * 60)

def exercise_1_basic():
    """
    Bài tập cơ bản: Tạo dataset đơn giản và so sánh 3 algorithms
    
    Yêu cầu:
    1. Tạo synthetic dataset với 3 features, 2 classes, 1000 samples
    2. Triển khai Decision Tree, Random Forest, SVM
    3. So sánh hiệu suất với accuracy và F1-score
    4. Vẽ confusion matrix cho từng model
    """
    
    # Bước 1: Tạo synthetic dataset
    print("\n1. Tạo Synthetic Dataset")
    X, y = make_classification(
        n_samples=1000,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Tạo feature names
    feature_names = ['Feature_1', 'Feature_2', 'Feature_3']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Visualize dataset
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axes[i].scatter(X[y==0, i], X[y==1, i==0], c='red', alpha=0.6, label='Class 0')
        axes[i].scatter(X[y==1, i], X[y==1, i==1], c='blue', alpha=0.6, label='Class 1')
        axes[i].set_xlabel(f'Feature {i+1}')
        axes[i].set_title(f'Feature {i+1} Distribution')
        axes[i].legend()
    plt.tight_layout()
    plt.show()
    
    # Bước 2: Split data
    print("\n2. Chia dữ liệu Train/Test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Bước 3: Triển khai các models
    print("\n3. Triển khai và huấn luyện Models")
    models = {}
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train_scaled, y_train)
    models['SVM'] = svm
    
    # Bước 4: Đánh giá và so sánh models
    print("\n4. Đánh giá và So sánh Models")
    results = {}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, model) in enumerate(models.items()):
        # Dự đoán
        if name == 'SVM':
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        # Tính metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        results[name] = {
            'Accuracy': accuracy,
            'F1-Score': f1,
            'Precision': precision,
            'Recall': recall
        }
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name}\nAccuracy: {accuracy:.3f}, F1: {f1:.3f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    # Results DataFrame
    results_df = pd.DataFrame(results).T
    print("\n=== RESULTS COMPARISON ===")
    print(results_df.round(4))
    
    # Tìm model tốt nhất
    best_model = results_df['F1-Score'].idxmax()
    print(f"\nBest Model: {best_model} (F1-Score: {results_df.loc[best_model, 'F1-Score']:.4f})")
    
    return models, results_df

# Chạy bài tập 1
models_basic, results_basic = exercise_1_basic()
```
### **Bài Tập 02 TRUNG BÌNH - Real Dataset với Feature Selection**
```python
# =============================================================================
# BÀI TẬP 2: TRUNG BÌNH - Real Dataset với Feature Selection
# =============================================================================

print("\n" + "=" * 60)
print("BÀI TẬP 2: TRUNG BÌNH - Wine Dataset với Feature Selection")
print("=" * 60)

def exercise_2_intermediate():
    """
    Bài tập trung bình: Sử dụng Wine dataset từ sklearn
    
    Yêu cầu:
    1. Load Wine dataset và EDA
    2. Feature selection với Random Forest importance
    3. Hyperparameter tuning với GridSearchCV
    4. Tạo ensemble model
    """
    
    # Bước 1: Load và explore dataset
    print("\n1. Load và Explore Wine Dataset")
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Classes: {target_names}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # EDA - Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Wine Dataset - Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Class distribution by some key features
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    key_features = ['alcohol', 'flavanoids', 'color_intensity', 'proline']
    
    for i, feature in enumerate(key_features):
        ax = axes[i//2, i%2]
        for class_idx, class_name in enumerate(target_names):
            class_data = df[df['target'] == class_idx][feature]
            ax.hist(class_data, alpha=0.7, label=class_name, bins=15)
        ax.set_title(f'{feature} Distribution by Class')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Bước 2: Split data
    print("\n2. Chia dữ liệu và Chuẩn hóa")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Bước 3: Feature Selection với Random Forest
    print("\n3. Feature Selection với Random Forest")
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selector.fit(X_train_scaled, y_train)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_selector.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Visualize feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importance (Random Forest)')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    # Select top k features
    k = 8  # Select top 8 features
    top_features_idx = feature_importance.head(k).index
    X_train_selected = X_train_scaled[:, top_features_idx]
    X_test_selected = X_test_scaled[:, top_features_idx]
    selected_feature_names = [feature_names[i] for i in top_features_idx]
    
    print(f"\nSelected {k} features: {selected_feature_names}")
    
    # Bước 4: Hyperparameter Tuning
    print("\n4. Hyperparameter Tuning với GridSearchCV")
    
    models_tuned = {}
    
    # Decision Tree tuning
    dt_params = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    dt_grid = GridSearchCV(dt, dt_params, cv=5, scoring='f1_macro', n_jobs=-1)
    dt_grid.fit(X_train_selected, y_train)
    models_tuned['Decision Tree'] = dt_grid.best_estimator_
    print(f"Best DT params: {dt_grid.best_params_}")
    
    # Random Forest tuning
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='f1_macro', n_jobs=-1)
    rf_grid.fit(X_train_selected, y_train)
    models_tuned['Random Forest'] = rf_grid.best_estimator_
    print(f"Best RF params: {rf_grid.best_params_}")
    
    # SVM tuning
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    
    svm = SVC(random_state=42)
    svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='f1_macro', n_jobs=-1)
    svm_grid.fit(X_train_selected, y_train)
    models_tuned['SVM'] = svm_grid.best_estimator_
    print(f"Best SVM params: {svm_grid.best_params_}")
    
    # Bước 5: Tạo Ensemble Model
    print("\n5. Tạo Ensemble Model")
    ensemble = VotingClassifier(
        estimators=[
            ('dt', models_tuned['Decision Tree']),
            ('rf', models_tuned['Random Forest']),
            ('svm', models_tuned['SVM'])
        ],
        voting='hard'  # Use hard voting
    )
    ensemble.fit(X_train_selected, y_train)
    models_tuned['Ensemble'] = ensemble
    
    # Bước 6: Đánh giá tất cả models
    print("\n6. Đánh giá và So sánh Models")
    results_intermediate = {}
    
    for name, model in models_tuned.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='f1_macro')
        
        # Test predictions
        y_pred = model.predict(X_test_selected)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        
        results_intermediate[name] = {
            'CV_F1_Mean': cv_scores.mean(),
            'CV_F1_Std': cv_scores.std(),
            'Test_Accuracy': accuracy,
            'Test_F1_Macro': f1_macro,
            'Test_Precision': precision_macro,
            'Test_Recall': recall_macro
        }
    
    # Results comparison
    results_df_intermediate = pd.DataFrame(results_intermediate).T
    print("\n=== MODEL COMPARISON RESULTS ===")
    print(results_df_intermediate.round(4))
    
    # Confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, (name, model) in enumerate(models_tuned.items()):
        y_pred = model.predict(X_test_selected)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name}\nAccuracy: {accuracy_score(y_test, y_pred):.3f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    # Best model
    best_model = results_df_intermediate['Test_F1_Macro'].idxmax()
    print(f"\nBest Model: {best_model}")
    print(f"F1-Macro Score: {results_df_intermediate.loc[best_model, 'Test_F1_Macro']:.4f}")
    
    return models_tuned, results_df_intermediate, selected_feature_names

# Chạy bài tập 2
models_intermediate, results_intermediate, selected_features = exercise_2_intermediate()
```

### **Bài Tập 03 NÂNG CAO - Customer Churn Prediction với Business Context**

```python
# =============================================================================
# BÀI TẬP 3: NÂNG CAO - Customer Churn Prediction với Business Context
# =============================================================================

print("\n" + "=" * 60)
print("BÀI TẬP 3: NÂNG CAO - Customer Churn Prediction")
print("=" * 60)

def exercise_3_advanced():
    """
    Bài tập nâng cao: Customer Churn Prediction với business context
    
    Yêu cầu:
    1. Tạo realistic churn dataset với feature engineering
    2. Xử lý imbalanced data
    3. Advanced model selection và tuning
    4. Business interpretation và cost analysis
    5. Model deployment simulation
    """
    
    # Bước 1: Tạo realistic churn dataset
    print("\n1. Tạo Realistic Customer Churn Dataset")
    np.random.seed(42)
    n_samples = 8000
    
    # Customer demographics
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(45, 15, n_samples).astype(int),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'tenure_months': np.random.exponential(24, n_samples).astype(int),
        'monthly_charges': np.random.gamma(2, 35, n_samples),
        'total_charges': np.random.gamma(3, 500, n_samples),
        
        # Service features
        'internet_service': np.random.choice(['DSL', 'Fiber', 'No'], n_samples, p=[0.35, 0.45, 0.2]),
        'phone_service': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
        'multiple_lines': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'online_security': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'online_backup': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
        'device_protection': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
        'tech_support': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'streaming_tv': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'streaming_movies': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        
        # Contract and payment
        'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                   n_samples, p=[0.55, 0.25, 0.2]),
        'paperless_billing': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 
                                          'Bank transfer', 'Credit card'],
                                         n_samples, p=[0.35, 0.2, 0.25, 0.2]),
        
        # Behavioral features
        'avg_monthly_calls': np.random.poisson(25, n_samples),
        'customer_service_calls': np.random.poisson(2, n_samples),
        'late_payments': np.random.poisson(1, n_samples)
    }
    
    df_churn = pd.DataFrame(data)
    
    # Fix data types and ranges
    df_churn['age'] = np.clip(df_churn['age'], 18, 80)
    df_churn['tenure_months'] = np.clip(df_churn['tenure_months'], 1, 72)
    df_churn['monthly_charges'] = np.clip(df_churn['monthly_charges'], 20, 120)
    
    # Feature Engineering
    print("\n2. Feature Engineering")
    
    # Create engineered features
    df_churn['charges_per_month'] = df_churn['total_charges'] / (df_churn['tenure_months'] + 1)
    df_churn['services_count'] = (df_churn[['phone_service', 'multiple_lines', 'online_security',
                                           'online_backup', 'device_protection', 'tech_support',
                                           'streaming_tv', 'streaming_movies']].sum(axis=1))
    df_churn['is_senior'] = (df_churn['age'] >= 65).astype(int)
    df_churn['high_monthly_charges'] = (df_churn['monthly_charges'] > df_churn['monthly_charges'].quantile(0.75)).astype(int)
    df_churn['short_tenure'] = (df_churn['tenure_months'] <= 12).astype(int)
    df_churn['high_support_calls'] = (df_churn['customer_service_calls'] >= 3).astype(int)
    
    # Create target variable with realistic churn logic
    churn_probability = (
        0.05 +  # Base churn rate
        0.35 * (df_churn['contract'] == 'Month-to-month') +
        0.25 * df_churn['short_tenure'] +
        0.20 * df_churn['high_monthly_charges'] +
        0.15 * (df_churn['tech_support'] == 0) +
        0.15 * df_churn['high_support_calls'] +
        0.10 * (df_churn['internet_service'] == 'Fiber') +
        0.10 * df_churn['is_senior'] +
        0.05 * (df_churn['payment_method'] == 'Electronic check') -
        0.10 * (df_churn['services_count'] > 4) -  # More services = less churn
        0.05 * (df_churn['tenure_months'] > 24)   # Longer tenure = less churn
    )
    
    # Ensure probability is between 0 and 1
    churn_probability = np.clip(churn_probability, 0, 0.8)
    df_churn['churn'] = np.random.binomial(1, churn_probability, n_samples)
    
    print(f"Dataset shape: {df_churn.shape}")
    print(f"Churn rate: {df_churn['churn'].mean():.3f}")
    print(f"Class distribution: {df_churn['churn'].value_counts()}")
    
    # EDA
    print("\n3. Exploratory Data Analysis")
    
    # Churn rate by key features
    categorical_features = ['contract', 'internet_service', 'payment_method', 'is_senior']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(categorical_features):
        churn_rate = df_churn.groupby(feature)['churn'].mean().sort_values(ascending=False)
        churn_rate.plot(kind='bar', ax=axes[i], color='skyblue', alpha=0.7)
        axes[i].set_title(f'Churn Rate by {feature}')
        axes[i].set_ylabel('Churn Rate')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Numerical features distribution
    numerical_features = ['age', 'tenure_months', 'monthly_charges', 'services_count']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(numerical_features):
        df_churn[df_churn['churn'] == 0][feature].hist(alpha=0.7, label='No Churn', bins=30, ax=axes[i])
        df_churn[df_churn['churn'] == 1][feature].hist(alpha=0.7, label='Churn', bins=30, ax=axes[i])
        axes[i].set_title(f'{feature} Distribution')
        axes[i].set_xlabel(feature)
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Bước 4: Data Preparation
    print("\n4. Data Preparation")
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['gender', 'internet_service', 'contract', 'payment_method']
    
    df_processed = df_churn.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    # Select features for modeling
    feature_columns = [
        'age', 'tenure_months', 'monthly_charges', 'total_charges',
        'phone_service', 'multiple_lines', 'online_security', 'online_backup',
        'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',
        'paperless_billing', 'avg_monthly_calls', 'customer_service_calls',
        'late_payments', 'charges_per_month', 'services_count', 'is_senior',
        'high_monthly_charges', 'short_tenure', 'high_support_calls',
        'gender_encoded', 'internet_service_encoded', 'contract_encoded', 'payment_method_encoded'
    ]
    
    X = df_processed[feature_columns]
    y = df_processed['churn']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Training churn rate: {y_train.mean():.3f}")
    
    # Bước 5: Handle Imbalanced Data (SMOTE simulation)
    print("\n5. Handling Imbalanced Data")
    
    # Simple oversampling simulation (in practice, use SMOTE from imbalanced-learn)
    from sklearn.utils import resample
    
    # Separate majority and minority classes
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_columns)
    X_train_df['target'] = y_train.values
    
    majority = X_train_df[X_train_df.target == 0]
    minority = X_train_df[X_train_df.target == 1]
    
    # Upsample minority class
    minority_upsampled = resample(minority, 
                                 replace=True,
                                 n_samples=len(majority),
                                 random_state=42)
    
    # Combine majority and upsampled minority
    balanced_df = pd.concat([majority, minority_upsampled])
    
    X_train_balanced = balanced_df.drop('target', axis=1).values
    y_train_balanced = balanced_df['target'].values
    
    print(f"Original training set: {np.bincount(y_train)}")
    print(f"Balanced training set: {np.bincount(y_train_balanced)}")
    
    # Bước 6: Advanced Model Selection
    print("\n6. Advanced Model Selection và Hyperparameter Tuning")
    
    models_advanced = {}
    
    # Decision Tree with class weights
    dt_params = {
        'max_depth': [5, 7, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    dt_grid = GridSearchCV(dt, dt_params, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    dt_grid.fit(X_train_balanced, y_train_balanced)
    models_advanced['Decision Tree'] = dt_grid.best_estimator_
    
    # Random Forest with class weights
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'class_weight': [None, 'balanced']
    }
    
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    rf_grid.fit(X_train_balanced, y_train_balanced)
    models_advanced['Random Forest'] = rf_grid.best_estimator_
    
    # SVM with class weights
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'class_weight': [None, 'balanced']
    }
    
    svm = SVC(random_state=42, probability=True)
    svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    svm_grid.fit(X_train_balanced, y_train_balanced)
    models_advanced['SVM'] = svm_grid.best_estimator_
    
    print("Best parameters:")
    print(f"DT: {dt_grid.best_params_}")
    print(f"RF: {rf_grid.best_params_}")
    print(f"SVM: {svm_grid.best_params_}")
    
    # Bước 7: Model Evaluation với Business Metrics
    print("\n7. Model Evaluation với Business Context")
    
    # Define business costs
    COST_FALSE_POSITIVE = 50   # Cost of incorrectly predicting churn (retention campaign cost)
    COST_FALSE_NEGATIVE = 200  # Cost of missing actual churn (lost customer value)
    REVENUE_PER_CUSTOMER = 1200  # Annual customer value
    
    def calculate_business_metrics(y_true, y_pred, model_name):
        """Calculate business-oriented metrics"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Traditional metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Business metrics
        total_cost = (fp * COST_FALSE_POSITIVE) + (fn * COST_FALSE_NEGATIVE)
        potential_revenue_saved = tp * REVENUE_PER_CUSTOMER * 0.3  # Assume 30% retention success
        net_benefit = potential_revenue_saved - total_cost
        cost_per_customer = total_cost / len(y_true)
        
        return {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Total_Cost': total_cost,
            'Revenue_Saved': potential_revenue_saved,
            'Net_Benefit': net_benefit,
            'Cost_Per_Customer': cost_per_customer,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
        }
    
    # Evaluate all models
    business_results = []
    
    for name, model in models_advanced.items():
        y_pred = model.predict(X_test_scaled)
        metrics = calculate_business_metrics(y_test, y_pred, name)
        business_results.append(metrics)
        
        print(f"\n=== {name} Results ===")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1-Score: {metrics['F1-Score']:.4f}")
        print(f"Total Cost: ${metrics['Total_Cost']:,.2f}")
        print(f"Revenue Saved: ${metrics['Revenue_Saved']:,.2f}")
        print(f"Net Benefit: ${metrics['Net_Benefit']:,.2f}")
        print(f"Cost per Customer: ${metrics['Cost_Per_Customer']:.2f}")
    
    # Create business results DataFrame
    business_df = pd.DataFrame(business_results)
    print("\n=== BUSINESS METRICS COMPARISON ===")
    print(business_df[['Model', 'Accuracy', 'F1-Score', 'Net_Benefit', 'Cost_Per_Customer']].round(4))
    
    # Bước 8: ROC Curves và Threshold Optimization
    print("\n8. ROC Analysis và Threshold Optimization")
    
    plt.figure(figsize=(15, 5))
    
    # ROC Curves
    plt.subplot(1, 3, 1)
    for name, model in models_advanced.items():
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature Importance (Random Forest)
    plt.subplot(1, 3, 2)
    rf_model = models_advanced['Random Forest']
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()
    
    # Net Benefit Comparison
    plt.subplot(1, 3, 3)
    models_list = business_df['Model'].values
    net_benefits = business_df['Net_Benefit'].values
    colors = ['skyblue' if x > 0 else 'lightcoral' for x in net_benefits]
    
    plt.bar(models_list, net_benefits, color=colors)
    plt.xlabel('Models')
    plt.ylabel('Net Benefit ($)')
    plt.title('Business Value Comparison')
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Bước 9: Model Interpretation và Business Insights
    print("\n9. Model Interpretation và Business Insights")
    
    # Best model selection based on business metrics
    best_model_idx = business_df['Net_Benefit'].idxmax()
    best_model_name = business_df.iloc[best_model_idx]['Model']
    best_model = models_advanced[best_model_name]
    
    print(f"Best Model for Business: {best_model_name}")
    print(f"Net Benefit: ${business_df.iloc[best_model_idx]['Net_Benefit']:,.2f}")
    
    # Feature importance insights
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        print("\nTop 10 Most Important Features:")
        for idx, row in importance_df.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")
    
    # Customer segmentation for business strategy
    print("\n10. Customer Segmentation untuk Business Strategy")
    
    # Predict probabilities
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Create customer segments based on churn probability
    def categorize_risk(prob):
        if prob < 0.3:
            return 'Low Risk'
        elif prob < 0.7:
            return 'Medium Risk'
        else:
            return 'High Risk'
    
    risk_segments = [categorize_risk(p) for p in y_pred_proba]
    segment_counts = pd.Series(risk_segments).value_counts()
    
    print("Customer Risk Segmentation:")
    for segment, count in segment_counts.items():
        percentage = count / len(risk_segments) * 100
        print(f"{segment}: {count} customers ({percentage:.1f}%)")
    
    # Business recommendations
    print("\n=== BUSINESS RECOMMENDATIONS ===")
    
    high_risk_customers = np.sum(np.array(risk_segments) == 'High Risk')
    medium_risk_customers = np.sum(np.array(risk_segments) == 'Medium Risk')
    
    print(f"1. IMMEDIATE ACTION REQUIRED:")
    print(f"   - {high_risk_customers} high-risk customers need immediate retention efforts")
    print(f"   - Estimated cost: ${high_risk_customers * 100:,} for targeted campaigns")
    print(f"   - Potential revenue at risk: ${high_risk_customers * REVENUE_PER_CUSTOMER:,}")
    
    print(f"\n2. PROACTIVE MEASURES:")
    print(f"   - {medium_risk_customers} medium-risk customers for proactive engagement")
    print(f"   - Focus on improving: {', '.join(importance_df.head(3)['Feature'].values)}")
    
    print(f"\n3. MODEL PERFORMANCE:")
    print(f"   - Expected net benefit: ${business_df.iloc[best_model_idx]['Net_Benefit']:,.2f}")
    print(f"   - Cost per customer: ${business_df.iloc[best_model_idx]['Cost_Per_Customer']:.2f}")
    
    # Bước 11: Model Deployment Simulation
    print("\n11. Model Deployment Simulation")
    
    def predict_customer_churn(customer_data, model, scaler, feature_columns):
        """
        Simulate model deployment for single customer prediction
        """
        # Prepare customer data
        customer_df = pd.DataFrame([customer_data])
        customer_scaled = scaler.transform(customer_df[feature_columns])
        
        # Predict
        churn_probability = model.predict_proba(customer_scaled)[0, 1]
        churn_prediction = model.predict(customer_scaled)[0]
        risk_level = categorize_risk(churn_probability)
        
        return {
            'churn_probability': churn_probability,
            'churn_prediction': churn_prediction,
            'risk_level': risk_level
        }
    
    # Example customer prediction
    example_customer = {
        'age': 35,
        'tenure_months': 6,
        'monthly_charges': 85.0,
        'total_charges': 500.0,
        'phone_service': 1,
        'multiple_lines': 1,
        'online_security': 0,
        'online_backup': 0,
        'device_protection': 0,
        'tech_support': 0,
        'streaming_tv': 1,
        'streaming_movies': 1,
        'paperless_billing': 1,
        'avg_monthly_calls': 30,
        'customer_service_calls': 4,
        'late_payments': 2,
        'charges_per_month': 85.0,
        'services_count': 4,
        'is_senior': 0,
        'high_monthly_charges': 1,
        'short_tenure': 1,
        'high_support_calls': 1,
        'gender_encoded': 1,
        'internet_service_encoded': 1,
        'contract_encoded': 0,
        'payment_method_encoded': 0
    }
    
    prediction_result = predict_customer_churn(
        example_customer, best_model, scaler, feature_columns
    )
    
    print("Example Customer Prediction:")
    print(f"Churn Probability: {prediction_result['churn_probability']:.3f}")
    print(f"Risk Level: {prediction_result['risk_level']}")
    print(f"Recommended Action: {'Immediate retention campaign' if prediction_result['risk_level'] == 'High Risk' else 'Monitor and engage'}")
    
    # Model Performance Summary
    print("\n" + "="*60)
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    final_summary = business_df.loc[business_df['Model'] == best_model_name].iloc[0]
    
    print(f"Best Model: {best_model_name}")
    print(f"Accuracy: {final_summary['Accuracy']:.4f}")
    print(f"Precision: {final_summary['Precision']:.4f}")
    print(f"Recall: {final_summary['Recall']:.4f}")
    print(f"F1-Score: {final_summary['F1-Score']:.4f}")
    print(f"Business Net Benefit: ${final_summary['Net_Benefit']:,.2f}")
    print(f"Cost per Customer: ${final_summary['Cost_Per_Customer']:.2f}")
    
    return {
        'models': models_advanced,
        'best_model': best_model,
        'business_results': business_df,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'label_encoders': label_encoders
    }

# Chạy bài tập 3
try:
    results_advanced = exercise_3_advanced()
    print("\n✓ Bài tập 3 hoàn thành thành công!")
except Exception as e:
    print(f"⚠ Lỗi trong bài tập 3: {e}")
    print("Gợi ý: Kiểm tra các thư viện đã được import đầy đủ")
```

### Bài Tập 04: Cơ bản
1. Tạo một dataset classification đơn giản với 3 features và 2 classes
2. Triển khai Decision Tree, Random Forest, và SVM
3. So sánh hiệu suất sử dụng accuracy và F1-score
4. Vẽ confusion matrix cho từng model

### Bài Tập 05: Trung bình
1. Sử dụng dataset Iris hoặc Wine từ sklearn
2. Thực hiện feature selection sử dụng Random Forest feature importance
3. Tune hyperparameters cho các models sử dụng GridSearchCV
4. Tạo ensemble model kết hợp 3 algorithms

### Bài Tập 06: Nâng cao - Customer Churn Prediction
1. Sử dụng dataset churn thực tế (Telco Customer Churn từ Kaggle)
2. Thực hiện EDA chi tiết và feature engineering
3. Xử lý imbalanced data sử dụng SMOTE hoặc class weights
4. Triển khai pipeline hoàn chỉnh với cross-validation
5. Tạo dashboard đơn giản để visualize results
6. Đề xuất business strategy dựa trên model predictions

### Code Template cho Bài Tập 3

```python
# Template cho bài tập Customer Churn
def advanced_churn_analysis():
    """
    Template cho phân tích churn nâng cao
    """
    # TODO: Load real dataset
    # df = pd.read_csv('telco_churn.csv')
    
    # TODO: EDA và feature engineering
    # - Tạo new features từ existing features
    # - Xử lý missing values
    # - Encode categorical variables
    
    # TODO: Handle imbalanced data
    # from imblearn.over_sampling import SMOTE
    # smote = SMOTE(random_state=42)
    # X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # TODO: Advanced model selection
    # - Thêm Gradient Boosting, XGBoost
    # - Ensemble methods
    # - Stacking classifier
    
    # TODO: Business interpretation
    # - Cost-benefit analysis
    # - Customer lifetime value integration
    # - Actionable insights
    
    pass

# Gợi ý đánh giá:
# - Precision vs Recall trade-off
# - ROC-AUC vs PR-AUC
# - Business metrics (cost of false positives vs false negatives)
```
## Hướng dẫn chấm điểm và đánh giá

### Rubric cho các bài tập:

**Bài Tập 1 (Cơ bản)**:
- Code functionality (40%)
- Data analysis quality (30%) 
- Business insights (20%)
- Presentation (10%)

**Bài Tập 2 (Trung bình)**:
- Technical implementation (35%)
- RFM analysis depth (25%)
- Method comparison (20%) 
- Strategic recommendations (20%)

**Bài Tập 3 (Nâng cao)**:
- Advanced techniques usage (30%)
- Feature engineering creativity (25%)
- Business impact analysis (25%)
- Innovation and insights (20%)

### Tiêu chí đánh giá chung:
- **Xuất sắc (90-100%)**: Vượt expectation, có insights độc đáo
- **Tốt (80-89%)**: Hoàn thành tốt tất cả requirements  
- **Khá (70-79%)**: Hoàn thành cơ bản với một số thiếu sót
- **Trung bình (60-69%)**: Hoàn thành một phần, thiếu insights
- **Yếu (<60%)
