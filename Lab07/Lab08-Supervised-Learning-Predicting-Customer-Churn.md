# Lab 08 Supervised Learning: Predicting Customer Churn

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

#### **Exercise 7.01: Comparing Predictions by Linear and Logistic Regression on the Shill Bidding Dataset**
Consider the **Shill_Bidding_Dataset.csv** dataset, which contains details regarding auctions done for various products on eBay.com. The target column, **Class**, provides information about the bidding behavior, **0** being normal and **1** being abnormal behavior. Abnormal behavior can be similar to malicious clicks or automatic bidding. You have been asked to develop a machine learning model that can predict whether the bidding behavior in a particular auction is normal (**0**) or not (**1**). Apply linear and logistic regression to predict the output and check which one of them is useful in this situation:


_Perform the following steps to achieve the aim of the exercise:_

**Code:**

```python
# 1.	Import the pandas, numpy, sklearn, and matplotlib libraries using the following code:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt

# 2.	Create a new DataFrame and name it data. Look at the first few rows using the following code:
data = pd.read_csv("Shill_Bidding_Dataset.csv")
data.head()

# 3.	Next, remove the columns that are irrelevant to the case study; that is, remove the Record_ID, Auction_ID,
# 		and Bidder_ID columns. This is because these columns contain unique IDs and thus do not add any new information
# 		to the model:
data.drop(["Record_ID","Auction_ID","Bidder_ID"],axis=1, inplace=True)

# 4.	Now view the first five rows of the revised data:
data.head()

# 5.	Split the data into training and testing sets as follows. We are sticking to the default value for test data 
# 		size (30% of the entire data). Moreover, to add reproducibility, we are using a random_state of 1, and to account  
# 		for any class imbalance, we will use stratified splitting. Reproducibility will ensure that when someone else  
# 		runs the code, they will get similar results. Stratified splitting will ensure that we take into account 
# 		the class distribution of the target column while splitting the data into train and test sets:
X = data.drop("Class",axis=1)
y = data["Class"]

# Split the dataset into training and testing sets 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=1, stratify=y)

# 6.	Print the size of training and testing datasets:
print("Training dataset size: {}, Testing dataset size: {}".format(X_train.shape,X_test.shape))

# 7.	Fit the model using linear regression:
linear = linear_model.LinearRegression()
linear.fit(X_train,y_train)


# 8.	Predict on the first 10 test data points using the following code:
linear.predict(X_test)[:10]

# 9.	Now check the actual target values using y_test[:10].values:
y_test[:10].values

# 10.	Evaluate the score of the linear regression model on the training and testing datasets:
print("Score on training dataset: {}, "\
       "Score on testing dataset: {}"\
       .format(linear.score(X_train,y_train), linear.score(X_test,y_test)))

# 11.	Fit the model using logistic regression as follows:
logit = linear_model.LogisticRegression()
logit.fit(X_train,y_train)

# 12.	Predict on the test data:
logit.predict(X_test)[:10]

# 13.	Check the actual target values with y_test[:10].values:
y_test[:10].values

# 14.	Similar to linear regression, find the score on the training and testing datasets as follows:
print("Score on training dataset: {}, "\
       "Score on testing dataset: {}"\
       .format(logit.score(X_train,y_train), logit.score(X_test,y_test)))

```


#### **Exercise 7.02: Obtaining the Data**
In this exercise, you will import the banking data (Churn_Modelling.csv) provided by the bank and do some initial checks, such as seeing how many rows and columns are present. This will give you a quick peek into the real-life problem statements that a marketing analyst and a data scientist will work on, where the dataset is not always clean. The more time you spend improving and getting familiar with the dataset, the better the observations you can make about the trend will be:

**Code:**

```python
# 1.	Import the pandas, numpy, matplotlib, and seaborn libraries:
import pandas as pd import numpy as np import matplotlib.pyplot as plt import seaborn as sns

# 2.	Read the data into a pandas DataFrame named data. Also, view the first five rows of the dataset:
data= pd.read_csv('Churn_Modelling.csv') data.head(5)

# 3. Check the number of rows and columns in the dataset:
len(data) data.shape

```

---

#### **Exercise 7.03: Imputing Missing Values**
After reading the banking data, you have to find any missing values in the data and perform imputation on the missing values:

**Code:**

```python
# 1.	Check for any missing values first using the following code:
data.isnull().values.any()

# 2.	Explore the columns that have these missing values by using the following code:
data.isnull().any()

# 3.	Use describe function to explore the data in the Age and EstimatedSalary columns:
data[["EstimatedSalary","Age"]].describe()

# 4.	Now use describe function for the entire DataFrame as well.
# 		This will help in understanding the statistical description of the columns:
data.describe()

# 5.	Now, check the count of 0s and 1s in this column using the following syntax:
data['HasCrCard'].value_counts()

# 6.	Use the following syntax to find out the total number of missing values:
data.isnull().sum()

# 7.	Find out the percentage of missing values using the following code:
round(data.isnull().sum()/len(data)*100,2)

# 8. Check the data types of the missing columns:
data[["Gender","Age","EstimatedSalary"]].dtypes

# 9.	Now you need to impute the missing values. You can do that by dropping the rows that have missing values,
# 		filling in the missing values with a test statistic (such as mean, mode, or median), or predicting
# 		the missing values using a machine learning algorithm. For EstimatedSalary,
# 		fill in the missing values with the mean of the data in that column using the following code:
mean_value=data['EstimatedSalary'].mean()
data['EstimatedSalary']=data['EstimatedSalary'].fillna(mean_value)

# 10.	For Gender, use value_count() to see how many instances of each gender are present:
data['Gender'].value_counts()

data['Gender']=data['Gender'].fillna(data['Gender'].value_counts().idxmax())

# 11.	For Age, use mode() to get the mode of the data, which is 37, and then replace the missing values with
# 		the mode of the values in the column using the following code:
data['Age'].mode() mode_value=data['Age'].mode()
data['Age']=data['Age'].fillna(mode_value[0])

# 12.	Check whether the missing values have been imputed:
data.isnull().any()

```


---

#### **Exercise 7.04: Renaming Columns and Changing the Data Type**
Scrubbing data also involves renaming columns in the right format and can include removing any special characters and spaces in the column names, shifting the target variable either to the extreme left or right for better visibility, and checking whether the data types of the columns are correct. In this exercise, you have to convert the column names into a more human-readable format. For example, you must have noticed column names such as ActMem and CredRate. These can be renamed to give a clearer idea of what the columns represent. The main reason behind doing this is that if someone else is going over your work, ambiguous column names can reduce the clarity. Therefore, in this exercise, you will rename some of the columns, change the data types, and shift the Churn column to the rightmost position. This will help differentiate the independent features from the dependent ones:


**Code:**

```python
# 1.	Rename the CredRate, ActMem, Prod Number, and Exited columns using the following command:
data = data.rename(columns={'CredRate': 'CreditScore', 'ActMem' : 'IsActiveMember',\
                            'Prod Number': 'NumOfProducts', 'Exited':'Churn'}) 

# 2.	Check that the preceding columns have been appropriately renamed using the columns command:
data.columns

# 3.	Move the Churn column to the right and drop the CustomerId column using the following code.
# 		You will need to drop the CustomerId column since it is unique for each entry and thus,
# 		does not provide any useful information:
data.drop(labels=['CustomerId'], axis=1,inplace = True)
column_churn = data['Churn']
data.drop(labels=['Churn'], axis=1,inplace = True)
data.insert(len(data.columns), 'Churn', column_churn.values)

# 4.	Check whether the order of the columns has been fixed using the following code:
data.columns

# 5.	Change the data type of the Geography, Gender, HasCrCard, Churn, and IsActiveMember columns to category as 
# 		shown. Recall that these columns were initially strings or objects. However, since these are distinct values,
# 		you will need to convert them to categorical variables by converting the data type to category:
data["Geography"] = data["Geography"].astype('category')
data["Gender"] = data["Gender"].astype('category')
data["HasCrCard"] = data["HasCrCard"].astype('category')
data["Churn"] = data["Churn"].astype('category')
data["IsActiveMember"] = data["IsActiveMember"].astype('category')

# 6.	Now check whether the data types have been converted or not using the following code:
data.dtypes

```


---

#### **Exercise 7.05: Obtaining the Statistical Overview and Correlation Plot**
You are requested to find out the number of customers that churned using basic exploration techniques. The churn column has two attributes: 0 indicates that the customer did not churn and 1 implies that the customer churned. You will be required to obtain the percentage of customers who churned, the percentage of customers who have a credit card, and more. This information will be valuable at a later stage to make inferences about consumer behavior. You are also required to plot the correlation matrix, which will give you a basic understanding of the relationship between the target variable and the rest of the variables.

**Code:**

```python
# 1.	Inspect the target variable to see how many of the customers have churned using the following code:
data['Churn'].value_counts(0)

# 2.	Inspect the percentage of customers who left the bank using the following code:
data['Churn'].value_counts(1)*100

# 3.	Inspect the percentage of customers that have a credit card using the following code:
data['HasCrCard'].value_counts(1)*100

# 4.	Get a statistical overview of the data:
data.describe()


# 5.	Inspect the mean attributes of customers who churned compared to those who did not churn:
summary_churn = data.groupby('Churn') summary_churn.mean()


# 6. Also, find the median attributes of the customers:
summary_churn.median()

# 7. Now use the seaborn library to plot the correlation plot using the following code:
corr = data.corr()
plt.figure(figsize=(15,8)) sns.heatmap(corr, xticklabels=corr.columns.values,\
             yticklabels=corr.columns.values, annot=True,cmap='Greys_r')
corr

```


---

#### **Exercise 7.06: Performing Exploratory Data Analysis (EDA)**
In this exercise, you will perform EDA, which includes univariate analysis and bivariate analysis, on the Churn_Modelling.csv dataset. You will use this analysis to come up with inferences regarding the dataset, and the relationship between features such as geography, customer age, customer bank balance, and more with respect to churn:


**Code:**

```python
# 1.	Start with univariate analysis. Plot the distribution graph of the customers for
# 		the EstimatedSalary, Age, and Balance variables using the following code:
f, axes = plt.subplots(ncols=3, figsize=(15, 6)) 
sns.distplot(data.EstimatedSalary, kde=True, color="gray",  ax=axes[0]).set_title('EstimatedSalary')
axes[0].set_ylabel('No of Customers')

sns.distplot(data.Age, kde=True, color="gray",  ax=axes[1]).set_title('Age')
axes[1].set_ylabel('No of Customers') 

sns.distplot(data.Balance, kde=True, color="gray", 
ax=axes[2]).set_title('Balance')
axes[2].set_ylabel('No of Customers')

# 2.	Now, move on to bivariate analysis. Inspect whether there is a difference in churn for
# 		Gender using bivariate analysis. Use the following code:
plt.figure(figsize=(15,4))
p=sns.countplot(y="Gender", hue='Churn', data=data, palette="Greys_r")

legend = p.get_legend()
legend_txt = legend.texts
legend_txt[0].set_text("No Churn")
legend_txt[1].set_text("Churn") 
p.set_title('Customer Churn Distribution by Gender')

# 3. Plot Geography versus Churn:
plt.figure(figsize=(15,4))
p=sns.countplot(x='Geography', hue='Churn', data=data, palette="Greys_r")

legend = p.get_legend()
legend_txt = legend.texts
legend_txt[0].set_text("No Churn")
legend_txt[1].set_text("Churn") 
p.set_title('Customer Geography Distribution')

# 4. Plot NumOfProducts versus Churn:
plt.figure(figsize=(15,4))
p=sns.countplot(x='NumOfProducts', hue='Churn', data=data, palette="Greys_r")

legend = p.get_legend()
legend_txt = legend.texts
legend_txt[0].set_text("No Churn")
legend_txt[1].set_text("Churn")
p.set_title('Customer Distribution by Product')

# 5. Inspect Churn versus Age:
plt.figure(figsize=(15,4))
ax=sns.kdeplot(data.loc[(data['Churn'] == 0),'Age'] , color=sns.color_palette("Greys_r")[0], shade=True,label='no churn', linestyle='--')
ax=sns.kdeplot(data.loc[(data['Churn'] == 1),'Age'] , color=sns.color_palette("Greys_r")[1], shade=True, label='churn')
ax.set(xlabel='Customer Age', ylabel='Frequency')
plt.title('Customer Age - churn vs no churn')
plt.legend()

# 6. Plot Balance versus Churn:
plt.figure(figsize=(15,4))
ax=sns.kdeplot(data.loc[(data['Churn'] == 0),'Balance'] , color=sns.color_palette("Greys_r")[0], shade=True,label='no churn',linestyle='--')
ax=sns.kdeplot(data.loc[(data['Churn'] == 1),'Balance'] , color=sns.color_palette("Greys_r")[1], shade=True, label='churn')
ax.set(xlabel='Customer Balance', ylabel='Frequency')

plt.title('Customer Balance - churn vs no churn')
plt.legend()

# 7. Plot CreditScore versus Churn:
plt.figure(figsize=(15,4))
ax=sns.kdeplot(data.loc[(data['Churn'] == 0),'CreditScore'] , color=sns.color_palette("Greys_r")[0], shade=True,label='no churn',linestyle='--')
ax=sns.kdeplot(data.loc[(data['Churn'] == 1),'CreditScore'] ,  color=sns.color_palette("Greys_r")[1], shade=True, label='churn')
ax.set(xlabel='CreditScore', ylabel='Frequency')

plt.title('Customer CreditScore - churn vs no churn')
plt.legend()

```

---

#### **Activity 7.01: Performing the OSE technique from OSEMN**
A large telecom company wants to know why customers are churning. You are tasked with first finding out the reason behind the customer churn and then preparing a plan to reduce it. For this purpose, you have been provided with some data regarding the current bill amounts of customers (Current Bill Amt), the average number of calls made by each customer (Avg Calls), the average number of calls made by customers during weekdays (Avg Calls Weekdays), how long each account has been active (Account Age), and the average number of days the customer has defaulted on their bill payments (Avg Days Delinquent). To solve the first problem, you will use the OSE technique from OSEMN to carry out an initial exploration of the data.
_ Follow these steps:_

1.	Import the necessary libraries.

2.	Download the dataset and save it in a file called **Telco_Churn_Data.csv**.

3.	Read the Telco_Churn_Data.csv dataset and look at the first few rows of the dataset. You should get the following output:

![Figure 7.43: The first few rows of read.csv](images/Figure-7.43.jpg)

4.	Check the length and shape of the data (the number of rows and columns). The length should be 4708 and the shape should be (4708, 15).

5.	Rename all the columns in a readable format. Make the column names look consistent by separating them with _ instead of spaces, for example, rename 
Target Code to Target_Code. Also, fix the typo in the  
Avg_Hours_WorkOrderOpenned column. Your column names should finally look as follows.

![Figure 7.44: Renamed column names](images/Figure-7.44.jpg)

6.	Change the data type of the Target_Code,  
Condition_of_Current_Handset, and Current_TechSupComplaints columns from continuous to the categorical object type.

7.	Check for any missing values.

8.	Perform data exploration by initially exploring the Target_Churn variable. You should get the following summary:

![Figure 7.45: Summary of Target_Churn](images/Figure-7.45.jpg)

9.	Find the correlation among different variables and explain the results. You should get the following statistics:

![Figure 7.46: Correlation statistics of the variables](images/Figure-7.46.jpg)

You should get the following plot:

![Figure 7.47: Correlation plot of different features](images/Figure-7.47.jpg)

10.	Perform univariate and bivariate analyses.
For univariate analysis, use the following columns: Avg_Calls_Weekdays, Avg_Calls, and Current_Bill_Amt. You should get the following plots:

![Figure 7.48: Univariate analysis](images/Figure-7.48.jpg)

For bivariate analysis, you should get the following plots.
First, the plot of Complaint_Code versus Target_Churn:

![Figure 7.49: Customer complaint code distribution by churn](images/Figure-7.49.jpg)

Then, the plot of Acct_Plan_Subtype versus Target_Churn:
	
![Figure 7.50: Customer account plan subtype distribution by churn](images/Figure-7.50.jpg)  

Then, the plot of Current_TechSupComplaints versus Target_Churn:

![Figure 7.51: Customer technical support complaints distribution by churn](images/Figure-7.51.jpg)  

Next, the plot of Avg_Days_Delinquent versus Target_Code:

![Figure 7.52: Distribution of the average number of days delinquent by churn](images/Figure-7.52.jpg)  

Then, the plot of Account_Age versus Target_Code:

![Figure 7.53: Distribution of account age by churn](images/Figure-7.53.jpg)  

Lastly, the plot of Percent_Increase_MOM versus Target_Code:

![Figure 7.54: Distribution of the percentage increase of month-on-month  usage by churn/no ](images/Figure-7.54.jpg)  


**Code:**

```python

```

---

#### **Exercise 7.07: Performing Feature Selection**
In this exercise, you will be performing feature selection using a tree-based selection method that performs well on classification tasks. By the end of this exercise, you will be able to extract the most relevant features that can then be used for model building. 
You will be using a different kind of classifier called random forest in this exercise. While we will go into the details of this in the next chapter, the intention is to show how to perform feature selection using a given model. The process for using the random forest classifier is the same as logistic regression using scikit-learn, with the only difference being that instead of importing linear_model, you will need to use the sklearn.ensemble package to import RandomForestClassifier. The steps given in this exercise will provide more details about this.


**Code:**

```python
# 1.	Import RandomForestClassifier and train_test_split from the sklearn library:
from sklearn.ensemble import RandomForestClassifier from sklearn.model_selection import train_test_split

# 2.	Encode the categorical variable using the following code:
data.dtypes ### Encoding the categorical variables data["Geography"] = data["Geography"].astype('category')\                     .cat.codes data["Gender"] = data["Gender"].astype('category').cat.codes data["HasCrCard"] = data["HasCrCard"].astype('category')\                     .cat.codes data["Churn"] = data["Churn"].astype('category').cat.codes

# 3.	Split the data into training and testing sets as follows:
target = 'Churn' X = data.drop('Churn', axis=1) y=data[target]
X_train, X_test, y_train, y_test = train_test_split\                                    (X,y,test_size=0.15, \                                     random_state=123, \                                     stratify=y)

# 4.	Fit the model using the random forest classifier for feature selection with the following code:
forest=RandomForestClassifier(n_estimators=500,random_state=1)
forest.fit(X_train,y_train)

# 5.	Call the random forest feature_importances_ attribute to find the important features and store them in a variable named importances:
importances=forest.feature_importances_

# 6.	Create a variable named features to store all the columns, except the target Churn variable. Sort the important features present in the importances variable using NumPy's argsort function:
features = data.drop(['Churn'],axis=1).columns
indices = np.argsort(importances)[::-1]

# 7.	Plot the important features obtained from the random forest using Matplotlib's plt attribute:
plt.figure(figsize=(15,4)) plt.title("Feature importances using Random Forest") plt.bar(range(X_train.shape[1]), importances[indices],\         color="gray", align="center") plt.xticks(range(X_train.shape[1]), features[indices], \            rotation='vertical',fontsize=15) plt.xlim([-1, X_train.shape[1]]) plt.show()

# 8. Place the features and their importance in a pandas DataFrame using the following code:
feature_importance_df = pd.DataFrame({"Feature":features,\                                       "Importance":importances}) print(feature_importance_df)

```
---

#### **Exercise 7.08: Building a Logistic Regression Model**

In the previous exercise, you extracted the importance values of all the features. Next, you are asked to build a logistic regression model using the five most relevant features for predicting the churning of a customer. The customer's attributes are as follows:
•	Age: 50
•	EstimatedSalary: 100,000
•	CreditScore: 600
•	Balance: 100,000
•	NumOfProducts: 2
Logistic regression has been chosen as the base model for churn prediction because of its easy interpretability. 



**Code:**

```python
# 1.	Import the statsmodel package and select only the top five features that you got from the previous exercise to fit your model. Use the following code:
import statsmodels.api as sm 
top5_features = ['Age','EstimatedSalary','CreditScore',\                  'Balance','NumOfProducts'] logReg = sm.Logit(y_train, X_train[top5_features]) logistic_regression = logReg.fit()

# 2.	Once the model has been fitted, obtain the summary and your parameters:
logistic_regression.summary logistic_regression.params

# 3.	Create a function to compute the coefficients. This function will first multiply each feature by its coefficient (obtained in the previous step) and then finally add up the values for all the features in order to compute the final target value:
coef = logistic_regression.params
def y (coef, Age, EstimatedSalary, CreditScore, Balance, \        NumOfProducts) : return coef[0]*Age+ coef[1]\
                        *EstimatedSalary+coef[2]*CreditScore\
                        +coef[1]*Balance+coef[2]*NumOfProducts

# 4.	Calculate the chance of a customer churning by inputting the following values: 
Age: 50
EstimatedSalary: 100,000
CreditScore: 600
Balance: 100,000
NumOfProducts: 2
Use the following code (here, we are implementing the formula we saw in Figure 7.4):
import numpy as np y1 = y(coef, 50, 100000, 600,100000,2) p = np.exp(y1) / (1+np.exp(y1)) p

# 5.	In the previous steps, you learned how to use the statsmodel package. In this step, you will implement scikit-learn's LogisticRegression module to build your classifier and predict on the test data to find out the accuracy of our model:
from sklearn.linear_model import LogisticRegression

# 6.	Fit the logistic regression model on the partitioned training data that was prepared previously:
clf = LogisticRegression(random_state=0, solver='lbfgs')\
      .fit(X_train[top5_features], y_train)

# 7.	Call the predict and predict_proba functions on the test data:
clf.predict(X_test[top5_features]) clf.predict_proba(X_test[top5_features])
<img width="468" height="189" alt="image" src="https://github.com/user-attachments/assets/2b4610a3-0964-4e2f-946a-7dcf194983f0" />

# 8. Calculate the accuracy of the model by calling the score function:
clf.score(X_test[top5_features], y_test)

```

---

#### **Activity 7.02: Performing the MN technique from OSEMN**
You are working as a data scientist for a large telecom company. The marketing team wants to know the reasons behind customer churn. Using this information, they want to prepare a plan to reduce customer churn. Your task is to analyze the reasons behind the customer churn and present your findings.
After you have reported your initial findings to the marketing team, they want you to build a machine learning model that can predict customer churn. With your results, the marketing team can send out discount coupons to customers who might otherwise churn. Use the MN technique from OSEMN to construct your model.


1.	Import the necessary libraries.

2.	Encode the Acct_Plan_Subtype and Complaint_Code columns using the the.astype('category').cat.codes command.

3.	Split the data into training (80%) and testing sets (20%).

4.	Perform feature selection using the random forest classifier. You should get the following output:

![Figure 7.60: Feature importance using random forest](images/Figure-7.60.jpg)

5.	Select the top seven features and save them in a variable named  top7_features. 

6.	Fit a logistic regression using the statsmodel package.

7.	Find out the probability that a customer will churn when the following data is used: Avg_Days_Delinquent: 40, Percent_Increase_MOM: 5,  Avg_Calls_Weekdays: 39000, Current_Bill_Amt: 12000,  Avg_Calls: 9000, Complaint_Code: 0, and Account_Age: 17.
The given customer should have a value of around 81.939% likelihood of churning.

---
## Bài tập tổng hợp

### Bài Tập 1: Cải Tiến Phương Pháp Customer Segmentation với Kỹ Thuật Clustering Hiện Đại

### Mô Tả Bài Toán
Bạn là Data Scientist tại một công ty thương mại điện tử. Công ty muốn cải tiến chiến lược phân khúc khách hàng hiện tại bằng cách sử dụng các kỹ thuật clustering hiện đại thay vì phương pháp truyền thống.

### Dataset
Sử dụng dữ liệu khách hàng với các đặc trưng:
- `customer_id`: ID khách hàng
- `recency`: Số ngày kể từ lần mua hàng cuối
- `frequency`: Tần suất mua hàng trong năm
- `monetary`: Tổng giá trị đơn hàng
- `avg_order_value`: Giá trị đơn hàng trung bình
- `days_since_first_purchase`: Số ngày từ lần mua đầu tiên
- `product_categories`: Số danh mục sản phẩm đã mua

### Yêu Cầu Thực Hiện

#### Phần A: Chuẩn Bị Dữ Liệu và EDA
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

# Tạo dữ liệu mẫu
np.random.seed(42)
n_customers = 2000

# Tạo 4 segment khách hàng khác nhau
segments = []
for i in range(4):
    segment_size = n_customers // 4
    if i == 0:  # High Value Customers
        segment = {
            'recency': np.random.normal(15, 5, segment_size),
            'frequency': np.random.normal(25, 5, segment_size),
            'monetary': np.random.normal(5000, 1000, segment_size),
            'avg_order_value': np.random.normal(200, 50, segment_size),
            'days_since_first_purchase': np.random.normal(400, 100, segment_size),
            'product_categories': np.random.normal(8, 2, segment_size)
        }
    elif i == 1:  # Regular Customers
        segment = {
            'recency': np.random.normal(45, 10, segment_size),
            'frequency': np.random.normal(12, 3, segment_size),
            'monetary': np.random.normal(2000, 500, segment_size),
            'avg_order_value': np.random.normal(100, 30, segment_size),
            'days_since_first_purchase': np.random.normal(200, 50, segment_size),
            'product_categories': np.random.normal(5, 1, segment_size)
        }
    elif i == 2:  # At Risk Customers
        segment = {
            'recency': np.random.normal(120, 30, segment_size),
            'frequency': np.random.normal(8, 2, segment_size),
            'monetary': np.random.normal(1500, 400, segment_size),
            'avg_order_value': np.random.normal(80, 20, segment_size),
            'days_since_first_purchase': np.random.normal(300, 80, segment_size),
            'product_categories': np.random.normal(3, 1, segment_size)
        }
    else:  # Lost Customers
        segment = {
            'recency': np.random.normal(200, 50, segment_size),
            'frequency': np.random.normal(3, 1, segment_size),
            'monetary': np.random.normal(500, 200, segment_size),
            'avg_order_value': np.random.normal(60, 15, segment_size),
            'days_since_first_purchase': np.random.normal(500, 150, segment_size),
            'product_categories': np.random.normal(2, 0.5, segment_size)
        }
    segments.append(pd.DataFrame(segment))

# Kết hợp tất cả segments
df = pd.concat(segments, ignore_index=True)
df['customer_id'] = range(1, len(df) + 1)

# Đảm bảo giá trị dương
for col in df.columns:
    if col != 'customer_id':
        df[col] = np.maximum(df[col], 1)

print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.describe())
```

**Nhiệm vụ 1.1**: Thực hiện EDA chi tiết
- Vẽ distribution plots cho từng feature
- Tạo correlation matrix
- Phân tích outliers bằng boxplots
- Tính toán và visualize skewness của các biến

**Nhiệm vụ 1.2**: So sánh các phương pháp scaling
```python
# So sánh StandardScaler vs RobustScaler
scalers = {
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler()
}

# Thực hiện scaling và so sánh kết quả
```

#### Phần B: Implement Clustering Algorithms Hiện Đại

**Nhiệm vụ 1.3**: Implement và so sánh các thuật toán clustering
```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class ModernClusteringComparison:
    def __init__(self, data):
        self.data = data
        self.results = {}
    
    def fit_kmeans_variants(self, n_clusters=4):
        """So sánh các variant của K-Means"""
        kmeans_variants = {
            'K-Means (Lloyd)': KMeans(n_clusters=n_clusters, algorithm='lloyd', random_state=42),
            'K-Means (Elkan)': KMeans(n_clusters=n_clusters, algorithm='elkan', random_state=42),
            'K-Means++': KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        }
        
        for name, model in kmeans_variants.items():
            labels = model.fit_predict(self.data)
            self.results[name] = {
                'labels': labels,
                'silhouette': silhouette_score(self.data, labels),
                'calinski_harabasz': calinski_harabasz_score(self.data, labels),
                'davies_bouldin': davies_bouldin_score(self.data, labels),
                'inertia': model.inertia_
            }
    
    def fit_gaussian_mixture(self, n_components=4):
        """Gaussian Mixture Models với các covariance types"""
        covariance_types = ['full', 'tied', 'diag', 'spherical']
        
        for cov_type in covariance_types:
            gmm = GaussianMixture(n_components=n_components, 
                                covariance_type=cov_type, 
                                random_state=42)
            labels = gmm.fit_predict(self.data)
            
            self.results[f'GMM ({cov_type})'] = {
                'labels': labels,
                'silhouette': silhouette_score(self.data, labels),
                'calinski_harabasz': calinski_harabasz_score(self.data, labels),
                'davies_bouldin': davies_bouldin_score(self.data, labels),
                'aic': gmm.aic(self.data),
                'bic': gmm.bic(self.data)
            }
    
    def fit_hierarchical_clustering(self):
        """Hierarchical Clustering với các linkage methods"""
        linkage_methods = ['ward', 'complete', 'average', 'single']
        
        for linkage in linkage_methods:
            if linkage == 'ward':
                model = AgglomerativeClustering(n_clusters=4, linkage=linkage)
            else:
                model = AgglomerativeClustering(n_clusters=4, linkage=linkage, 
                                              metric='euclidean')
            labels = model.fit_predict(self.data)
            
            self.results[f'Hierarchical ({linkage})'] = {
                'labels': labels,
                'silhouette': silhouette_score(self.data, labels),
                'calinski_harabasz': calinski_harabasz_score(self.data, labels),
                'davies_bouldin': davies_bouldin_score(self.data, labels)
            }
    
    def compare_results(self):
        """So sánh kết quả của tất cả các thuật toán"""
        comparison_df = pd.DataFrame({
            'Algorithm': list(self.results.keys()),
            'Silhouette Score': [self.results[alg]['silhouette'] for alg in self.results.keys()],
            'Calinski-Harabasz': [self.results[alg]['calinski_harabasz'] for alg in self.results.keys()],
            'Davies-Bouldin': [self.results[alg]['davies_bouldin'] for alg in self.results.keys()]
        })
        
        return comparison_df.sort_values('Silhouette Score', ascending=False)

# Sử dụng class
features = ['recency', 'frequency', 'monetary', 'avg_order_value', 
           'days_since_first_purchase', 'product_categories']
X_scaled = StandardScaler().fit_transform(df[features])

clustering_comparison = ModernClusteringComparison(X_scaled)
clustering_comparison.fit_kmeans_variants()
clustering_comparison.fit_gaussian_mixture()
clustering_comparison.fit_hierarchical_clustering()

results_comparison = clustering_comparison.compare_results()
print(results_comparison)
```

#### Phần C: Cluster Evaluation và Interpretation

**Nhiệm vụ 1.4**: Tạo comprehensive evaluation framework
```python
def comprehensive_cluster_evaluation(X, labels, original_data):
    """
    Đánh giá toàn diện các cluster được tạo
    """
    evaluation_metrics = {}
    
    # Internal metrics
    evaluation_metrics['silhouette_score'] = silhouette_score(X, labels)
    evaluation_metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
    evaluation_metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
    
    # Business metrics
    cluster_profiles = original_data.copy()
    cluster_profiles['cluster'] = labels
    
    # Tính toán business metrics cho từng cluster
    business_metrics = cluster_profiles.groupby('cluster').agg({
        'recency': ['mean', 'std'],
        'frequency': ['mean', 'std'],
        'monetary': ['mean', 'std', 'sum'],
        'avg_order_value': ['mean', 'std'],
        'days_since_first_purchase': ['mean', 'std'],
        'product_categories': ['mean', 'std']
    })
    
    # Cluster size distribution
    cluster_sizes = cluster_profiles['cluster'].value_counts().sort_index()
    evaluation_metrics['cluster_sizes'] = cluster_sizes
    evaluation_metrics['cluster_balance'] = cluster_sizes.std() / cluster_sizes.mean()
    
    return evaluation_metrics, business_metrics

# Áp dụng evaluation
best_algorithm = 'K-Means++'  # Từ kết quả comparison
best_labels = clustering_comparison.results[best_algorithm]['labels']

eval_metrics, business_profiles = comprehensive_cluster_evaluation(
    X_scaled, best_labels, df[features]
)
```

---

## Bài Tập 2: Xác Định Số Cluster Tối Ưu Một Cách Có Nguyên Tắc

### Mô Tả Bài Toán
Phát triển một framework toàn diện để xác định số cluster tối ưu cho customer segmentation, đảm bảo các segment có ý nghĩa thống kê và khả thi trong kinh doanh.

### Yêu Cầu Thực Hiện

#### Phần A: Multiple Methods for Optimal K Selection

**Nhiệm vụ 2.1**: Implement các phương pháp xác định K tối ưu
```python
class OptimalClusterSelector:
    def __init__(self, data, max_clusters=15):
        self.data = data
        self.max_clusters = max_clusters
        self.results = {}
        
    def elbow_method(self):
        """Elbow Method với improved detection"""
        inertias = []
        k_range = range(1, self.max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.data)
            inertias.append(kmeans.inertia_)
        
        # Tính gradient để detect elbow point
        gradients = np.diff(inertias)
        second_gradients = np.diff(gradients)
        
        # Elbow point là điểm có second gradient lớn nhất (most negative)
        elbow_point = np.argmax(second_gradients) + 2
        
        self.results['elbow'] = {
            'k_range': k_range,
            'inertias': inertias,
            'optimal_k': elbow_point,
            'gradients': gradients,
            'second_gradients': second_gradients
        }
        
        return elbow_point
    
    def silhouette_analysis(self):
        """Silhouette Analysis với detailed scores"""
        silhouette_scores = []
        k_range = range(2, self.max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.data)
            score = silhouette_score(self.data, labels)
            silhouette_scores.append(score)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        self.results['silhouette'] = {
            'k_range': k_range,
            'scores': silhouette_scores,
            'optimal_k': optimal_k
        }
        
        return optimal_k
    
    def gap_statistic(self, n_refs=10):
        """Gap Statistic method"""
        def compute_inertia(data, k):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            return kmeans.inertia_
        
        k_range = range(1, self.max_clusters + 1)
        gaps = []
        errors = []
        
        for k in k_range:
            # Original data inertia
            original_inertia = compute_inertia(self.data, k)
            
            # Reference data inertias
            ref_inertias = []
            for _ in range(n_refs):
                # Generate reference data
                ref_data = np.random.uniform(
                    low=self.data.min(axis=0),
                    high=self.data.max(axis=0),
                    size=self.data.shape
                )
                ref_inertia = compute_inertia(ref_data, k)
                ref_inertias.append(ref_inertia)
            
            # Gap statistic
            gap = np.log(np.mean(ref_inertias)) - np.log(original_inertia)
            error = np.sqrt(1 + 1/n_refs) * np.std(np.log(ref_inertias))
            
            gaps.append(gap)
            errors.append(error)
        
        # Find optimal k using Gap(k) >= Gap(k+1) - s_{k+1}
        optimal_k = 1
        for i in range(len(gaps) - 1):
            if gaps[i] >= gaps[i + 1] - errors[i + 1]:
                optimal_k = k_range[i]
                break
        
        self.results['gap_statistic'] = {
            'k_range': k_range,
            'gaps': gaps,
            'errors': errors,
            'optimal_k': optimal_k
        }
        
        return optimal_k
    
    def calinski_harabasz_method(self):
        """Calinski-Harabasz Index method"""
        ch_scores = []
        k_range = range(2, self.max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.data)
            score = calinski_harabasz_score(self.data, labels)
            ch_scores.append(score)
        
        optimal_k = k_range[np.argmax(ch_scores)]
        
        self.results['calinski_harabasz'] = {
            'k_range': k_range,
            'scores': ch_scores,
            'optimal_k': optimal_k
        }
        
        return optimal_k
    
    def davies_bouldin_method(self):
        """Davies-Bouldin Index method (lower is better)"""
        db_scores = []
        k_range = range(2, self.max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.data)
            score = davies_bouldin_score(self.data, labels)
            db_scores.append(score)
        
        optimal_k = k_range[np.argmin(db_scores)]
        
        self.results['davies_bouldin'] = {
            'k_range': k_range,
            'scores': db_scores,
            'optimal_k': optimal_k
        }
        
        return optimal_k
    
    def consensus_optimal_k(self):
        """Tìm consensus từ tất cả các phương pháp"""
        methods = ['elbow', 'silhouette', 'gap_statistic', 'calinski_harabasz', 'davies_bouldin']
        optimal_ks = []
        
        for method in methods:
            if method == 'elbow':
                k = self.elbow_method()
            elif method == 'silhouette':
                k = self.silhouette_analysis()
            elif method == 'gap_statistic':
                k = self.gap_statistic()
            elif method == 'calinski_harabasz':
                k = self.calinski_harabasz_method()
            elif method == 'davies_bouldin':
                k = self.davies_bouldin_method()
            
            optimal_ks.append(k)
        
        # Tìm mode (giá trị xuất hiện nhiều nhất)
        consensus_k = max(set(optimal_ks), key=optimal_ks.count)
        
        consensus_results = pd.DataFrame({
            'Method': methods,
            'Optimal_K': optimal_ks
        })
        
        return consensus_k, consensus_results
    
    def plot_all_methods(self):
        """Visualize kết quả của tất cả các phương pháp"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        # Elbow Method
        axes[0].plot(self.results['elbow']['k_range'], self.results['elbow']['inertias'], 'bo-')
        axes[0].axvline(x=self.results['elbow']['optimal_k'], color='red', linestyle='--', 
                       label=f'Optimal K = {self.results["elbow"]["optimal_k"]}')
        axes[0].set_title('Elbow Method')
        axes[0].set_xlabel('Number of Clusters (K)')
        axes[0].set_ylabel('Inertia')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette Analysis
        axes[1].plot(self.results['silhouette']['k_range'], self.results['silhouette']['scores'], 'go-')
        axes[1].axvline(x=self.results['silhouette']['optimal_k'], color='red', linestyle='--',
                       label=f'Optimal K = {self.results["silhouette"]["optimal_k"]}')
        axes[1].set_title('Silhouette Analysis')
        axes[1].set_xlabel('Number of Clusters (K)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Gap Statistic
        axes[2].errorbar(self.results['gap_statistic']['k_range'], 
                        self.results['gap_statistic']['gaps'],
                        yerr=self.results['gap_statistic']['errors'], 
                        fmt='ro-', capsize=5)
        axes[2].axvline(x=self.results['gap_statistic']['optimal_k'], color='red', linestyle='--',
                       label=f'Optimal K = {self.results["gap_statistic"]["optimal_k"]}')
        axes[2].set_title('Gap Statistic')
        axes[2].set_xlabel('Number of Clusters (K)')
        axes[2].set_ylabel('Gap Statistic')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Calinski-Harabasz
        axes[3].plot(self.results['calinski_harabasz']['k_range'], 
                    self.results['calinski_harabasz']['scores'], 'mo-')
        axes[3].axvline(x=self.results['calinski_harabasz']['optimal_k'], color='red', linestyle='--',
                       label=f'Optimal K = {self.results["calinski_harabasz"]["optimal_k"]}')
        axes[3].set_title('Calinski-Harabasz Index')
        axes[3].set_xlabel('Number of Clusters (K)')
        axes[3].set_ylabel('CH Score')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # Davies-Bouldin
        axes[4].plot(self.results['davies_bouldin']['k_range'], 
                    self.results['davies_bouldin']['scores'], 'co-')
        axes[4].axvline(x=self.results['davies_bouldin']['optimal_k'], color='red', linestyle='--',
                       label=f'Optimal K = {self.results["davies_bouldin"]["optimal_k"]}')
        axes[4].set_title('Davies-Bouldin Index')
        axes[4].set_xlabel('Number of Clusters (K)')
        axes[4].set_ylabel('DB Score')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        
        # Summary plot
        methods_data = []
        for method, result in self.results.items():
            methods_data.append({
                'Method': method.replace('_', ' ').title(),
                'Optimal K': result['optimal_k']
            })
        
        methods_df = pd.DataFrame(methods_data)
        axes[5].bar(methods_df['Method'], methods_df['Optimal K'], color='skyblue', edgecolor='navy')
        axes[5].set_title('Optimal K by Different Methods')
        axes[5].set_xlabel('Methods')
        axes[5].set_ylabel('Optimal K')
        axes[5].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# Sử dụng class
selector = OptimalClusterSelector(X_scaled, max_clusters=10)
consensus_k, methods_summary = selector.consensus_optimal_k()
selector.plot_all_methods()

print(f"Consensus Optimal K: {consensus_k}")
print("\nMethods Summary:")
print(methods_summary)
```

#### Phần B: Business-Driven Cluster Validation

**Nhiệm vụ 2.2**: Tạo business validation framework
```python
class BusinessClusterValidator:
    def __init__(self, data, features, business_metrics):
        self.data = data
        self.features = features
        self.business_metrics = business_metrics
        
    def validate_cluster_actionability(self, labels):
        """
        Kiểm tra tính khả thi của clusters trong kinh doanh
        """
        cluster_data = self.data.copy()
        cluster_data['cluster'] = labels
        
        validation_results = {}
        
        # 1. Cluster Size Adequacy
        cluster_sizes = cluster_data['cluster'].value_counts()
        min_viable_size = len(self.data) * 0.05  # Ít nhất 5% của total customers
        
        validation_results['size_adequacy'] = {
            'min_size': cluster_sizes.min(),
            'max_size': cluster_sizes.max(),
            'min_viable_size': min_viable_size,
            'all_adequate': cluster_sizes.min() >= min_viable_size,
            'cluster_sizes': cluster_sizes.to_dict()
        }
        
        # 2. Statistical Separation
        separation_scores = {}
        for metric in self.business_metrics:
            cluster_means = cluster_data.groupby('cluster')[metric].mean()
            overall_std = cluster_data[metric].std()
            
            # Tính Cohen's d between clusters
            cohens_d_matrix = np.zeros((len(cluster_means), len(cluster_means)))
            for i, cluster1 in enumerate(cluster_means.index):
                for j, cluster2 in enumerate(cluster_means.index):
                    if i != j:
                        mean_diff = abs(cluster_means.iloc[i] - cluster_means.iloc[j])
                        cohens_d = mean_diff / overall_std
                        cohens_d_matrix[i, j] = cohens_d
            
            separation_scores[metric] = {
                'min_cohens_d': cohens_d_matrix[cohens_d_matrix > 0].min(),
                'max_cohens_d': cohens_d_matrix.max(),
                'avg_cohens_d': cohens_d_matrix[cohens_d_matrix > 0].mean()
            }
        
        validation_results['statistical_separation'] = separation_scores
        
        # 3. Business Interpretability
        cluster_profiles = cluster_data.groupby('cluster')[self.business_metrics].agg(['mean', 'std'])
        
        # RFM-like interpretation
        interpretations = {}
        for cluster_id in cluster_data['cluster'].unique():
            cluster_subset = cluster_data[cluster_data['cluster'] == cluster_id]
            
            # Define cluster characteristics
            recency_level = 'Low' if cluster_subset['recency'].mean() < cluster_data['recency'].quantile(0.33) else \
                          'Medium' if cluster_subset['recency'].mean() < cluster_data['recency'].quantile(0.67) else 'High'
            
            frequency_level = 'Low' if cluster_subset['frequency'].mean() < cluster_data['frequency'].quantile(0.33) else \
                            'Medium' if cluster_subset['frequency'].mean() < cluster_data['frequency'].quantile(0.67) else 'High'
            
            monetary_level = 'Low' if cluster_subset['monetary'].mean() < cluster_data['monetary'].quantile(0.33) else \
                           'Medium' if cluster_subset['monetary'].mean() < cluster_data['monetary'].quantile(0.67) else 'High'
            
            interpretations[cluster_id] = {
                'recency': recency_level,
                'frequency': frequency_level,
                'monetary': monetary_level,
                'suggested_name': f"R:{recency_level[0]}-F:{frequency_level[0]}-M:{monetary_level[0]}",
                'size': len(cluster_subset),
                'percentage': len(cluster_subset) / len(cluster_data) * 100
            }
        
        validation_results['business_interpretability'] = interpretations
        
        return validation_results
    
    def stability_analysis(self, n_iterations=10, sample_ratio=0.8):
        """
        Phân tích stability của clustering qua multiple runs
        """
        stability_scores = []
        
        for iteration in range(n_iterations):
            # Random sampling
            sample_size = int(len(self.data) * sample_ratio)
            sample_indices = np.random.choice(len(self.data), sample_size, replace=False)
            
            sample_data = self.data.iloc[sample_indices][self.features]
            sample_scaled = StandardScaler().fit_transform(sample_data)
            
            # Clustering
            kmeans = KMeans(n_clusters=4, random_state=iteration)
            labels = kmeans.fit_predict(sample_scaled)
            
            # Calculate stability metric (silhouette score)
            stability_score = silhouette_score(sample_scaled, labels)
            stability_scores.append(stability_score)
        
        return {
            'mean_stability': np.mean(stability_scores),
            'std_stability': np.std(stability_scores),
            'stability_scores': stability_scores,
            'coefficient_of_variation': np.std(stability_scores) / np.mean(stability_scores)
        }

# Sử dụng Business Validator
business_metrics = ['recency', 'frequency', 'monetary', 'avg_order_value']
validator = BusinessClusterValidator(df, features, business_metrics)

# Validate với optimal K
kmeans_optimal = KMeans(n_clusters=consensus_k, random_state=42)
optimal_labels = kmeans_optimal.fit_predict(X_scaled)

business_validation = validator.validate_cluster_actionability(optimal_labels)
stability_results = validator.stability_analysis()

print("Business Validation Results:")
print(f"All clusters adequate size: {business_validation['size_adequacy']['all_adequate']}")
print(f"Stability coefficient of variation: {stability_results['coefficient_of_variation']:.3f}")
```

---

## Bài Tập 3: Áp Dụng Evaluation Approaches cho Multiple Business Problems

### Mô Tả Bài Toán
Áp dụng các phương pháp đánh giá cluster cho 3 bài toán kinh doanh khác nhau: E-commerce, Banking, và Telecommunications.

### Dataset cho Multiple Domains

**Nhiệm vụ 3.1**: Tạo domain-specific datasets
```python
class MultiDomainDataGenerator:
    @staticmethod
    def generate_ecommerce_data(n_customers=1500):
        """E-commerce customer data"""
        np.random.seed(42)
        
        # 5 segments: Champions, Loyal, Potential Loyalists, New Customers, At Risk
        segments_config = [
            {'name': 'Champions', 'size': 0.2, 'recency': (1, 10), 'frequency': (15, 25), 
             'monetary': (3000, 5000), 'avg_session_duration': (20, 30), 'bounce_rate': (0.1, 0.3)},
            {'name': 'Loyal', 'size': 0.25, 'recency': (10, 30), 'frequency': (8, 15), 
             'monetary': (1500, 3000), 'avg_session_duration': (15, 25), 'bounce_rate': (0.2, 0.4)},
            {'name': 'Potential Loyalists', 'size': 0.2, 'recency': (5, 20), 'frequency': (3, 8), 
             'monetary': (800, 1500), 'avg_session_duration': (10, 20), 'bounce_rate': (0.3, 0.5)},
            {'name': 'New Customers', 'size': 0.15, 'recency': (1, 15), 'frequency': (1, 3), 
             'monetary': (200, 800), 'avg_session_duration': (5, 15), 'bounce_rate': (0.4, 0.7)},
            {'name': 'At Risk', 'size': 0.2, 'recency': (50, 100), 'frequency': (5, 12), 
             'monetary': (1000, 2500), 'avg_session_duration': (5, 10), 'bounce_rate': (0.6, 0.8)}
        ]
        
        data = []
        for segment in segments_config:
            size = int(n_customers * segment['size'])
            segment_data = {
                'customer_id': range(len(data), len(data) + size),
                'recency': np.random.uniform(segment['recency'][0], segment['recency'][1], size),
                'frequency': np.random.uniform(segment['frequency'][0], segment['frequency'][1], size),
                'monetary': np.random.uniform(segment['monetary'][0], segment['monetary'][1], size),
                'avg_session_duration': np.random.uniform(segment['avg_session_duration'][0], 
                                                        segment['avg_session_duration'][1], size),
                'bounce_rate': np.random.uniform(segment['bounce_rate'][0], segment['bounce_rate'][1], size),
                'true_segment': [segment['name']] * size
            }
            data.append(pd.DataFrame(segment_data))
        
        return pd.concat(data, ignore_index=True)
    
    @staticmethod
    def generate_banking_data(n_customers=1500):
        """Banking customer data"""
        np.random.seed(42)
        
        segments_config = [
            {'name': 'High Value', 'size': 0.15, 'balance': (50000, 200000), 'transaction_count': (20, 50),
             'credit_score': (750, 850), 'products_count': (4, 8), 'digital_engagement': (0.8, 1.0)},
            {'name': 'Mass Affluent', 'size': 0.25, 'balance': (15000, 50000), 'transaction_count': (10, 25),
             'credit_score': (650, 750), 'products_count': (2, 5), 'digital_engagement': (0.6, 0.8)},
            {'name': 'Mainstream', 'size': 0.35, 'balance': (2000, 15000), 'transaction_count': (5, 15),
             'credit_score': (550, 700), 'products_count': (1, 3), 'digital_engagement': (0.4, 0.7)},
            {'name': 'Young Professionals', 'size': 0.15, 'balance': (1000, 8000), 'transaction_count': (8, 20),
             'credit_score': (600, 750), 'products_count': (2, 4), 'digital_engagement': (0.8, 1.0)},
            {'name': 'Inactive', 'size': 0.1, 'balance': (100, 2000), 'transaction_count': (0, 5),
             'credit_score': (400, 600), 'products_count': (1, 2), 'digital_engagement': (0.0, 0.3)}
        ]
        
        data = []
        for segment in segments_config:
            size = int(n_customers * segment['size'])
            segment_data = {
                'customer_id': range(len(data), len(data) + size),
                'account_balance': np.random.uniform(segment['balance'][0], segment['balance'][1], size),
                'monthly_transactions': np.random.uniform(segment['transaction_count'][0], 
                                                        segment['transaction_count'][1], size),
                'credit_score': np.random.uniform(segment['credit_score'][0], segment['credit_score'][1], size),
                'products_owned': np.random.uniform(segment['products_count'][0], 
                                                   segment['products_count'][1], size),
                'digital_engagement_score': np.random.uniform(segment['digital_engagement'][0], 
                                                            segment['digital_engagement'][1], size),
                'true_segment': [segment['name']] * size
            }
            data.append(pd.DataFrame(segment_data))
        
        return pd.concat(data, ignore_index=True)
    
    @staticmethod
    def generate_telecom_data(n_customers=1500):
        """Telecommunications customer data"""
        np.random.seed(42)
        
        segments_config = [
            {'name': 'Heavy Users', 'size': 0.2, 'monthly_minutes': (800, 1500), 'data_usage': (15, 30),
             'monthly_revenue': (80, 150), 'tenure': (24, 60), 'customer_service_calls': (0, 2)},
            {'name': 'Standard Users', 'size': 0.4, 'monthly_minutes': (300, 800), 'data_usage': (5, 15),
             'monthly_revenue': (40, 80), 'tenure': (12, 36), 'customer_service_calls': (1, 4)},
            {'name': 'Light Users', 'size': 0.2, 'monthly_minutes': (50, 300), 'data_usage': (1, 5),
             'monthly_revenue': (20, 40), 'tenure': (6, 24), 'customer_service_calls': (0, 3)},
            {'name': 'Business Users', 'size': 0.1, 'monthly_minutes': (1000, 2000), 'data_usage': (20, 40),
             'monthly_revenue': (100, 200), 'tenure': (12, 48), 'customer_service_calls': (2, 6)},
            {'name': 'Churners', 'size': 0.1, 'monthly_minutes': (100, 400), 'data_usage': (2, 8),
             'monthly_revenue': (25, 50), 'tenure': (1, 12), 'customer_service_calls': (3, 8)}
        ]
        
        data = []
        for segment in segments_config:
            size = int(n_customers * segment['size'])
            segment_data = {
                'customer_id': range(len(data), len(data) + size),
                'monthly_voice_minutes': np.random.uniform(segment['monthly_minutes'][0], 
                                                         segment['monthly_minutes'][1], size),
                'monthly_data_gb': np.random.uniform(segment['data_usage'][0], segment['data_usage'][1], size),
                'monthly_revenue': np.random.uniform(segment['monthly_revenue'][0], 
                                                   segment['monthly_revenue'][1], size),
                'tenure_months': np.random.uniform(segment['tenure'][0], segment['tenure'][1], size),
                'service_calls': np.random.uniform(segment['customer_service_calls'][0], 
                                                 segment['customer_service_calls'][1], size),
                'true_segment': [segment['name']] * size
            }
            data.append(pd.DataFrame(segment_data))
        
        return pd.concat(data, ignore_index=True)

# Generate datasets
ecommerce_data = MultiDomainDataGenerator.generate_ecommerce_data()
banking_data = MultiDomainDataGenerator.generate_banking_data()
telecom_data = MultiDomainDataGenerator.generate_telecom_data()

print("E-commerce data shape:", ecommerce_data.shape)
print("Banking data shape:", banking_data.shape)
print("Telecom data shape:", telecom_data.shape)
```

#### Phần A: Domain-Specific Evaluation Metrics

**Nhiệm vụ 3.2**: Tạo domain-specific evaluation framework
```python
class DomainSpecificEvaluator:
    def __init__(self, domain_type):
        self.domain_type = domain_type
        self.domain_weights = self._get_domain_weights()
        
    def _get_domain_weights(self):
        """Trọng số cho từng metric theo domain"""
        weights = {
            'ecommerce': {
                'recency': 0.3,
                'frequency': 0.25,
                'monetary': 0.35,
                'engagement': 0.1
            },
            'banking': {
                'balance': 0.4,
                'transactions': 0.2,
                'credit_score': 0.25,
                'products': 0.15
            },
            'telecom': {
                'usage': 0.3,
                'revenue': 0.35,
                'tenure': 0.2,
                'satisfaction': 0.15
            }
        }
        return weights.get(self.domain_type, {})
    
    def calculate_business_value_score(self, data, labels):
        """Tính Business Value Score cho từng cluster"""
        cluster_data = data.copy()
        cluster_data['cluster'] = labels
        
        if self.domain_type == 'ecommerce':
            return self._ecommerce_business_value(cluster_data)
        elif self.domain_type == 'banking':
            return self._banking_business_value(cluster_data)
        elif self.domain_type == 'telecom':
            return self._telecom_business_value(cluster_data)
    
    def _ecommerce_business_value(self, cluster_data):
        """E-commerce specific business value calculation"""
        cluster_values = {}
        
        for cluster_id in cluster_data['cluster'].unique():
            cluster_subset = cluster_data[cluster_data['cluster'] == cluster_id]
            
            # Customer Lifetime Value approximation
            avg_frequency = cluster_subset['frequency'].mean()
            avg_monetary = cluster_subset['monetary'].mean()
            avg_recency = cluster_subset['recency'].mean()
            
            # CLV = (Average Order Value × Purchase Frequency × Gross Margin × Lifespan)
            # Simplified: Higher frequency and monetary, lower recency = higher value
            clv_score = (avg_monetary * avg_frequency) / (avg_recency + 1)
            
            # Engagement score
            avg_session = cluster_subset['avg_session_duration'].mean()
            avg_bounce = cluster_subset['bounce_rate'].mean()
            engagement_score = avg_session * (1 - avg_bounce)
            
            # Weighted business value
            business_value = (
                self.domain_weights['monetary'] * (avg_monetary / cluster_data['monetary'].max()) +
                self.domain_weights['frequency'] * (avg_frequency / cluster_data['frequency'].max()) +
                self.domain_weights['recency'] * (1 - avg_recency / cluster_data['recency'].max()) +
                self.domain_weights['engagement'] * (engagement_score / 
                    (cluster_data['avg_session_duration'] * (1 - cluster_data['bounce_rate'])).max())
            )
            
            cluster_values[cluster_id] = {
                'business_value_score': business_value,
                'clv_approximation': clv_score,
                'size': len(cluster_subset),
                'avg_monetary': avg_monetary,
                'avg_frequency': avg_frequency,
                'avg_recency': avg_recency
            }
        
        return cluster_values
    
    def _banking_business_value(self, cluster_data):
        """Banking specific business value calculation"""
        cluster_values = {}
        
        for cluster_id in cluster_data['cluster'].unique():
            cluster_subset = cluster_data[cluster_data['cluster'] == cluster_id]
            
            # Profitability indicators
            avg_balance = cluster_subset['account_balance'].mean()
            avg_transactions = cluster_subset['monthly_transactions'].mean()
            avg_credit_score = cluster_subset['credit_score'].mean()
            avg_products = cluster_subset['products_owned'].mean()
            
            # Revenue potential (balance × products × transaction activity)
            revenue_potential = avg_balance * avg_products * (avg_transactions / 10)
            
            # Risk adjustment (credit score)
            risk_factor = avg_credit_score / 850  # Normalize to 0-1
            
            # Weighted business value
            business_value = (
                self.domain_weights['balance'] * (avg_balance / cluster_data['account_balance'].max()) +
                self.domain_weights['transactions'] * (avg_transactions / cluster_data['monthly_transactions'].max()) +
                self.domain_weights['credit_score'] * (avg_credit_score / 850) +
                self.domain_weights['products'] * (avg_products / cluster_data['products_owned'].max())
            )
            
            cluster_values[cluster_id] = {
                'business_value_score': business_value,
                'revenue_potential': revenue_potential,
                'risk_factor': risk_factor,
                'size': len(cluster_subset),
                'avg_balance': avg_balance,
                'avg_products': avg_products
            }
        
        return cluster_values
    
    def _telecom_business_value(self, cluster_data):
        """Telecom specific business value calculation"""
        cluster_values = {}
        
        for cluster_id in cluster_data['cluster'].unique():
            cluster_subset = cluster_data[cluster_data['cluster'] == cluster_id]
            
            avg_voice = cluster_subset['monthly_voice_minutes'].mean()
            avg_data = cluster_subset['monthly_data_gb'].mean()
            avg_revenue = cluster_subset['monthly_revenue'].mean()
            avg_tenure = cluster_subset['tenure_months'].mean()
            avg_service_calls = cluster_subset['service_calls'].mean()
            
            # Usage intensity
            usage_score = (avg_voice + avg_data * 100) / 1000  # Normalize
            
            # Customer satisfaction proxy (fewer service calls = higher satisfaction)
            satisfaction_score = max(0, 1 - avg_service_calls / 10)
            
            # Customer lifetime value (revenue × tenure)
            clv_estimate = avg_revenue * avg_tenure
            
            # Weighted business value
            business_value = (
                self.domain_weights['usage'] * usage_score +
                self.domain_weights['revenue'] * (avg_revenue / cluster_data['monthly_revenue'].max()) +
                self.domain_weights['tenure'] * (avg_tenure / cluster_data['tenure_months'].max()) +
                self.domain_weights['satisfaction'] * satisfaction_score
            )
            
            cluster_values[cluster_id] = {
                'business_value_score': business_value,
                'clv_estimate': clv_estimate,
                'usage_intensity': usage_score,
                'satisfaction_proxy': satisfaction_score,
                'size': len(cluster_subset),
                'avg_revenue': avg_revenue
            }
        
        return cluster_values
    
    def evaluate_clustering_quality(self, X, labels, original_data):
        """Comprehensive clustering evaluation"""
        # Technical metrics
        silhouette = silhouette_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        # Business metrics
        business_values = self.calculate_business_value_score(original_data, labels)
        
        # Cluster balance
        cluster_sizes = pd.Series(labels).value_counts()
        balance_score = 1 - (cluster_sizes.std() / cluster_sizes.mean())
        
        # Overall business impact score
        total_business_value = sum([cv['business_value_score'] for cv in business_values.values()])
        weighted_business_value = sum([
            cv['business_value_score'] * cv['size'] 
            for cv in business_values.values()
        ]) / len(original_data)
        
        return {
            'technical_metrics': {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin
            },
            'business_metrics': {
                'total_business_value': total_business_value,
                'weighted_business_value': weighted_business_value,
                'cluster_balance_score': balance_score,
                'cluster_business_values': business_values
            }
        }

# Apply domain-specific evaluation
domains_data = {
    'ecommerce': (ecommerce_data, ['recency', 'frequency', 'monetary', 'avg_session_duration', 'bounce_rate']),
    'banking': (banking_data, ['account_balance', 'monthly_transactions', 'credit_score', 'products_owned', 'digital_engagement_score']),
    'telecom': (telecom_data, ['monthly_voice_minutes', 'monthly_data_gb', 'monthly_revenue', 'tenure_months', 'service_calls'])
}

evaluation_results = {}

for domain_name, (data, features) in domains_data.items():
    print(f"\n=== {domain_name.upper()} DOMAIN EVALUATION ===")
    
    # Prepare data
    X = StandardScaler().fit_transform(data[features])
    
    # Find optimal K
    selector = OptimalClusterSelector(X, max_clusters=8)
    optimal_k, _ = selector.consensus_optimal_k()
    
    # Apply clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Domain-specific evaluation
    evaluator = DomainSpecificEvaluator(domain_name)
    results = evaluator.evaluate_clustering_quality(X, labels, data)
    
    evaluation_results[domain_name] = {
        'optimal_k': optimal_k,
        'results': results,
        'labels': labels
    }
    
    print(f"Optimal K: {optimal_k}")
    print(f"Silhouette Score: {results['technical_metrics']['silhouette_score']:.3f}")
    print(f"Weighted Business Value: {results['business_metrics']['weighted_business_value']:.3f}")
    print(f"Cluster Balance Score: {results['business_metrics']['cluster_balance_score']:.3f}")
```

---

## Bài Tập 4: Áp Dụng Các Thuật Toán Clustering Nâng Cao

### Mô Tả Bài Toán
Học và implement các thuật toán clustering nâng cao: Mean-Shift, K-Modes (cho categorical data), và K-Prototypes (cho mixed data).

### Yêu Cầu Thực Hiện

#### Phần A: Mean-Shift Clustering

**Nhiệm vụ 4.1**: Implement và optimize Mean-Shift
```python
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.neighbors import NearestNeighbors

class AdvancedMeanShift:
    def __init__(self, data):
        self.data = data
        self.results = {}
    
    def find_optimal_bandwidth(self, quantile_range=(0.1, 0.3), n_samples_range=(100, 500)):
        """
        Tìm bandwidth tối ưu cho Mean-Shift
        """
        bandwidths = []
        quantiles = np.arange(quantile_range[0], quantile_range[1], 0.05)
        n_samples_list = range(n_samples_range[0], n_samples_range[1], 100)
        
        bandwidth_scores = []
        
        for quantile in quantiles:
            for n_samples in n_samples_list:
                try:
                    bandwidth = estimate_bandwidth(
                        self.data, 
                        quantile=quantile, 
                        n_samples=min(n_samples, len(self.data))
                    )
                    
                    if bandwidth > 0:
                        # Test clustering with this bandwidth
                        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                        labels = ms.fit_predict(self.data)
                        
                        n_clusters = len(np.unique(labels))
                        
                        if n_clusters > 1 and n_clusters < len(self.data) * 0.5:
                            silhouette = silhouette_score(self.data, labels)
                            
                            bandwidth_scores.append({
                                'bandwidth': bandwidth,
                                'quantile': quantile,
                                'n_samples': n_samples,
                                'n_clusters': n_clusters,
                                'silhouette_score': silhouette
                            })
                except:
                    continue
        
        if bandwidth_scores:
            # Chọn bandwidth có silhouette score cao nhất
            best_config = max(bandwidth_scores, key=lambda x: x['silhouette_score'])
            return best_config['bandwidth'], bandwidth_scores
        else:
            # Fallback to default
            return estimate_bandwidth(self.data, quantile=0.2), []
    
    def adaptive_mean_shift(self):
        """
        Mean-Shift với adaptive bandwidth cho từng vùng dữ liệu
        """
        # Chia dữ liệu thành các vùng khác nhau
        n_regions = 5
        kmeans_regions = KMeans(n_clusters=n_regions, random_state=42)
        region_labels = kmeans_regions.fit_predict(self.data)
        
        all_labels = np.zeros(len(self.data))
        cluster_counter = 0
        
        for region in range(n_regions):
            region_mask = region_labels == region
            region_data = self.data[region_mask]
            
            if len(region_data) > 10:  # Minimum points for clustering
                # Tìm bandwidth tối ưu cho region này
                bandwidth = estimate_bandwidth(region_data, quantile=0.2)
                
                if bandwidth > 0:
                    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                    region_cluster_labels = ms.fit_predict(region_data)
                    
                    # Adjust labels to be unique across all regions
                    unique_labels = np.unique(region_cluster_labels)
                    for old_label in unique_labels:
                        mask = region_cluster_labels == old_label
                        region_cluster_labels[mask] = cluster_counter
                        cluster_counter += 1
                    
                    all_labels[region_mask] = region_cluster_labels
                else:
                    all_labels[region_mask] = cluster_counter
                    cluster_counter += 1
        
        return all_labels.astype(int)
    
    def compare_mean_shift_variants(self):
        """
        So sánh các variant của Mean-Shift
        """
        variants = {}
        
        # 1. Standard Mean-Shift với optimal bandwidth
        optimal_bandwidth, _ = self.find_optimal_bandwidth()
        ms_standard = MeanShift(bandwidth=optimal_bandwidth, bin_seeding=True)
        labels_standard = ms_standard.fit_predict(self.data)
        
        variants['Standard'] = {
            'labels': labels_standard,
            'n_clusters': len(np.unique(labels_standard)),
            'bandwidth': optimal_bandwidth
        }
        
        # 2. Adaptive Mean-Shift
        labels_adaptive = self.adaptive_mean_shift()
        variants['Adaptive'] = {
            'labels': labels_adaptive,
            'n_clusters': len(np.unique(labels_adaptive))
        }
        
        # 3. Mean-Shift với different seeds
        ms_random_seed = MeanShift(bandwidth=optimal_bandwidth, bin_seeding=False)
        labels_random = ms_random_seed.fit_predict(self.data)
        
        variants['Random Seed'] = {
            'labels': labels_random,
            'n_clusters': len(np.unique(labels_random)),
            'bandwidth': optimal_bandwidth
        }
        
        # Evaluate all variants
        for variant_name, variant_data in variants.items():
            labels = variant_data['labels']
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(self.data, labels)
                calinski_harabasz = calinski_harabasz_score(self.data, labels)
                davies_bouldin = davies_bouldin_score(self.data, labels)
                
                variant_data.update({
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski_harabasz,
                    'davies_bouldin_score': davies_bouldin
                })
        
        return variants

# Test Mean-Shift trên ecommerce data
ecommerce_features = ['recency', 'frequency', 'monetary', 'avg_session_duration', 'bounce_rate']
X_ecommerce_scaled = StandardScaler().fit_transform(ecommerce_data[ecommerce_features])

mean_shift_analyzer = AdvancedMeanShift(X_ecommerce_scaled)
mean_shift_variants = mean_shift_analyzer.compare_mean_shift_variants()

print("Mean-Shift Variants Comparison:")
for variant_name, results in mean_shift_variants.items():
    if 'silhouette_score' in results:
        print(f"{variant_name}: {results['n_clusters']} clusters, "
              f"Silhouette: {results['silhouette_score']:.3f}")
```

#### Phần B: K-Modes cho Categorical Data

**Nhiệm vụ 4.2**: Implement K-Modes clustering
```python
# Cần cài đặt: pip install kmodes
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

class CategoricalClusteringFramework:
    def __init__(self):
        self.results = {}
    
    def create_categorical_customer_data(self, n_customers=2000):
        """
        Tạo dữ liệu khách hàng với categorical features
        """
        np.random.seed(42)
        
        # Define categorical segments
        segments = {
            'Premium': 0.2,
            'Standard': 0.4, 
            'Budget': 0.3,
            'Inactive': 0.1
        }
        
        data = []
        customer_id = 1
        
        for segment, proportion in segments.items():
            size = int(n_customers * proportion)
            
            if segment == 'Premium':
                segment_data = {
                    'customer_id': range(customer_id, customer_id + size),
                    'age_group': np.random.choice(['18-24', '25-34', '35-44'], size, p=[0.2, 0.5, 0.3]),
                    'gender': np.random.choice(['Male', 'Female'], size, p=[0.5, 0.5]),
                    'education': np.random.choice(['High School', 'Graduate'], size, p=[0.6, 0.4]),
                    'income_bracket': np.random.choice(['Medium', 'High'], size, p=[0.7, 0.3]),
                    'city_tier': np.random.choice(['Tier 1', 'Tier 2'], size, p=[0.4, 0.6]),
                    'preferred_channel': np.random.choice(['Online', 'Store'], size, p=[0.6, 0.4]),
                    'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'Cash'], size, p=[0.4, 0.4, 0.2]),
                    'product_category': np.random.choice(['Fashion', 'Electronics', 'Books'], size, p=[0.5, 0.3, 0.2]),
                    'membership_type': np.random.choice(['Standard', 'Silver'], size, p=[0.8, 0.2]),
                    'true_segment': [segment] * size
                }
            elif segment == 'Budget':
                segment_data = {
                    'customer_id': range(customer_id, customer_id + size),
                    'age_group': np.random.choice(['18-24', '45-54', '55+'], size, p=[0.4, 0.3, 0.3]),
                    'gender': np.random.choice(['Male', 'Female'], size, p=[0.4, 0.6]),
                    'education': np.random.choice(['High School', 'Graduate'], size, p=[0.8, 0.2]),
                    'income_bracket': np.random.choice(['Low', 'Medium'], size, p=[0.6, 0.4]),
                    'city_tier': np.random.choice(['Tier 2', 'Tier 3'], size, p=[0.5, 0.5]),
                    'preferred_channel': np.random.choice(['Store', 'Online'], size, p=[0.7, 0.3]),
                    'payment_method': np.random.choice(['Cash', 'Debit Card'], size, p=[0.6, 0.4]),
                    'product_category': np.random.choice(['Books', 'Home', 'Fashion'], size, p=[0.4, 0.4, 0.2]),
                    'membership_type': np.random.choice(['Basic'], size),
                    'true_segment': [segment] * size
                }
            else:  # Inactive
                segment_data = {
                    'customer_id': range(customer_id, customer_id + size),
                    'age_group': np.random.choice(['25-34', '45-54', '55+'], size, p=[0.2, 0.4, 0.4]),
                    'gender': np.random.choice(['Male', 'Female'], size, p=[0.5, 0.5]),
                    'education': np.random.choice(['High School', 'Graduate'], size, p=[0.7, 0.3]),
                    'income_bracket': np.random.choice(['Low', 'Medium'], size, p=[0.8, 0.2]),
                    'city_tier': np.random.choice(['Tier 2', 'Tier 3'], size, p=[0.6, 0.4]),
                    'preferred_channel': np.random.choice(['Store'], size),
                    'payment_method': np.random.choice(['Cash', 'Debit Card'], size, p=[0.8, 0.2]),
                    'product_category': np.random.choice(['Books'], size),
                    'membership_type': np.random.choice(['Basic'], size),
                    'true_segment': [segment] * size
                }
            
            data.append(pd.DataFrame(segment_data))
            customer_id += size
        
        return pd.concat(data, ignore_index=True)
    
    def implement_kmodes_clustering(self, categorical_data, k_range=(2, 8)):
        """
        Implement K-Modes clustering cho categorical data
        """
        # Chuẩn bị data cho K-Modes (chỉ categorical columns)
        categorical_columns = ['age_group', 'gender', 'education', 'income_bracket', 
                             'city_tier', 'preferred_channel', 'payment_method', 
                             'product_category', 'membership_type']
        
        X_categorical = categorical_data[categorical_columns].values
        
        # Custom distance function cho categorical data
        def matching_dissimilarity(X, Y):
            """Hamming distance cho categorical data"""
            return np.sum(X != Y, axis=1) / X.shape[1]
        
        kmodes_results = {}
        
        for k in range(k_range[0], k_range[1] + 1):
            try:
                kmodes = KModes(n_clusters=k, init='Huang', verbose=0, random_state=42)
                labels = kmodes.fit_predict(X_categorical)
                
                # Calculate categorical-specific metrics
                # Purity score
                true_labels = categorical_data['true_segment'].values
                purity = self._calculate_purity(labels, true_labels)
                
                # Categorical silhouette approximation
                # Use Gower distance for mixed data types
                cat_silhouette = self._categorical_silhouette(X_categorical, labels)
                
                kmodes_results[k] = {
                    'labels': labels,
                    'cost': kmodes.cost_,
                    'purity': purity,
                    'categorical_silhouette': cat_silhouette,
                    'n_iterations': kmodes.n_iter_
                }
                
            except Exception as e:
                print(f"Error with k={k}: {e}")
                continue
        
        return kmodes_results
    
    def _calculate_purity(self, cluster_labels, true_labels):
        """
        Calculate purity score for categorical clustering
        """
        total_samples = len(cluster_labels)
        cluster_purity = 0
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_true_labels = true_labels[cluster_mask]
            
            if len(cluster_true_labels) > 0:
                # Find most frequent true label in this cluster
                unique, counts = np.unique(cluster_true_labels, return_counts=True)
                max_count = counts.max()
                cluster_purity += max_count
        
        return cluster_purity / total_samples
    
    def _categorical_silhouette(self, X, labels):
        """
        Approximation của silhouette score cho categorical data
        """
        n_samples = len(X)
        silhouette_scores = []
        
        for i in range(n_samples):
            same_cluster_mask = labels == labels[i]
            same_cluster_indices = np.where(same_cluster_mask)[0]
            same_cluster_indices = same_cluster_indices[same_cluster_indices != i]
            
            if len(same_cluster_indices) == 0:
                silhouette_scores.append(0)
                continue
            
            # Average distance to same cluster
            a = np.mean([
                np.sum(X[i] != X[j]) / len(X[i]) 
                for j in same_cluster_indices
            ])
            
            # Average distance to nearest different cluster
            b_scores = []
            for other_cluster in np.unique(labels):
                if other_cluster != labels[i]:
                    other_cluster_indices = np.where(labels == other_cluster)[0]
                    if len(other_cluster_indices) > 0:
                        avg_dist_to_cluster = np.mean([
                            np.sum(X[i] != X[j]) / len(X[i])
                            for j in other_cluster_indices
                        ])
                        b_scores.append(avg_dist_to_cluster)
            
            if b_scores:
                b = min(b_scores)
                silhouette_score = (b - a) / max(a, b) if max(a, b) > 0 else 0
                silhouette_scores.append(silhouette_score)
            else:
                silhouette_scores.append(0)
        
        return np.mean(silhouette_scores)

# Tạo và test categorical clustering
cat_framework = CategoricalClusteringFramework()
categorical_customer_data = cat_framework.create_categorical_customer_data()

print("Categorical Customer Data:")
print(categorical_customer_data.head())
print("\nData shape:", categorical_customer_data.shape)
print("\nCategorical columns info:")
for col in categorical_customer_data.select_dtypes(include=['object']).columns:
    if col not in ['customer_id', 'true_segment']:
        print(f"{col}: {categorical_customer_data[col].nunique()} unique values")

# Apply K-Modes clustering
kmodes_results = cat_framework.implement_kmodes_clustering(categorical_customer_data)

print("\nK-Modes Results:")
for k, results in kmodes_results.items():
    print(f"K={k}: Cost={results['cost']:.2f}, Purity={results['purity']:.3f}, "
          f"Cat_Silhouette={results['categorical_silhouette']:.3f}")
```

#### Phần C: K-Prototypes cho Mixed Data

**Nhiệm vụ 4.3**: Implement K-Prototypes cho mixed categorical và numerical data
```python
class MixedDataClusteringFramework:
    def __init__(self):
        self.results = {}
    
    def create_mixed_customer_data(self, n_customers=2000):
        """
        Tạo dữ liệu mixed (categorical + numerical)
        """
        # Sử dụng categorical data đã tạo
        cat_data = CategoricalClusteringFramework().create_categorical_customer_data(n_customers)
        
        # Thêm numerical features
        np.random.seed(42)
        
        # Numerical features based on segments
        numerical_features = {}
        for idx, segment in enumerate(cat_data['true_segment']):
            if segment == 'Premium':
                numerical_features.setdefault('annual_spend', []).append(
                    np.random.normal(8000, 1500))
                numerical_features.setdefault('avg_order_value', []).append(
                    np.random.normal(200, 50))
                numerical_features.setdefault('website_visits_per_month', []).append(
                    np.random.normal(25, 5))
                numerical_features.setdefault('customer_service_interactions', []).append(
                    np.random.normal(2, 1))
            elif segment == 'Standard':
                numerical_features.setdefault('annual_spend', []).append(
                    np.random.normal(3000, 800))
                numerical_features.setdefault('avg_order_value', []).append(
                    np.random.normal(100, 30))
                numerical_features.setdefault('website_visits_per_month', []).append(
                    np.random.normal(12, 4))
                numerical_features.setdefault('customer_service_interactions', []).append(
                    np.random.normal(1, 0.5))
            elif segment == 'Budget':
                numerical_features.setdefault('annual_spend', []).append(
                    np.random.normal(800, 300))
                numerical_features.setdefault('avg_order_value', []).append(
                    np.random.normal(50, 15))
                numerical_features.setdefault('website_visits_per_month', []).append(
                    np.random.normal(6, 2))
                numerical_features.setdefault('customer_service_interactions', []).append(
                    np.random.normal(3, 1))
            else:  # Inactive
                numerical_features.setdefault('annual_spend', []).append(
                    np.random.normal(200, 100))
                numerical_features.setdefault('avg_order_value', []).append(
                    np.random.normal(30, 10))
                numerical_features.setdefault('website_visits_per_month', []).append(
                    np.random.normal(2, 1))
                numerical_features.setdefault('customer_service_interactions', []).append(
                    np.random.normal(0.5, 0.3))
        
        # Add numerical features to dataframe
        for feature, values in numerical_features.items():
            cat_data[feature] = np.maximum(values, 0)  # Ensure non-negative
        
        return cat_data
    
    def optimize_kprototypes_gamma(self, mixed_data, k=4, gamma_range=(0.1, 2.0, 0.1)):
        """
        Optimize gamma parameter for K-Prototypes
        """
        categorical_columns = ['age_group', 'gender', 'education', 'income_bracket', 
                             'city_tier', 'preferred_channel', 'payment_method', 
                             'product_category', 'membership_type']
        numerical_columns = ['annual_spend', 'avg_order_value', 'website_visits_per_month', 
                           'customer_service_interactions']
        
        # Prepare data
        X_cat = mixed_data[categorical_columns].values
        X_num = mixed_data[numerical_columns].values
        X_mixed = np.column_stack([X_num, X_cat])
        
        # Mark categorical columns (last len(categorical_columns) columns)
        categorical_indices = list(range(len(numerical_columns), len(numerical_columns) + len(categorical_columns)))
        
        gamma_scores = []
        gammas = np.arange(gamma_range[0], gamma_range[1], gamma_range[2])
        
        for gamma in gammas:
            try:
                kproto = KPrototypes(n_clusters=k, gamma=gamma, verbose=0, random_state=42)
                labels = kproto.fit_predict(X_mixed, categorical=categorical_indices)
                
                # Custom evaluation metric for mixed data
                # Numerical part evaluation
                if len(np.unique(labels)) > 1:
                    num_silhouette = silhouette_score(X_num, labels)
                    
                    # Categorical part evaluation (purity)
                    true_labels = mixed_data['true_segment'].values
                    purity = self._calculate_purity(labels, true_labels)
                    
                    # Combined score
                    combined_score = 0.6 * num_silhouette + 0.4 * purity
                    
                    gamma_scores.append({
                        'gamma': gamma,
                        'combined_score': combined_score,
                        'numerical_silhouette': num_silhouette,
                        'categorical_purity': purity,
                        'cost': kproto.cost_,
                        'n_iterations': kproto.n_iter_
                    })
                    
            except Exception as e:
                print(f"Error with gamma={gamma}: {e}")
                continue
        
        if gamma_scores:
            best_gamma_config = max(gamma_scores, key=lambda x: x['combined_score'])
            return best_gamma_config['gamma'], gamma_scores
        else:
            return 1.0, []
    
    def compare_mixed_data_algorithms(self, mixed_data):
        """
        So sánh các thuật toán cho mixed data
        """
        categorical_columns = ['age_group', 'gender', 'education', 'income_bracket', 
                             'city_tier', 'preferred_channel', 'payment_method', 
                             'product_category', 'membership_type']
        numerical_columns = ['annual_spend', 'avg_order_value', 'website_visits_per_month', 
                           'customer_service_interactions']
        
        # Prepare different data representations
        X_num = StandardScaler().fit_transform(mixed_data[numerical_columns])
        X_cat = mixed_data[categorical_columns].values
        X_mixed = np.column_stack([mixed_data[numerical_columns].values, X_cat])
        categorical_indices = list(range(len(numerical_columns), len(numerical_columns) + len(categorical_columns)))
        
        results = {}
        
        # 1. K-Means on numerical only
        kmeans_num = KMeans(n_clusters=4, random_state=42)
        labels_num_only = kmeans_num.fit_predict(X_num)
        
        results['K-Means (Numerical Only)'] = {
            'labels': labels_num_only,
            'silhouette': silhouette_score(X_num, labels_num_only),
            'purity': self._calculate_purity(labels_num_only, mixed_data['true_segment'].values)
        }
        
        # 2. K-Modes on categorical only
        try:
            kmodes = KModes(n_clusters=4, init='Huang', verbose=0, random_state=42)
            labels_cat_only = kmodes.fit_predict(X_cat)
            
            results['K-Modes (Categorical Only)'] = {
                'labels': labels_cat_only,
                'cost': kmodes.cost_,
                'purity': self._calculate_purity(labels_cat_only, mixed_data['true_segment'].values)
            }
        except Exception as e:
            print(f"K-Modes error: {e}")
        
        # 3. K-Prototypes (optimal gamma)
        try:
            optimal_gamma, _ = self.optimize_kprototypes_gamma(mixed_data)
            kproto = KPrototypes(n_clusters=4, gamma=optimal_gamma, verbose=0, random_state=42)
            labels_mixed = kproto.fit_predict(X_mixed, categorical=categorical_indices)
            
            results['K-Prototypes (Mixed Data)'] = {
                'labels': labels_mixed,
                'cost': kproto.cost_,
                'gamma': optimal_gamma,
                'numerical_silhouette': silhouette_score(X_num, labels_mixed),
                'purity': self._calculate_purity(labels_mixed, mixed_data['true_segment'].values)
            }
        except Exception as e:
            print(f"K-Prototypes error: {e}")
        
        # 4. Ensemble approach: Combine numerical and categorical clustering
        ensemble_labels = self._ensemble_clustering(labels_num_only, labels_cat_only if 'K-Modes (Categorical Only)' in results else labels_num_only)
        
        results['Ensemble Approach'] = {
            'labels': ensemble_labels,
            'numerical_silhouette': silhouette_score(X_num, ensemble_labels),
            'purity': self._calculate_purity(ensemble_labels, mixed_data['true_segment'].values)
        }
        
        return results
    
    def _calculate_purity(self, cluster_labels, true_labels):
        """Calculate purity score"""
        total_samples = len(cluster_labels)
        cluster_purity = 0
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_true_labels = true_labels[cluster_mask]
            
            if len(cluster_true_labels) > 0:
                unique, counts = np.unique(cluster_true_labels, return_counts=True)
                max_count = counts.max()
                cluster_purity += max_count
        
        return cluster_purity / total_samples
    
    def _ensemble_clustering(self, num_labels, cat_labels):
        """
        Combine numerical and categorical clustering results
        """
        # Create consensus labels based on majority voting
        n_samples = len(num_labels)
        ensemble_labels = np.zeros(n_samples)
        
        # Create mapping based on co-occurrence
        for i in range(n_samples):
            num_cluster = num_labels[i]
            cat_cluster = cat_labels[i]
            
            # Simple combination: weight both equally
            ensemble_labels[i] = num_cluster * 10 + cat_cluster
        
        # Remap to consecutive integers
        unique_labels = np.unique(ensemble_labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        
        return np.array([label_mapping[label] for label in ensemble_labels])

# Generate và test mixed data clustering
mixed_framework = MixedDataClusteringFramework()
mixed_customer_data = mixed_framework.create_mixed_customer_data()

print("Mixed Data Sample:")
print(mixed_customer_data.head())
print("\nData types:")
print(mixed_customer_data.dtypes)

# Compare algorithms on mixed data
mixed_results = mixed_framework.compare_mixed_data_algorithms(mixed_customer_data)

print("\nMixed Data Clustering Results:")
for algorithm, results in mixed_results.items():
    print(f"\n{algorithm}:")
    if 'silhouette' in results:
        print(f"  Silhouette Score: {results['silhouette']:.3f}")
    if 'numerical_silhouette' in results:
        print(f"  Numerical Silhouette: {results['numerical_silhouette']:.3f}")
    if 'purity' in results:
        print(f"  Purity Score: {results['purity']:.3f}")
    if 'cost' in results:
        print(f"  Algorithm Cost: {results['cost']:.2f}")
```

---

## Bài Tập 5: Xây Dựng Arsenal Segmentation Techniques cho Marketing Impact

### Mô Tả Bài Toán
Phát triển một bộ công cụ segmentation toàn diện có thể tạo ra impact lớn trong marketing và business strategy.

### Yêu Cầu Thực Hiện

#### Phần A: Advanced Segmentation Techniques

**Nhiệm vụ 5.1**: Implement advanced segmentation framework
```python
import scipy.stats as stats
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

class AdvancedSegmentationArsenal:
    def __init__(self):
        self.segmentation_models = {}
        self.evaluation_results = {}
    
    def cohort_based_segmentation(self, data, cohort_column='days_since_first_purchase'):
        """
        Cohort-based segmentation với time-series analysis
        """
        # Chia thành các cohort theo thời gian
        data_copy = data.copy()
        
        # Define cohort groups
        cohort_boundaries = np.percentile(data_copy[cohort_column], [25, 50, 75])
        
        def assign_cohort(days):
            if days <= cohort_boundaries[0]:
                return 'New (0-25%)'
            elif days <= cohort_boundaries[1]:
                return 'Growing (25-50%)'
            elif days <= cohort_boundaries[2]:
                return 'Mature (50-75%)'
            else:
                return 'Veteran (75%+)'
        
        data_copy['cohort'] = data_copy[cohort_column].apply(assign_cohort)
        
        # Analyze behavior patterns within each cohort
        cohort_analysis = {}
        
        for cohort in data_copy['cohort'].unique():
            cohort_data = data_copy[data_copy['cohort'] == cohort]
            
            # Apply clustering within cohort
            numerical_features = ['recency', 'frequency', 'monetary']
            if all(col in cohort_data.columns for col in numerical_features):
                X_cohort = StandardScaler().fit_transform(cohort_data[numerical_features])
                
                # Optimal K for this cohort
                optimal_k = min(4, len(cohort_data) // 50)  # At least 50 customers per cluster
                if optimal_k >= 2:
                    kmeans_cohort = KMeans(n_clusters=optimal_k, random_state=42)
                    cohort_labels = kmeans_cohort.fit_predict(X_cohort)
                    
                    cohort_analysis[cohort] = {
                        'size': len(cohort_data),
                        'n_clusters': optimal_k,
                        'labels': cohort_labels,
                        'silhouette': silhouette_score(X_cohort, cohort_labels),
                        'avg_metrics': cohort_data[numerical_features].mean().to_dict()
                    }
        
        return cohort_analysis, data_copy
    
    def behavioral_segmentation_with_feature_engineering(self, data):
        """
        Behavioral segmentation với advanced feature engineering
        """
        data_copy = data.copy()
        
        # Create advanced behavioral features
        if all(col in data_copy.columns for col in ['recency', 'frequency', 'monetary']):
            # RFM Composite Scores
            data_copy['recency_score'] = pd.qcut(data_copy['recency'], 5, labels=range(1, 6), duplicates='drop')
            data_copy['frequency_score'] = pd.qcut(data_copy['frequency'], 5, labels=range(1, 6), duplicates='drop')
            data_copy['monetary_score'] = pd.qcut(data_copy['monetary'], 5, labels=range(1, 6), duplicates='drop')
            
            # Convert to numeric
            data_copy['recency_score'] = pd.to_numeric(data_copy['recency_score'])
            data_copy['frequency_score'] = pd.to_numeric(data_copy['frequency_score'])
            data_copy['monetary_score'] = pd.to_numeric(data_copy['monetary_score'])
            
            # Composite behavioral scores
            data_copy['rfm_score'] = (data_copy['recency_score'] + 
                                    data_copy['frequency_score'] + 
                                    data_copy['monetary_score']) / 3
            
            # Customer lifecycle stage
            def lifecycle_stage(row):
                if row['frequency'] < data_copy['frequency'].quantile(0.25):
                    if row['recency'] < data_copy['recency'].quantile(0.5):
                        return 'New'
                    else:
                        return 'At Risk'
                elif row['frequency'] > data_copy['frequency'].quantile(0.75):
                    if row['monetary'] > data_copy['monetary'].quantile(0.75):
                        return 'Champion'
                    else:
                        return 'Loyal'
                else:
                    if row['recency'] < data_copy['recency'].quantile(0.5):
                        return 'Potential Loyalist'
                    else:
                        return 'Hibernating'
            
            data_copy['lifecycle_stage'] = data_copy.apply(lifecycle_stage, axis=1)
            
            # Advanced ratios
            data_copy['avg_order_value'] = data_copy['monetary'] / np.maximum(data_copy['frequency'], 1)
            data_copy['purchase_intensity'] = data_copy['frequency'] / np.maximum(data_copy['recency'], 1)
            data_copy['value_consistency'] = data_copy['monetary'] / (data_copy['recency'] + 1)
        
        # Feature selection based on business importance
        advanced_features = ['rfm_score', 'avg_order_value', 'purchase_intensity', 'value_consistency']
        if all(col in data_copy.columns for col in advanced_features):
            X_advanced = StandardScaler().fit_transform(data_copy[advanced_features])
            
            # Multiple clustering approaches
            clustering_results = {}
            
            # 1. K-Means with advanced features
            kmeans_advanced = KMeans(n_clusters=5, random_state=42)
            labels_advanced = kmeans_advanced.fit_predict(X_advanced)
            
            clustering_results['K-Means Advanced Features'] = {
                'labels': labels_advanced,
                'silhouette': silhouette_score(X_advanced, labels_advanced),
                'features_used': advanced_features
            }
            
            # 2. Gaussian Mixture with advanced features
            gmm_advanced = GaussianMixture(n_components=5, random_state=42)
            labels_gmm = gmm_advanced.fit_predict(X_advanced)
            
            clustering_results['GMM Advanced Features'] = {
                'labels': labels_gmm,
                'silhouette': silhouette_score(X_advanced, labels_gmm),
                'aic': gmm_advanced.aic(X_advanced),
                'bic': gmm_advanced.bic(X_advanced)
            }
            
            return clustering_results, data_copy
        
        return {}, data_copy
    
    def predictive_segmentation_with_validation(self, data, target_column='true_segment'):
        """
        Predictive segmentation với cross-validation
        """
        # Prepare features (exclude target và ID columns)
        feature_columns = [col for col in data.columns 
                          if col not in [target_column, 'customer_id', 'lifecycle_stage']]
        
        # Handle categorical variables
        X = pd.get_dummies(data[feature_columns], drop_first=True)
        y = data[target_column]
        
        # Feature importance analysis
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Cross-validation score
        cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
        
        # Predict clusters and analyze
        predicted_segments = rf.predict(X)
        
        # Confusion matrix analysis
        from sklearn.metrics import classification_report, confusion_matrix
        
        return {
            'feature_importance': feature_importance,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'predicted_segments': predicted_segments,
            'classification_report': classification_report(y, predicted_segments),
            'confusion_matrix': confusion_matrix(y, predicted_segments)
        }
    
    def business_impact_quantification(self, data, labels, revenue_column='monetary'):
        """
        Quantify business impact của segmentation
        """
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = labels
        
        # Calculate business metrics
        cluster_analysis = data_with_clusters.groupby('cluster').agg({
            revenue_column: ['sum', 'mean', 'count'],
            'frequency': ['mean'],
            'recency': ['mean']
        }).round(2)
        
        # Business impact calculations.choice(['25-34', '35-44', '45-54'], size, p=[0.3, 0.5, 0.2]),
                    'gender': np.random.choice(['Male', 'Female'], size, p=[0.6, 0.4]),
                    'education': np.random.choice(['Graduate', 'Post-Graduate'], size, p=[0.4, 0.6]),
                    'income_bracket': np.random.choice(['High', 'Very High'], size, p=[0.3, 0.7]),
                    'city_tier': np.random.choice(['Tier 1'], size),
                    'preferred_channel': np.random.choice(['Online', 'Store', 'Mobile'], size, p=[0.5, 0.3, 0.2]),
                    'payment_method': np.random.choice(['Credit Card', 'Digital Wallet'], size, p=[0.6, 0.4]),
                    'product_category': np.random.choice(['Electronics', 'Fashion', 'Home'], size, p=[0.4, 0.4, 0.2]),
                    'membership_type': np.random.choice(['Premium', 'Gold'], size, p=[0.7, 0.3]),
                    'true_segment': [segment] * size
                }
            elif segment == 'Standard':
                segment_data = {
                    'customer_id': range(customer_id, customer_id + size),
                    'age_group': np.random
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
