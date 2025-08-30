# Chapter 05 Predicting Customer Revenue Using Linear Regression

## **Bài tập Thực hành**
### Bài tập cơ bản

#### **Exercise 5.01: Predicting Sales from Advertising Spend Using Linear Regression**
HINA Inc. is a large FMCG company that is streamlining its marketing budget. This involves taking stock of all its marketing strategies. This, in turn, means re-assessing the effectiveness of its existing spend on various marketing channels. As a marketing analyst, you need to figure out if spending money on TV advertising campaigns results in a direct increase in sales. In other words, you need to find out if the TV advertising spend and the sales figures share a linear relationship. Linear regression seems perfect for the job as it models the relationship as a line. 
You are provided with historical advertising data – weekly sales and spend on each channel – for almost the 4 previous years. Using linear regression, you will make a model that predicts sales based on TV channel spend and study the obtained relationship.

**Code:**

```python
# 1.	Import the relevant libraries for plotting and data manipulation, load advertising.csv
#    dataset into a pandas DataFrame, and print the top five records using the following code:
import numpy as np, pandas as pd import matplotlib.pyplot as plt, seaborn as sns advertising = pd.read_csv("advertising.csv") advertising.head()


# 2.	Visualize the association between TV and Sales through a scatter plot using the following code:
plt.scatter(advertising.TV, advertising.Sales, \             marker="+", color='gray') plt.xlabel("TV") plt.ylabel("Sales") plt.show()


# 3.	Import LinearRegression from sklearn and create an instance of LinearRegression using the following code:
from sklearn.linear_model import LinearRegression lr = LinearRegression()

# 4.	Fit a linear regression model, supplying the TV column as the features and Sales as the outcome, using the fit method of LinearRegression: 
lr.fit(advertising[['TV']], advertising[['Sales']])

# 5.	Using the predict method of the model, create a sales_pred variable containing the predictions from the model:
sales_pred = lr.predict(advertising[['TV']])

# 6.	Plot the predicted sales as a line over the scatter plot of Sales versus TV (using the simple line plot). This should help you assess how well the line fits the data and if it indeed is a good representation of the relationship:
plt.plot(advertising.TV, sales_pred,"k--") plt.scatter(advertising.TV, advertising.Sales, \             marker='+', color='gray') plt.xlabel("TV") plt.ylabel('Sales') plt.show()


```

---

#### **Exercise 5.02: Creating Features for Customer Revenue Prediction**
Azra is a big high-fashion retailer with operations in multiple countries. To optimize their marketing activities, Azra seeks to identify high-value customers – customers that are expected to bring high revenue to the retailer – and have a differential marketing strategy for them. You are a marketing analytics manager at Azra and have a solution to this business problem. The key idea is that a predictive model can be employed to predict the next year's revenue of the customer based on the previous year's purchases. A customer with higher predicted revenue is naturally  a higher-value customer. 
To validate this approach, you plan to build a model to predict the revenue of the customer in 2020 based on the purchases made in 2019. If the model performs well, the approach gets validated. The 2020 purchase data can then be used to predict customer revenue for 2021 and help the company identify high-value customers. 
You have historical transaction data for the years 2019 and 2020 in the file  azra_retail_transactions.csv. 

The first few records are shown in Figure 5.10. For each transaction, you have the following: 
•	Customer identifier (CustomerID)
•	The number of units purchased (Quantity) 
•	The date and time of the purchase (InvoiceDate) 
•	The unit cost (UnitPrice) 
•	Some other information about the item purchased (StockCode, Description) and the customer (Country)
The dataset looks like the following:

![Figure 5.10: Sample records from the file azra_retail_transactions.csv](images/Figure-5.10.jpg)

The goal of this exercise is to manipulate the data and create variables that will allow you to model the customer spend for the year 2020, based on the past activity. The total customer spends for 2020 will therefore be the dependent variable. The independent variables will be features that capture information about the customer's past purchase behavior. Note that this also requires aggregation of the data in order to get one record for each customer. 
More concretely, you will be creating the following variables from the transactions data: 
•	revenue_2019 (total revenue for the year 2019)
•	days_since_first_purchase (the number of days since the first purchase by the customer)
•	days_since_last_purchase (the number of days since the customer's most recent purchase)
•	number_of_purchases (the total number of purchases by the customer in 2019)
•	avg_order_cost (the average value of the orders placed by the customer in 2019)
•	revenue_2020 (the total revenue for the year 2020)
revenue_2020 will be the dependent variable in the model, the rest being the independent variables. The modified dataset with the created features should look like the table in Figure 5.11.

![Figure 5.11: The expected result with the created variables](images/Figure-5.11.jpg)


**Code:**

```python
# 1. Import pandas and load the data from retail_transactions.csv into a DataFrame named df, then print the first five records of the DataFrame. Also, import the datetime module as it will come in handy later:
import pandas as pd import datetime as dt df = pd.read_csv('azra_retail_transactions.csv') df.head()

# 2.	Convert the InvoiceDate column to date format using the to_datetime method from pandas: 
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 3.	Calculate the revenue for each row, by multiplying Quantity by UnitPrice. 
Print the first five records of the dataset to verify the result:
df['revenue'] = df['UnitPrice']*df['Quantity'] df.head()


# 4.	In the dataset, each invoice could be spread over multiple rows, one for each type of product purchased (since the row is for each product, and a customer can buy multiple products in an order). These can be combined such that data for each transaction is on a single row. To do so, perform a groupby operation on InvoiceNo. However, before that, you need to specify how to combine those rows that are grouped together. Use the following code: 
operations = {'revenue':'sum',\
              'InvoiceDate':'first',\               'CustomerID':'first'} df = df.groupby('InvoiceNo').agg(operations)

# 5.	Finally, use the head function to display the result:
df.head()

# 6.	You will be using the year of the transaction to derive features for 2019 and 2020. Create a separate column named year for the year. To do that, use the year attribute of the InvoiceDate column, as follows:
df['year'] = df['InvoiceDate'].dt.year

# 7.	For each transaction, calculate how many days' difference there is between the last day of 2019 and the invoice date using the following code. Use the datetime module we imported earlier:
df['days_since'] = (dt.datetime(year=2019, month=12, day=31) \                     - df['InvoiceDate']).apply(lambda x: x.days)

# 8.	Next, create the features for days since the first and last purchase, along with the number of purchases and total revenue for 2019. Define a set of aggregation functions for each of the variables and apply them using the groupby method. You will calculate the sum of revenue. For the days_since column, you will calculate the maximum and the minimum number of days, as well as the number of unique values (giving you how many separate days this customer made a purchase on). Since these are your predictors, store the result in a variable, X, using the following code:
operations = {'revenue':'sum',\               'days_since':['max','min','nunique']}
X = df[df['year'] == 2019].groupby('CustomerID').agg(operations)

# 9.	Now, use the head function to see the results:
X.head()

# 10. To simplify this, reset the names of the columns to make them easier to refer to later. Use the following code and print the results using the head function:
X.columns = [' '.join(col).strip() for col in X.columns.values]
X.head()

# 11.	Derive one more feature: the average spend per order. Calculate this by dividing revenue sum by days_since nunique (note that this is the average spend per day. For simplicity, assume that a customer only makes one order in a day): 
X['avg_order_cost'] = X['revenue sum']/X['days_since nunique']

# 12.	You need the outcome that you will be predicting, which is just the sum of revenue for 2020. Calculate this with a simple groupby operation and store the values in the y variable, as follows:
y = df[df['year'] == 2020].groupby('CustomerID')['revenue'].sum()

# 13.	Put your predictors and outcomes into a single DataFrame, wrangled_df, and rename the columns to have more intuitive names. Finally, look at the resulting DataFrame, using the head function:
wrangled_df = pd.concat([X,y], axis=1) wrangled_df.columns = ['revenue_2019',\
                       'days_since_first_purchase',\
                       'days_since_last_purchase',\
                       'number_of_purchases',\
                       'avg_order_cost',\                        'revenue_2020'] wrangled_df.head()

# 14.	To drop the customers without values, drop rows where either of the revenue columns are null, as follows:
wrangled_df = wrangled_df[~wrangled_df.revenue_2019.isnull()] wrangled_df = wrangled_df[~wrangled_df.revenue_2020.isnull()]

# 15.	As a final data-cleaning step, it's often a good idea to get rid of outliers. A standard definition is that an outlier is any data point more than three standard deviations above the median. Use this criterion to drop customers that are outliers in terms of 2019 or 2020 revenue:
wrangled_df = wrangled_df[wrangled_df.revenue_2020 \               < ((wrangled_df.revenue_2020.median()) \               + wrangled_df.revenue_2020.std()*3)] wrangled_df = wrangled_df[wrangled_df.revenue_2019 \               < ((wrangled_df.revenue_2019.median()) \
              + wrangled_df.revenue_2019.std()*3)]
16.	It's often a good idea after you've done your data cleaning and feature engineering, to save the new data as a new file, so that, as you're developing your model, you don't need to run the data through the whole feature engineering and cleaning pipeline each time you want to rerun your code. You can do this using the to_csv function. Also, take a look at your final DataFrame using the head function:
wrangled_df.to_csv('wrangled_transactions.csv') wrangled_df.head()

```

---

#### **Exercise 5.03: Examining Relationships between Predictors and the Outcome**
In _Exercise 5.02, Creating Features for Customer Revenue Prediction_, you helped the e-commerce company Azra to transform the raw transaction data into a transformed dataset that has useful dependent and independent features that can be used for model building. In this exercise, you will continue the model-building process by analyzing the relationship between the predictors (independent variables) and the outcome (dependent variable). This will help you identify how the different purchase history-related features affect the future revenue of the customer. This will also help you assess whether the associations in the data make business sense. 
You will use scatter plots to visualize the relationships and use correlations to quantify them. Continue in the same Jupyter notebook you used for the previous exercises. Perform the following steps: 

**Code:**

```python
# 1.	Use pandas to import the data you saved at the end of the last exercise (wrangled_transactions.csv). The CustomerID field is not needed for the analysis. Assign CustomerId as the index for the DataFrame:
df = pd.read_csv('wrangled_transactions.csv', \                  index_col='CustomerID')


# 2.	Using the plot method of the pandas DataFrame, make a scatter plot with days_since_first_purchase on the x axis and revenue_2020 on the y axis to examine the relationship between them:
df.plot.scatter(x="days_since_first_purchase", \                 y="revenue_2020", \                 figsize=[6,6], color='gray') plt.show()

# 3. Using the pairplot function of the seaborn library, create pairwise scatter plots of all the features. Use the following code:
import seaborn as sns sns.set_palette('Greys_r') sns.pairplot(df) plt.show()

# 4. Using the pairplot function and the y_vars parameter, limit the view to the row for your target variable, that is, revenue_2020:
sns.pairplot(df, x_vars=df.columns, y_vars="revenue_2020") plt.show()

# 5. Next, use correlations to quantify the associations between the variables. Use the corr method on the pandas DataFrame, as in the following code:
df.corr()

 
```

---

#### **Activity 5.01: Examining the Relationship between Store Location and Revenue**
The fashion giant Azra also has several physical retail stores where customers can try and buy apparel and fashion accessories. With increased internet penetration and higher adoption of e-commerce among customers, the footfall to the physical stores has been decreasing. To optimize operation costs, the company wishes to understand the factors that affect the revenue of a store. This will help them take better calls regarding setting up future stores and making decisions about the existing ones. 
The data for the activity is in the file location_rev.csv. The file has data on several storefront locations and information about the surrounding area. This information includes the following: 
•	revenue (the revenue of the storefront at each location)
•	location_age (the number of years since the store opened)
•	num_competitors (the number of competitors in a 20-mile radius)
•	median_income (the median income of the residents in the area)
•	num_loyalty_members (the members enrolled in the loyalty program in the area)
•	population_density (the population density of the area)
The goal of the activity is to use the data to uncover some business insights that will help the company decide on the locations for its stores. You will visualize the different associations in the data and then quantify them using correlations. You will interpret the results and answer some questions pertinent to the business: 
•	Which variable has the strongest association with the revenue? 
•	Are all the associations intuitive and make business sense? 
 
_Perform these steps to complete the activity._

1. Load the necessary libraries (pandas, pyplot from matplotlib, and seaborn), read the data from location_rev.csv into a DataFrame, and examine the data by printing the top five rows. 

![Figure 5.24: First five records of the storefront data](images/Figure-5.24.jpg)

2. Using the pandas DataFrame's plot method, create a scatter plot between median_income and the revenue of the store. The output should look like the following:

 ![Figure 5.25: Scatter plot of median_income and revenue](images/Figure-5.25.jpg)  

3. Use seaborn's pairplot function to visualize the data and its relationships. 
You should get the following plot:

 ![Figure 5.26: The seaborn pairplot of the entire dataset](images/Figure-5.26.jpg)  

4. Using the y_vars parameter, plot only the row for associations with the revenue variable. The output should be as follows:

 ![Figure 5.27: Associations with revenue](images/Figure-5.27.jpg)  

5. Finally, calculate correlations using the appropriate method(s) to quantify the relationships between the different variables and location revenue. Analyze the data so that you can answer the following questions:
a)	Which variables have the highest association with revenue? 
b)	Do the associations make business sense?
   
**Code:**

```python


```

---

#### **Exercise 5.04: Building a Linear Model Predicting Customer Spend**
Predicting the future revenue for a customer based on past transactions is a classic problem that linear regression can solve. In this exercise, you will create a linear regression model to predict customer revenue for 2020 for the high-fashion company Azra. In the previous exercises, you performed feature engineering to get the data ready for modeling and analyzed the relationships in the data. Now, using linear regression, you will create a model that describes how future revenue relates to the features based on past transactions.
You will train a linear regression model with revenue_2020 as the dependent variable and the rest of the variables as the independent variables. You will use the train-test approach to make sure you train the model on part of the data and assess it on the unseen test data. You will interpret the coefficients from the trained model and check whether they make business sense. To mathematically assess the performance of the model, you will check the correlation between the predicted values of revenue_2020 and the actual values. A higher correlation would indicate a higher performance of the model.
You will use the file **wrangled_transactions.csv** created in _Exercise 5.02, Creating Features for Transaction Data_. 
 

**Code:**

```python
# 1.	Import pandas and numpy using the following code:
import pandas as pd, numpy as np

# 2.	Create a new DataFrame named df and read the data from  wrangled_transactions.csv with CustomerID as the index:
df = pd.read_csv('wrangled_transactions.csv', \                  index_col='CustomerID')

# 3.	Look at the correlations between the variables again using the corr function:
df.corr()

# 4.	Store the independent and dependent variables in the X and y variables, respectively:
X = df[['revenue_2019',\
        'days_since_last_purchase',\
        'number_of_purchases',\         'avg_order_cost']] y = df['revenue_2020']

# 5.	Use sklearn to perform a train-test split on the data, so that you can assess the model on a dataset it was not trained on:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split\
                                   (X, y, random_state = 100)

# 6.	Import LinearRegression from sklearn using the following code:
from sklearn.linear_model import LinearRegression

# 7.	Create a LinearRegression model, and fit it on the training data:
model = LinearRegression() model.fit(X_train,y_train)

# 8.	Examine the model coefficients by checking the coef_ property. Note that these are in the same order as your X columns: revenue_2019,  days_since_last_purchase, number_of_purchases, and  avg_order_cost:
model.coef_

# 9.	Check the intercept term of the model by checking the intercept_ property:
model.intercept_
This should give a value of 264.86. From steps 8 and 9, you can arrive at the model's full equation:
revenue_2020= 264.86.74 + 5.79*(revenue_2019) + 7.477*(days_since_ last_purchase) + 336.61*(number_of_purchases) – 2.056*(avg_order_ cost)

# 10.	You can now use the fitted model to make predictions about a customer outside of your dataset. Make a DataFrame that holds data for one customer, where revenue for 2019 is 1,000, the number of days since the last purchase is 20, the number of purchases made is 2, and the average order cost is 500. Have the model make a prediction on this one customer's data:
single_customer = pd.DataFrame({'revenue_2019': [1000],\                                 'days_since_last_purchase': [20],\
                                'number_of_purchases': [2],\                                 'avg_order_cost': [500]})
model.predict(single_customer)
The result should be an array with a single value of about 5847.67, indicating the predicted revenue for 2020 for a customer with this data.

# 11.	You can plot the model's predictions on the test set against the true value. First, import matplotlib, and make a scatter plot of the model predictions on X_test against y_test. Limit the x and y axes to a maximum value of 10,000 so that we get a better view of where most of the data points lie. Finally, add a line with slope 1, which will serve as your reference—if all the points lie on this line, it means you have a perfect relationship between your predictions and the true answer:
import matplotlib.pyplot as plt %matplotlib inline
plt.scatter(model.predict(X_test), y_test, color='gray') plt.xlim(0,10000) plt.ylim(0,10000) plt.plot([0, 10000], [0, 10000], 'k-') plt.xlabel('Model Predictions') plt.ylabel('True Value') plt.show()


# 12. To further examine the relationship, you can use correlation. Use the corrcoef method from NumPy to calculate the correlation between the predicted and the actual values of revenue_2020 for the test data:
np.corrcoef(model.predict(X_test), y_test)


```

---

#### **Activity 5.02: Predicting Store Revenue Using Linear Regression**
Revisit the problem you were solving earlier for the high-fashion company Azra. A good understanding of which factors drive the revenue for a storefront will be critical in helping the company decide the locations for upcoming stores in a way that maximizes the overall revenue. 
You will continue working on the dataset you explored in_ Activity 5.01, Examining the Relationship between Store Location and Revenue_. You have, for each store, the revenue along with information about the location of the store. In _Activity 5.01, Examining the Relationship between Store Location and Revenue_, you analyzed the relationship between the store revenue and the location-related features. 
Now, you will build a predictive model using linear regression to predict the revenue of a store using information about its location. You will use a train-test split approach to train the model on part of the data and assess the performance on unseen test data. You will assess the performance of the test data by calculating the correlation between the actual values and the predicted values of revenue. Additionally, you will examine the coefficients of the model to ensure that the model makes business sense. 
Complete the following tasks. Continue in the Jupyter notebook used for _Activity 5.01, Examining the Relationship between Store Location and Revenue_.

1. Import the necessary libraries and the data from **location_rev.csv** and view the first few rows, which should look as follows:

![Figure 5.33: The first five rows of the location revenue data](images/Figure-5.33.jpg)

2.	Create a variable, X, with the predictors (all columns except revenue) in it, and store the outcome (revenue) in a separate variable, y.
3.	Split the data into a training and test set. Use random_state = 100.
4.	Create a linear regression model and fit it on the training data.
5.	Print out the model coefficients.
6.	Print out the model intercept.
7.	Produce a prediction for a location that has three competitors; a median income of 30,000; 1,200 loyalty members; a population density of 2,000; and a location age of 10. The result should be an array with a single value of 27573.21782447, indicating the predicted revenue for a customer with this data.
8.	Plot the model's predictions versus the true values on the test data. Your plot should look as follows:

![Figure 5.34: The model predictions plotted against the true value](images/Figure-5.34.jpg)

9. Calculate the correlation between the model predictions and the true values of the test data.
The result should be around 0.91.

---
