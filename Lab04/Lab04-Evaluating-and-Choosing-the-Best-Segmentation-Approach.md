# Lab 04 Evaluating and Choosing the Best Segmentation Approach

## **Mục tiêu học tập**
Sau khi hoàn thành bài học này, học viên sẽ có thể:
- Hiểu rõ nhu cầu và tầm quan trọng của customer segmentation
- Nắm vững thuật toán K-means và ứng dụng trong phân khúc khách hàng
- Thực hiện phân tích thống kê mô tả và tổng hợp dữ liệu
- Sử dụng các công cụ Python để thực hiện segmentation
- Phân tích và diễn giải kết quả phân khúc khách hàng
- Áp dụng các kỹ thuật nâng cao trong customer segmentation
## **Bài tập Thực hành**
### Bài tập cơ bản
#### **Exercise 4.01: Data Staging and Visualization**
You will be revisiting the business problem you worked on in _Chapter 3, Unsupervised Learning and Customer Segmentation_. You are a data scientist at a leading consulting company and its new client is a popular chain of malls spread across many countries. The mall wishes to re-design its existing offers and marketing communications to improve sales in one of its key markets. An understanding of their customers is critical for this objective, and for that, good customer segmentation is needed. 

The goal of this exercise is to load the data and perform basic clean-up so that you can use it conveniently for further tasks. Also, you will visualize the data to understand better how the customers are distributed on two key attributes –  **Income and Spend_score.** You will be using these fields later to perform clustering. 
 

**Code:**

```python
# 1. In a fresh Jupyter notebook, import pandas, numpy, matplotlib and seaborn libraries and
#    load the mall customer data from the file  Mall_Customers.csv into a DataFrame (mall0)
#    and print the top five records, using the code below. 
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
mall0 = pd.read_csv("Mall_Customers.csv")
mall0.head()

# 2. Rename the columns 'Annual Income (k$)' and  
#    'Spending Score (1-100)' to 'Income' and 'Spend_score' respectively.
#    Print the top five records of the dataset to confirm that the change was completed.
mall0.rename({'Annual Income (k$)':'Income',  'Spending Score (1-100)':'Spend_score'},
             axis=1, inplace=True)
mall0.head()

# 3. Plot a scatterplot of the Income and Spend_score fields using the following code.
#    You will be performing clustering later using these two features as the criteria.
mall0.plot.scatter(x='Income', y='Spend_score', color='gray')
plt.show()

```
#### **Exercise 4.02: Choosing the Number of Clusters Based on Visual Inspection**
The goal of the exercise is to further refine the customer segmentation approach by using visual inspection to decide on the optimal number of clusters. You will try different numbers of clusters (ranging from two to six) and use visual inspection to evaluate the results and choose the right number of clusters. Continue in the Jupyter notebook from _Exercise 4.01, Data Staging and Visualization and perform_ the following steps.

**Code:**

```python
# 1.	Standardize the columns Age, Income and Spend_score, using the StandardScaler from sklearn,
#    after copying the information into new dataset named mall_scaled, using the following code: 
mall_scaled = mall0.copy()
cols_to_scale = ['Age', 'Income', 'Spend_score']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
mall_scaled[cols_to_scale] = scaler.fit_transform(mall_scaled[cols_to_scale])

# 2.	Import the Kmeans module from the sklearn package. Create a list, 
#    'cluster_cols' that stores the names of the fields (Income and  Spend_score)
#    and define the colors and shapes that you will use for each cluster
#    (since you will be visualizing up to seven clusters in all, define seven different shapes), as follows: 
from sklearn.cluster import KMeans
cluster_cols = ['Income', 'Spend_score']
markers = ['x', '*', '.', '|', '_', '1', '2']

#    When plotting the obtained clusters, items in the clusters will be represented by
#    the symbols in the list in order. 'x' will represent the first cluster (Cluster 0).
#    For the final clustering with 7 clusters, all the shapes in the list will be used and
#    Cluster 6 will be represented by the marker '2' (called the 'tickup').

# 3. Then, using a for loop, cluster the data using a different number of clusters,
#    ranging from two to seven, and visualize the resulting plots obtained in a subplot.
#    Use a separate for loop to plot each cluster in each subplot,
#    so we can use different shapes for each cluster. Use the following snippet:
plt.figure(figsize=[12,8])
for n in range(2,8):
    model = KMeans(n_clusters=n, random_state=42)
    mall_scaled['Cluster']= model.fit_predict(mall_scaled[cluster_cols])

    plt.subplot(2,3, n-1)
    for clust in range(n):
        temp = mall_scaled[mall_scaled.Cluster == clust]
        plt.scatter(temp.Income, temp.Spend_score, marker=markers[clust],
                    label="Cluster "+str(clust), color='gray')
        plt.title("N clusters: "+str(n))
        plt.xlabel('Income')
        plt.ylabel('Spend_score')
        plt.legend()
plt.show()

```

#### **Exercise 4.03: Determining the Number of Clusters Using the Elbow Method**
In this exercise, you will use the elbow method to identify the optimal number of clusters. The goal is to improve upon the mall customer segmentation approach by using a principled method to determine the number of clusters so that all involved stakeholders, including business teams, gain more confidence in the soundness of the approach and the resulting clusters. Try the range 2 – 10 for the number of clusters using the age and income data. Continue in the same Jupyter notebook you have been using for the exercises so far. 

**Code:**

```python
# 1.	On the scaled mall customer data (mall_scaled), using the columns 'Income' and 'Spend_score',
#    create three clusters using the KMeans algorithm:
K = 3
model = KMeans(n_clusters=K, random_state=42)
model.fit(mall_scaled[cluster_cols])

# 2.	Once the model is fit, the SSE/inertia is available very conveniently in
#    the 'inertia_' attribute of the model object.
#    Print out the SSE/ inertia for the model with 3 clusters using the following code:
print(model.inertia_)

# 3.	Next, fit multiple KMeans models with the number of clusters
#    ranging from 2 to 10 and store the inertia values for the different models in a list. 
X = mall_scaled[cluster_cols]
inertia_scores = []
for K in range(2,11):
    inertia = KMeans(n_clusters=K, random_state=42).fit(X).inertia_
    inertia_scores.append(inertia)


# 4.	Create the SSE/inertia plot as a line plot with the following code. 
plt.figure(figsize=[7,5])
plt.plot(range(2,11), inertia_scores, color='gray')
plt.title("SSE/Inertia vs. number of clusters")
plt.xlabel("Number of clusters: K")
plt.ylabel('SSE/Inertia')
plt.show()
 
```
#### **Activity 4.01: Optimizing a Luxury Clothing Brand's Marketing Campaign Using Clustering**
You are working at a company that sells luxury clothing. Their sales team has collected data on customer age, income, their annual spend at the business, and the number of days since their last purchase. The company wants to start targeted marketing campaigns but doesn't know how many different types of customers they have. If they understood the number of different segments, it would help design the campaign better by helping define the channels to use, the messaging to employ, and more. 
Your goal is to perform customer segmentation for the company which will help them optimize their campaigns. To make your approach robust and more reliable to business, you need to arrive at the right number of segments by using the visualization approach as well as the elbow method with the sum of squared errors. 

_Execute the following steps to complete the activity:_

1.	Import the libraries required for DataFrame handling and plotting (pandas, numpy, matplotlib). Read in the data from the file  
'Clothing_Customers.csv' into a DataFrame and print the top 5 rows to understand it better.
2.	Standardize all the columns in the data. You will be using all four columns for the segmentation. 
3.	Visualize the data to get a good understanding of it. Since you are dealing with four dimensions, use PCA to reduce to two dimensions before plotting. The resulting plot should be as follows.

![Figure 4.8: Scatterplot of the dimensionality reduced data](images/Figure-4.8.jpg)

4. Visualize clustering with two through seven clusters. You should get the following plot.

 ![Figure 4.9: Resulting clusters for different number of specified clusters](images/Figure-4.9.jpg)  

Choosing clusters using elbow method - create a plot of the sum of squared errors and look for an elbow. Vary the number of clusters from 2 to 11. You should get the following plot.

 ![Figure 4.10: SSE plot for different values of k](images/Figure-4.10.jpg)  

5. Do both the methods agree on the optimal number of clusters? Looking at the results from both, and based on your business understanding, what is the number of clusters you would choose? Explain your decision.

   
**Code:**

```python


```

---

#### **Exercise 4.04: Mean-Shift Clustering on Mall Customers**
In this exercise, you will cluster mall customers using the mean-shift algorithm. You will employ the columns Income and Spend_score as criteria. You will first manually specify the bandwidth parameter. Then, you will estimate the bandwidth parameter using the estimate_bandwidth method and see how it varies with the choice of quantile. Continue in the Jupyter notebook from Exercise 4.03, Determining the Number of Clusters Using the Elbow Method and perform the following steps. 

**Code:**

```python
# 1.	Import MeanShift and estimate_bandwidth from sklearn and create a variable 'bandwidth'
#    with a value of 0.9 – the bandwidth to use (an arbitrary, high value). The code is as follows - 
from sklearn.cluster import MeanShift, estimate_bandwidth
bandwidth = 0.9

# 2.	To perform mean-shift clustering on the standardized data,
#    create an instance of MeanShift, specifying the bandwidth and setting bin_seeding
#    to True (to speed up the algorithm). Fit the model on the data and assign
#    the cluster to the variable 'Cluster'. Use the following code:
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(mall_scaled[cluster_cols])

mall_scaled['Cluster']= ms.predict(X)

# 3.	Visualize the clusters using a scatter plot. 
markers = ['x', '*', '.', '|', '_', '1', '2']
plt.figure(figsize=[8,6])
for clust in range(mall_scaled.Cluster.nunique()):
     temp = mall_scaled[mall_scaled.Cluster == clust]
     plt.scatter(temp.Income, temp.Spend_score, marker=markers[clust], 
                 label="Cluster"+str(clust), color='gray')
plt.xlabel("Income")
plt.ylabel("Spend_score")
plt.legend()
plt.show()

# 4.	Estimate the required bandwidth using the estimate_bandwidth method.
#    Use the estimate_bandwidth function with a quantile value of 0.1 (an arbitrary choice)
#    to estimate the best bandwidth to use. Print the value, fit the model,
#    and note the number of clusters, using the following code: 
bandwidth = estimate_bandwidth(mall_scaled[cluster_cols], quantile=0.1)
print(bandwidth)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True) ms.fit(mall_scaled[cluster_cols])
mall_scaled['Cluster']= ms.predict(mall_scaled[cluster_cols])
mall_scaled.Cluster.nunique()


# 5.	Visualize the obtained clusters using a scatter plot.
plt.figure(figsize=[8,6])
for clust in range(mall_scaled.Cluster.nunique()):
     temp = mall_scaled[mall_scaled.Cluster == clust]
     plt.scatter(temp.Income, temp.Spend_score, marker=markers[clust], 
                 label="Cluster"+str(clust),  color='gray')
plt.xlabel("Income")
plt.ylabel("Spend_score")
plt.legend()
plt.show()


#  6. Estimate the bandwidth again, this time with a quantile value of 0.15.
#    Print out the number of clusters obtained. 
bandwidth = estimate_bandwidth(mall_scaled[cluster_cols], quantile=0.15)
print(bandwidth)

# 7. Use the bandwidth calculated in the previous step to fit and extract the number of clusters.
 ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
 ms.fit(mall_scaled[cluster_cols])
 mall_scaled['Cluster']= ms.predict(mall_scaled[cluster_cols])
 mall_scaled.Cluster.nunique()
 The result should be 5. 

# 8. Visualize the clusters obtained. 
plt.figure(figsize=[8,6])
for clust in range(mall_scaled.Cluster.nunique()):
    temp = mall_scaled[mall_scaled.Cluster == clust]
    plt.scatter(temp.Income, temp.Spend_score, marker=markers[clust], 
                label="Cluster"+str(clust), color='gray')
 plt.xlabel("Income")
 plt.ylabel("Spend_score")
 plt.legend()
 plt.show()
```

---

#### **Exercise 4.05: Clustering Data Using the k-prototypes Method**
For this exercise, you will revisit the customer segmentation problem for Therabank, that you encountered in_ Activity 3.01, Bank Customer Segmentation for Loan Campaign_. The business goal is to get more customers to opt for a personal loan to increase the profitability of the bank's portfolio. Creating customer segments will help the bank identify the types of customers, tune their messaging in the marketing campaigns for the personal loan product. The dataset provided contains data for customers including demographics, some financial information, and how these customers responded to a previous campaign. 

An important feature for business is the education level of the customer and needs to be included in the segmentation. The values in the data are **Primary**, **Secondary**, and **Tertiary**. Since this is a categorical feature, K-means is not a suitable approach. You need to create customer segmentation with this data by applying k-prototype clustering to data that has a mix of categorical (education) and continuous (**income**) variables. 

**Code:**

```python
# 1. Import pandas and read in the data from the file  
#    Bank_Personal_Loan_Modelling-2.csv into a pandas DataFrame named bank0:
import pandas as pd
bank0 = pd.read_csv("Bank_Personal_Loan_Modelling-2.csv")
bank0.head()

# 2.	Standardize the Income column:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
bank_scaled = bank0.copy()
bank_scaled['Income'] = scaler.fit_transform(bank0[['Income']])

# 3.	Import KPrototypes from the kmodes module. Perform k-prototypes clustering
#    using three clusters, specifying the education column (in column index 1) as categorical,
#    and save the result of the clustering as a new column called cluster.
#    Specify a random_state of 42 for consistency.
from kmodes.kprototypes import KPrototypes
cluster_cols = ['Income', 'Education']
kp = KPrototypes(n_clusters=3, random_state=42)
bank_scaled['Cluster'] = kp.fit_predict(bank_scaled[cluster_cols], categorical=[1])

# 4.	To understand the obtained clusters, get the proportions of
#    the different education levels in each cluster using the following code.
res = bank_scaled.groupby('Cluster')['Education'].value_counts(normalize=True)
res.unstack().plot.barh(figsize=[9,6], color=['black','lightgray','dimgray'])
plt.show()


# 3.	Visualize the clusters using a scatter plot. 
markers = ['x', '*', '.', '|', '_', '1', '2']
plt.figure(figsize=[8,6])
for clust in range(mall_scaled.Cluster.nunique()):
     temp = mall_scaled[mall_scaled.Cluster == clust]
     plt.scatter(temp.Income, temp.Spend_score, marker=markers[clust], 
                 label="Cluster"+str(clust), color='gray')
plt.xlabel("Income")
plt.ylabel("Spend_score")
plt.legend()
plt.show()

# 4.	Estimate the required bandwidth using the estimate_bandwidth method.
#    Use the estimate_bandwidth function with a quantile value of 0.1 (an arbitrary choice)
#    to estimate the best bandwidth to use. Print the value, fit the model,
#    and note the number of clusters, using the following code: 
bandwidth = estimate_bandwidth(mall_scaled[cluster_cols], quantile=0.1)
print(bandwidth)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True) ms.fit(mall_scaled[cluster_cols])
mall_scaled['Cluster']= ms.predict(mall_scaled[cluster_cols])
mall_scaled.Cluster.nunique()


# 5.	Visualize the obtained clusters using a scatter plot.
plt.figure(figsize=[8,6])
for clust in range(mall_scaled.Cluster.nunique()):
     temp = mall_scaled[mall_scaled.Cluster == clust]
     plt.scatter(temp.Income, temp.Spend_score, marker=markers[clust], 
                 label="Cluster"+str(clust),  color='gray')
plt.xlabel("Income")
plt.ylabel("Spend_score")
plt.legend()
plt.show()


#  6. Estimate the bandwidth again, this time with a quantile value of 0.15.
#    Print out the number of clusters obtained. 
bandwidth = estimate_bandwidth(mall_scaled[cluster_cols], quantile=0.15)
print(bandwidth)

# 7. Use the bandwidth calculated in the previous step to fit and extract the number of clusters.
 ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
 ms.fit(mall_scaled[cluster_cols])
 mall_scaled['Cluster']= ms.predict(mall_scaled[cluster_cols])
 mall_scaled.Cluster.nunique()
 The result should be 5. 

# 8. Visualize the clusters obtained. 
plt.figure(figsize=[8,6])
for clust in range(mall_scaled.Cluster.nunique()):
    temp = mall_scaled[mall_scaled.Cluster == clust]
    plt.scatter(temp.Income, temp.Spend_score, marker=markers[clust], 
                label="Cluster"+str(clust), color='gray')
 plt.xlabel("Income")
 plt.ylabel("Spend_score")
 plt.legend()
 plt.show()
```

---

#### **Activity 301: Bank Customer Segmentation for Loan Campaign**
Banks often have marketing campaigns for their individual products. Therabank is an established bank that offers personal loans as a product. Most of Therabank's customers have deposits, which is a liability for the bank and not profitable. Loans are profitable for the bank. Therefore, getting more customers to opt for a personal loan makes the equation more profitable. The task at hand is to create customer segments to maximize the effectiveness of their personal loan campaign. 
The bank has data for customers including demographics, some financial information, and how these customers responded to a previous campaign (see Figure 3.21). Some key columns are described here:
•	Experience: The work experience of the customer in years
•	Income: The estimated annual income of the customer (thousands of US dollars)
•	CCAvg: The average spending on credit cards per month (thousands of US dollars)
•	Mortgage: The value of the customer's house mortgage (if any)
•	Age: The age (in years) of the customer

![Figure 3.21: First few records of the Therabank dataset](images/Figure-3.21.jpg)

Your goal is to create customer segments for the marketing campaign. You will also identify which of these segments have the highest propensity to respond to the campaign – information that will greatly help optimize future campaigns.
Note that while the previous campaign's response is available to you, if you use it as a criterion/feature for segmentation, you will not be able to segment other customers for whom the previous campaign was never run, thereby severely limiting the number of customers you can target. You will, therefore, exclude the feature (previous campaign response) for clustering, but you can use it to evaluate how your clusters overall would respond to the campaign. Execute the following steps in a fresh Jupyter notebook to complete the activity:

#### **Exercise 306: Dealing with High-Dimensional Data**
In this exercise, you will use machine learning to discover natural groups in the mall customers. You will perform k-means clustering on the mall customer data that was standardized in the previous exercise. You will use only the Income and  Spend_score columns. Continue using the same Jupyter notebook from the previous exercises. Perform clustering using the scikit-learn package and visualize the clusters: 

1.	Import the necessary libraries for data processing, visualization, and clustering.
2.	Load the data into a pandas DataFrame and display the top five rows. Using the info method, get an understanding of the columns and their types.
3.	Perform standard scaling on the Income and CCAvg columns to create new Income_scaled and CCAvg_scaled columns. You will be using these two variables for customer segmentation. Get a descriptive summary of the processed columns to verify that the scaling has been applied correctly.
4.	Perform k-means clustering, specifying 3 clusters using Income and CCAvg as the features. Specify random_state as 42 (an arbitrary choice) to ensure the consistency of the results. Create a new column, Cluster, containing the predicted cluster from the model.
5.	Visualize the clusters by using different markers for the clusters on a scatter plot between Income and CCAvg. The output should be as follows:

![Figure 3.22: Clusters on a scatter plot](images/Figure-3.22.jpg)

6.	To understand the clusters, print the average values of Income and CCAvg for the three clusters. 
7.	Perform a visual comparison of the clusters using the standardized values for Income and CCAvg. You should get the following plot:

![Figure 3.23: Clusters on a scatter plot](images/Figure-3.23.jpg)

8.	To understand the clusters better using other relevant features, print the average values against the clusters for the Age, Mortgage, Family, CreditCard, Online, and Personal Loan features. Check which cluster has the highest propensity for taking a personal loan.
9.	Based on your understanding of the clusters, assign descriptive labels to the clusters.

#### **Exercise 3.06: Dealing with High-Dimensional Data**
In this exercise, you will perform clustering on the mall customers dataset using the age, income, and spend score. The goal is to find natural clusters in the data based on these three criteria and analyze the customer segments to identify their differentiating characteristics, providing the business with some valuable insight into the nature of its customers. This time though, visualization will not be easy. You will need to use PCA to reduce the data to two dimensions to visualize the clusters: 

**Code:**

```python
# 1. Create a list, cluster_cols, containing the Age, Income, and Spend_score columns,
#    which will be used for clustering. Print the first three rows of the dataset for these columns:
cluster_cols = ['Age', 'Income', 'Spend_score']
data_scaled[cluster_cols].head(3)

# 2.	Perform k-means clustering, specifying 4 clusters using the scaled features. 
#    Specify random_state as 42. Assign the clusters to the Cluster column:
model = KMeans(n_clusters=4, random_state=42)
model.fit(data_scaled[cluster_cols])
data_scaled['Cluster'] = model.predict(data_scaled[cluster_cols])

# 3.	Using PCA on the scaled columns, create two new columns, pc1 and pc2,
#   containing the data for PC1 and PC2 respectively:
from sklearn import decomposition

pca = decomposition.PCA(n_components=2)
pca_res = pca.fit_transform(data_scaled[cluster_cols])

data_scaled['pc1'] = pca_res[:,0]
data_scaled['pc2'] = pca_res[:,1]

# 4.	Visualize the clusters by using different markers and colors for
#    the clusters on a scatter plot between pc1 and pc2 using the following code:
markers = ['x', '*', 'o','|']
for clust in range(4):
     temp = data_scaled[data_scaled.Cluster == clust]
     plt.scatter(temp.pc1, temp.pc2, marker=markers[clust], label="Cluster "+str(clust), color='gray')
 plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# 5. To understand the clusters, print the average values of
#    the original features used for clustering against the four clusters:
data0['Cluster'] = data_scaled.Cluster
data0.groupby('Cluster')[['Age', 'Income', 'Spend_score']].mean()

# 6. Next, visualize this information using bar plots.
#    Check which features are the most differentiated for the clusters using the following code:
data0.groupby('Cluster')[['Age', 'Income', 'Spend_score']].mean().plot.bar(
color=['lightgray', 'darkgray', 'black'])

plt.show()

# 7. Based on your understanding of the clusters, assign descriptive labels to the clusters.
#    One way to describe the clusters is as follows: 
#    Cluster 0: Middle-aged penny pinchers (high income, low spend)
#    Cluster 1: Young high rollers (younger age, high income, high spend)
#    Cluster 2: Young aspirers (low income, high spend)
#    Cluster 3: Old average Joes (average income, average spend)

```
#### **Activity 302: Bank Customer Segmentation with Multiple**
In this activity, you will be revisiting the Therabank problem statement. You'll need to create customer segments to maximize the effectiveness of their personal loan campaign. You will accomplish this by finding the natural customer types in the data and discovering the features that differentiate them. Then, you'll identify the customer segments that have the highest propensity to take a loan. 
In Activity 3.01, Bank Customer Segmentation for Loan Campaign, you employed just two features of the customer. In this activity, you will employ additional features, namely, Age, Experience, and Mortgage. As you are dealing with high-dimensional data, you will use PCA for visualizing the clusters. You will understand the customer segments obtained and provide them with business-friendly labels. As a part of your evaluation and understanding of the segments, you will also check the historical response rates for the obtained segments. 

_Execute the following steps to complete the activity:_
1.	Create a copy of the dataset named bank_scaled, and perform standard scaling of the Income, CCAvg, Age, Experience, and Mortgage columns.
2.	Get a descriptive summary of the processed columns to verify that the scaling has been applied correctly.
3.	Perform k-means clustering, specifying 3 clusters using the scaled features. Specify random_state as 42.
4.	Using PCA on the scaled columns, create two new columns, pc1 and pc2, containing the data for PC1 and PC2 respectively.
5.	Visualize the clusters by using different markers for the clusters on a scatter plot between pc1 and pc2. The plot should appear as in the following figure: 
 
![Figure 3.30: Clusters on a scatter plot](images/Figure-3.30.jpg)

6.	To understand the clusters, print the average values of the features used for clustering against the three clusters. Check which features are the most differentiated for the clusters.
7.	To understand the clusters better using other relevant features, print the average values against the clusters for the Age, Mortgage, Family, CreditCard, Online, and Personal Loan features and check which cluster has the highest propensity for taking a personal loan.
8.	Based on your understanding of the clusters, assign descriptive labels to the clusters.


---
## Bài tập tổng hợp

### Bài Tập 1: Phân khúc khách hàng E-commerce

**Mô tả**: Bạn là Data Analyst của một công ty thương mại điện tử. Công ty cung cấp cho bạn dữ liệu về hành vi mua sắm của khách hàng và yêu cầu phân khúc để tối ưu hóa chiến lược marketing.

**Dữ liệu**: 
- CustomerID: ID khách hàng
- Age: Tuổi
- Annual_Income: Thu nhập hàng năm (USD)
- Spending_Score: Điểm chi tiêu (1-100)
- Purchase_Amount: Tổng giá trị đơn hàng
- Days_Since_Last_Purchase: Số ngày từ lần mua cuối
- Number_of_Purchases: Tổng số lần mua hàng

```python
def create_ecommerce_dataset():
    """
    Tạo dataset mô phỏng dữ liệu e-commerce
    """
    np.random.seed(123)
    n_customers = 2000
    
    # Tạo 4 nhóm khách hàng khác biệt
    data = []
    
    # Nhóm 1: High Value Customers (15%)
    n_hvc = int(n_customers * 0.15)
    hvc_data = {
        'Age': np.random.normal(40, 8, n_hvc),
        'Annual_Income': np.random.normal(90000, 20000, n_hvc),
        'Spending_Score': np.random.normal(85, 10, n_hvc),
        'Purchase_Amount': np.random.normal(5000, 1000, n_hvc),
        'Days_Since_Last_Purchase': np.random.normal(15, 5, n_hvc),
        'Number_of_Purchases': np.random.normal(25, 5, n_hvc)
    }
    
    # Nhóm 2: Regular Customers (40%)
    n_rc = int(n_customers * 0.4)
    rc_data = {
        'Age': np.random.normal(35, 10, n_rc),
        'Annual_Income': np.random.normal(55000, 15000, n_rc),
        'Spending_Score': np.random.normal(55, 15, n_rc),
        'Purchase_Amount': np.random.normal(2500, 800, n_rc),
        'Days_Since_Last_Purchase': np.random.normal(30, 10, n_rc),
        'Number_of_Purchases': np.random.normal(12, 4, n_rc)
    }
    
    # Nhóm 3: Price Sensitive Customers (30%)
    n_psc = int(n_customers * 0.3)
    psc_data = {
        'Age': np.random.normal(45, 12, n_psc),
        'Annual_Income': np.random.normal(35000, 10000, n_psc),
        'Spending_Score': np.random.normal(30, 12, n_psc),
        'Purchase_Amount': np.random.normal(800, 300, n_psc),
        'Days_Since_Last_Purchase': np.random.normal(60, 20, n_psc),
        'Number_of_Purchases': np.random.normal(6, 2, n_psc)
    }
    
    # Nhóm 4: Inactive Customers (15%)
    n_ic = n_customers - n_hvc - n_rc - n_psc
    ic_data = {
        'Age': np.random.normal(50, 15, n_ic),
        'Annual_Income': np.random.normal(45000, 12000, n_ic),
        'Spending_Score': np.random.normal(20, 8, n_ic),
        'Purchase_Amount': np.random.normal(500, 200, n_ic),
        'Days_Since_Last_Purchase': np.random.normal(120, 30, n_ic),
        'Number_of_Purchases': np.random.normal(2, 1, n_ic)
    }
    
    # Kết hợp tất cả dữ liệu
    all_data = {}
    for key in hvc_data.keys():
        all_data[key] = np.concatenate([
            hvc_data[key], rc_data[key], psc_data[key], ic_data[key]
        ])
    
    df = pd.DataFrame(all_data)
    df['CustomerID'] = range(1, len(df) + 1)
    
    # Đảm bảo giá trị hợp lệ
    df['Age'] = np.clip(df['Age'], 18, 80)
    df['Annual_Income'] = np.abs(df['Annual_Income'])
    df['Spending_Score'] = np.clip(df['Spending_Score'], 1, 100)
    df['Purchase_Amount'] = np.abs(df['Purchase_Amount'])
    df['Days_Since_Last_Purchase'] = np.abs(df['Days_Since_Last_Purchase'])
    df['Number_of_Purchases'] = np.abs(df['Number_of_Purchases'])
    
    return df

# Yêu cầu bài tập
def exercise_1_requirements():
    """
    Yêu cầu chi tiết cho bài tập 1
    """
    print("""
    BÀI TẬP 1: PHÂN KHÚC KHÁCH HÀNG E-COMMERCE
    ==========================================
    
    NHIỆM VỤ:
    1. Tải và khám phá dữ liệu
    2. Tiền xử lý dữ liệu (xử lý outliers, standardization)
    3. Thực hiện EDA (Exploratory Data Analysis)
    4. Tìm số cụm tối ưu bằng cả Elbow Method và Silhouette Score
    5. Áp dụng K-means clustering
    6. Phân tích và diễn giải kết quả
    7. Tạo customer personas cho từng segment
    8. Đề xuất chiến lược marketing cho từng nhóm
    
    KẾT QUẢ MONG MUỐN:
    - Report chi tiết về đặc điểm từng segment
    - Visualization hấp dẫn
    - Actionable insights cho business
    - Code có comment đầy đủ
    
    TIÊU CHÍ ĐÁNH GIÁ:
    - Chất lượng phân tích dữ liệu (25%)
    - Tính đúng đắn của clustering (25%)
    - Chất lượng visualization (20%)
    - Business insights (20%)
    - Code quality (10%)
    """)

# Template code cho học viên
exercise_1_template = '''
# BÀI TẬP 1: E-COMMERCE CUSTOMER SEGMENTATION
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Bước 1: Tạo dữ liệu
df_ecommerce = create_ecommerce_dataset()

# Bước 2: EDA - Khám phá dữ liệu
# TODO: Thực hiện EDA chi tiết
print("=== KHÁM PHÁ DỮ LIỆU ===")
# Your code here

# Bước 3: Tiền xử lý
# TODO: Xử lý outliers và chuẩn hóa dữ liệu
features = ['Age', 'Annual_Income', 'Spending_Score', 'Purchase_Amount', 
           'Days_Since_Last_Purchase', 'Number_of_Purchases']
# Your code here

# Bước 4: Tìm K tối ưu
# TODO: Implement Elbow Method và Silhouette Analysis
# Your code here

# Bước 5: Clustering
# TODO: Áp dụng K-means với K tối ưu
# Your code here

# Bước 6: Phân tích kết quả
# TODO: Tạo cluster profiles và visualization
# Your code here

# Bước 7: Business Insights
# TODO: Đưa ra insights và recommendations
# Your code here
'''
```

### Bài Tập 2: Phân khúc khách hàng ngân hàng với RFM Analysis

**Mô tả**: Bạn làm việc cho một ngân hàng và được yêu cầu phân tích hành vi giao dịch của khách hàng để phát triển các sản phẩm tài chính phù hợp.

```python
def create_banking_dataset():
    """
    Tạo dataset mô phỏng dữ liệu ngân hàng
    """
    np.random.seed(456)
    n_customers = 1500
    
    # Tạo transaction data
    customers = []
    
    for customer_id in range(1, n_customers + 1):
        # Random customer characteristics
        age = np.random.randint(25, 70)
        account_balance = np.random.lognormal(8, 1.5)  # Log-normal distribution
        
        # Generate transaction patterns based on customer type
        if np.random.random() < 0.2:  # 20% high-value customers
            n_transactions = np.random.randint(50, 200)
            avg_transaction = np.random.normal(5000, 2000)
            recency = np.random.randint(1, 7)
        elif np.random.random() < 0.5:  # 30% regular customers  
            n_transactions = np.random.randint(20, 80)
            avg_transaction = np.random.normal(1500, 800)
            recency = np.random.randint(3, 15)
        else:  # 50% low-activity customers
            n_transactions = np.random.randint(5, 30)
            avg_transaction = np.random.normal(500, 300)
            recency = np.random.randint(10, 60)
        
        # Ensure positive values
        avg_transaction = max(100, avg_transaction)
        total_transaction_value = n_transactions * avg_transaction
        
        customers.append({
            'CustomerID': customer_id,
            'Age': age,
            'Account_Balance': max(0, account_balance),
            'Total_Transaction_Value': total_transaction_value,
            'Transaction_Frequency': n_transactions,
            'Avg_Transaction_Amount': avg_transaction,
            'Days_Since_Last_Transaction': recency,
            'Credit_Score': np.random.randint(300, 850),
            'Number_of_Products': np.random.randint(1, 8)
        })
    
    return pd.DataFrame(customers)

def create_rfm_features_banking(df):
    """
    Tạo RFM features cho dữ liệu ngân hàng
    """
    df_rfm = df.copy()
    
    # R - Recency (Days since last transaction)
    df_rfm['Recency'] = df_rfm['Days_Since_Last_Transaction']
    
    # F - Frequency (Number of transactions)
    df_rfm['Frequency'] = df_rfm['Transaction_Frequency']
    
    # M - Monetary (Total transaction value)
    df_rfm['Monetary'] = df_rfm['Total_Transaction_Value']
    
    # Additional banking-specific features
    df_rfm['CLV'] = df_rfm['Monetary'] / df_rfm['Age']  # Customer Lifetime Value proxy
    df_rfm['Engagement_Score'] = (df_rfm['Frequency'] * df_rfm['Number_of_Products']) / df_rfm['Recency']
    
    return df_rfm

def exercise_2_requirements():
    """
    Yêu cầu chi tiết cho bài tập 2
    """
    print("""
    BÀI TẬP 2: PHÂN KHÚC KHÁCH HÀNG NGÂN HÀNG VỚI RFM
    ===============================================
    
    BỐI CẢNH:
    Bạn là Data Analyst tại một ngân hàng. Ngân hàng muốn:
    - Xác định khách hàng VIP để tập trung chăm sóc
    - Phát hiện khách hàng có nguy cơ churn
    - Phát triển sản phẩm tài chính phù hợp từng segment
    
    NHIỆM VỤ:
    1. Tạo RFM features từ dữ liệu giao dịch
    2. Thực hiện RFM analysis truyền thống (quintile scoring)
    3. Áp dụng K-means clustering trên RFM data
    4. So sánh kết quả giữa 2 phương pháp
    5. Tạo customer journey mapping
    6. Đề xuất chiến lược retention và cross-selling
    7. Tính toán potential revenue impact
    
    FEATURES CHÍNH:
    - Recency: Số ngày từ giao dịch cuối
    - Frequency: Tần suất giao dịch
    - Monetary: Tổng giá trị giao dịch
    - Account_Balance: Số dư tài khoản
    - Credit_Score: Điểm tín dụng
    - Number_of_Products: Số sản phẩm đang sử dụng
    
    DELIVERABLES:
    - RFM segments với business interpretation
    - Customer personas chi tiết
    - Action plan cho từng segment
    - ROI estimation cho các chiến lược đề xuất
    """)

# Template code
exercise_2_template = '''
# BÀI TẬP 2: BANKING CUSTOMER SEGMENTATION WITH RFM
# ==============================================

# Bước 1: Tạo và khám phá dữ liệu banking
df_banking = create_banking_dataset()

# Bước 2: Tạo RFM features
df_rfm = create_rfm_features_banking(df_banking)

# Bước 3: Traditional RFM Analysis
def traditional_rfm_analysis(df):
    """
    TODO: Implement traditional RFM scoring
    - Chia mỗi RFM thành 5 quintiles (1-5)
    - Tạo RFM score (ví dụ: 555 = top customer)
    - Phân loại customers thành segments
    """
    # Your code here
    pass

# Bước 4: K-means RFM Clustering
def kmeans_rfm_clustering(df):
    """
    TODO: Áp dụng K-means trên RFM features
    - Standardize RFM features
    - Find optimal K
    - Perform clustering
    """
    # Your code here
    pass

# Bước 5: So sánh 2 phương pháp
def compare_methods(df):
    """
    TODO: So sánh traditional RFM vs K-means
    - Visualization comparison
    - Silhouette analysis
    - Business interpretation
    """
    # Your code here
    pass

# Bước 6: Business Strategy
def develop_strategies(df):
    """
    TODO: Phát triển chiến lược cho từng segment
    - Customer personas
    - Product recommendations
    - Marketing strategies
    - Churn prevention plans
    """
    # Your code here
    pass
'''
```

### Bài Tập 3: Multi-dimensional Customer Segmentation cho Retail Chain

**Mô tả**: Bạn là Head of Analytics cho một chuỗi bán lẻ lớn với nhiều cửa hàng trên toàn quốc. Công ty muốn thực hiện segmentation phức tạp kết hợp nhiều chiều dữ liệu.

```python
def create_retail_chain_dataset():
    """
    Tạo dataset phức tạp cho retail chain
    """
    np.random.seed(789)
    n_customers = 3000
    
    # Define store locations và seasonal patterns
    store_locations = ['North', 'South', 'East', 'West', 'Central']
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    product_categories = ['Electronics', 'Clothing', 'Home', 'Sports', 'Beauty']
    
    customers = []
    
    for customer_id in range(1, n_customers + 1):
        # Demographic info
        age = np.random.randint(18, 80)
        gender = np.random.choice(['M', 'F'])
        location = np.random.choice(store_locations)
        
        # Shopping behavior varies by demographics và location
        if age < 30:  # Young customers
            digital_engagement = np.random.normal(0.8, 0.1)
            price_sensitivity = np.random.normal(0.7, 0.1)
            brand_loyalty = np.random.normal(0.4, 0.1)
        elif age < 50:  # Middle-aged customers
            digital_engagement = np.random.normal(0.6, 0.15)
            price_sensitivity = np.random.normal(0.5, 0.15)
            brand_loyalty = np.random.normal(0.6, 0.1)
        else:  # Older customers
            digital_engagement = np.random.normal(0.3, 0.1)
            price_sensitivity = np.random.normal(0.4, 0.1)
            brand_loyalty = np.random.normal(0.8, 0.1)
        
        # Seasonal shopping patterns
        seasonal_spending = {}
        base_spending = np.random.lognormal(6, 1)
        
        for season in seasons:
            if season == 'Winter':  # Holiday season
                seasonal_multiplier = np.random.normal(1.5, 0.3)
            elif season == 'Summer':  # Vacation season
                seasonal_multiplier = np.random.normal(1.2, 0.2)
            else:
                seasonal_multiplier = np.random.normal(1.0, 0.1)
            
            seasonal_spending[season] = max(0, base_spending * seasonal_multiplier)
        
        # Product preferences
        preferred_categories = np.random.choice(product_categories, 
                                              size=np.random.randint(1, 4), 
                                              replace=False)
        
        # Transaction patterns
        avg_basket_size = np.random.normal(3, 1)
        visit_frequency = np.random.normal(15, 5)  # visits per month
        online_vs_offline_ratio = digital_engagement + np.random.normal(0, 0.1)
        
        # Engagement metrics
        email_open_rate = digital_engagement * np.random.normal(1, 0.2)
        social_media_engagement = digital_engagement * np.random.normal(1, 0.3)
        loyalty_program_usage = brand_loyalty * np.random.normal(1, 0.2)
        
        # Ensure values are in valid ranges
        for var in [digital_engagement, price_sensitivity, brand_loyalty, 
                   online_vs_offline_ratio, email_open_rate, 
                   social_media_engagement, loyalty_program_usage]:
            var = max(0, min(1, var))
        
        customers.append({
            'CustomerID': customer_id,
            'Age': age,
            'Gender': gender,
            'Location': location,
            'Annual_Spending': sum(seasonal_spending.values()),
            'Spring_Spending': seasonal_spending['Spring'],
            'Summer_Spending': seasonal_spending['Summer'],
            'Fall_Spending': seasonal_spending['Fall'],
            'Winter_Spending': seasonal_spending['Winter'],
            'Avg_Basket_Size': max(1, avg_basket_size),
            'Visit_Frequency': max(1, visit_frequency),
            'Online_Offline_Ratio': max(0, min(1, online_vs_offline_ratio)),
            'Email_Open_Rate': max(0, min(1, email_open_rate)),
            'Social_Media_Engagement': max(0, min(1, social_media_engagement)),
            'Loyalty_Program_Usage': max(0, min(1, loyalty_program_usage)),
            'Price_Sensitivity': max(0, min(1, price_sensitivity)),
            'Brand_Loyalty': max(0, min(1, brand_loyalty)),
            'Preferred_Categories': ','.join(preferred_categories),
            'Digital_Engagement': max(0, min(1, digital_engagement))
        })
    
    return pd.DataFrame(customers)

def exercise_3_requirements():
    """
    Yêu cầu chi tiết cho bài tập 3
    """
    print("""
    BÀI TẬP 3: MULTI-DIMENSIONAL CUSTOMER SEGMENTATION
    ===============================================
    
    BỐI CẢNH:
    Chuỗi bán lẻ QuickMart có 50 cửa hàng trên toàn quốc với:
    - 3 triệu khách hàng active
    - 5 categories sản phẩm chính  
    - Cả online và offline channels
    - Loyalty program với 60% participation rate
    
    THÁCH THỨC KINH DOANH:
    - Competition từ e-commerce giants
    - Changing consumer behavior post-COVID
    - Need for personalized experience
    - Inventory optimization across locations
    
    NHIỆM VỤ NÂNG CAO:
    1. Feature Engineering:
       - Tạo seasonal indices
       - Channel preference scores
       - Category affinity metrics
       - Geographic clustering
    
    2. Advanced Clustering:
       - Thử multiple algorithms (K-means, Hierarchical, DBSCAN)
       - Ensemble clustering methods
       - Stability analysis across time periods
    
    3. Multi-level Segmentation:
       - Macro segments (behavioral)
       - Micro segments (demographic + geographic)
       - Temporal segments (seasonal patterns)
    
    4. Validation và Business Impact:
       - A/B test design for segment validation
       - Revenue impact modeling
       - Customer satisfaction prediction
    
    5. Advanced Analytics:
       - Customer journey mapping
       - Next-best-action recommendations
       - Churn prediction by segment
       - LTV calculation
    
    EXPECTED OUTPUTS:
    - Comprehensive segmentation strategy
    - Interactive dashboard design
    - Implementation roadmap
    - Success metrics framework
    """)

# Template code cho bài tập phức tạp
exercise_3_template = '''
# BÀI TẬP 3: MULTI-DIMENSIONAL CUSTOMER SEGMENTATION
# ===============================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df_retail = create_retail_chain_dataset()

# PHASE 1: ADVANCED FEATURE ENGINEERING
# =====================================

def create_advanced_features(df):
    """
    TODO: Tạo features nâng cao
    
    1. Seasonal Features:
       - Seasonal spending variation
       - Peak season indicator
       - Spending consistency index
    
    2. Behavioral Features:
       - Channel preference score
       - Engagement composite score
       - Purchase pattern regularity
    
    3. Value Features:
       - Customer lifetime value estimate
       - Profitability score
       - Growth potential index
    """
    # Your advanced feature engineering here
    pass

def geographic_analysis(df):
    """
    TODO: Phân tích địa lý
    - Regional spending patterns
    - Location-based clustering
    - Market penetration analysis
    """
    # Your code here
    pass

# PHASE 2: ENSEMBLE CLUSTERING APPROACH
# ==================================

def ensemble_clustering(df, features):
    """
    TODO: Implement ensemble clustering
    - Multiple algorithms comparison
    - Consensus clustering
    - Stability analysis
    """
    # Your code here
    pass

def hierarchical_clustering_analysis(df, features):
    """
    TODO: Hierarchical clustering với dendrogram
    - Determine optimal number of clusters
    - Analyze cluster hierarchy
    - Compare with K-means results
    """
    # Your code here
    pass

# PHASE 3: MULTI-LEVEL SEGMENTATION
# ==============================

def macro_segmentation(df):
    """
    TODO: Macro-level behavioral segments
    - High-level customer archetypes
    - Cross-segment analysis
    - Segment migration patterns
    """
    # Your code here
    pass

def micro_segmentation(df):
    """
    TODO: Micro-level segments within macro segments
    - Demographic refinement
    - Geographic sub-segments
    - Product preference clusters
    """
    # Your code here
    pass

# PHASE 4: BUSINESS IMPACT ANALYSIS
# ==============================

def calculate_segment_value(df):
    """
    TODO: Tính toán giá trị của từng segment
    - Revenue contribution
    - Profitability analysis
    - Growth potential
    """
    # Your code here
    pass

def develop_action_plans(df):
    """
    TODO: Phát triển action plans
    - Personalization strategies
    - Channel optimization
    - Product recommendations
    - Retention programs
    """
    # Your code here
    pass

# PHASE 5: ADVANCED VISUALIZATION
# ============================

def create_interactive_dashboard():
    """
    TODO: Tạo interactive dashboard concept
    - Segment overview
    - Drill-down capabilities  
    - Real-time updates
    """
    # Your dashboard design here
    pass

def segment_journey_mapping(df):
    """
    TODO: Customer journey mapping by segment
    - Touchpoint analysis
    - Conversion funnels
    - Experience optimization opportunities
    """
    # Your code here
    pass

# RUN COMPLETE ANALYSIS
# ===================
if __name__ == "__main__":
    print("Starting Multi-Dimensional Customer Segmentation Analysis...")
    
    # Execute all phases
    # Phase 1: Feature Engineering
    # Phase 2: Ensemble Clustering  
    # Phase 3: Multi-level Segmentation
    # Phase 4: Business Impact Analysis
    # Phase 5: Visualization and Reporting
'''
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
