# Lab01 - Data Preparation and Cleaning

## **Mục tiêu học tập**
Sau khi hoàn thành bài học này, học viên sẽ có khả năng:
- Đọc và import dữ liệu từ các định dạng file khác nhau (CSV, JSON) vào DataFrame
- Thực hiện các thao tác slicing, aggregation và filtering trên DataFrame
- Kết hợp các DataFrame, xử lý dữ liệu thiếu và làm sạch dữ liệu từ nhiều nguồn khác nhau
## **Bài tập Thực hành**
## Bài tập cơ bản
### **Exercise 101: Loading Data Stored in a JSON File**

```python
 import pandas as pd #  import the pandas library


 # Create a new DataFrame called user_info and read the user_info.json
 user_info = pd.read_json("user_info.json")

 # checking the first five values in the DataFrame
  user_info.head()

# Are there any missing values in any of the columns?
# What are the data types of all the columns?

 user_info.info()

# How many rows and columns are present in the dataset?

 user_info.shape
```
### **Exercise 102: Loading Data from Multiple Sources**
Loading Data from data.csv

```python
 import pandas as pd

 # Create a new DataFrame called campaign_data.
 campaign_data = pd.read_csv("data.csv")

 # Examine the first five rows of the DataFrame
  campaign_data.head()

```

To use the header parameter to make sure that the entries in the first row are read as column names. The header = 1 parameter reads the first row as the header:

```python
campaign_data = pd.read_csv("data.csv", header = 1)

 campaign_data.head()
```

Examine the last five rows using the tail() function

```python
campaign_data.tail()
Loading Data from sales.csv
```

Create a new DataFrame called sales.

 ```python
sales = pd.read_csv("sales.csv")
```

Look at the first five rows of the sales DataFrame

```python
sales.head()
```

To check for null values and examine the data types of the columns

```python
sales.info()
```

#### **Exercise 103: Combining DataFrames and Handling**
You will combine the DataFrame containing the time spent by the users with the other DataFrame containing the cost of acquisition of the user. You will merge both these DataFrames to get an idea of user behavior. 
_Perform the following steps to achieve the aim of this exercise:_

Code:

```python
# 1.	Import the pandas modules that you will be using in this exercise:
import pandas as pd

# 2.	Load the CSV files into the DataFrames df1 and df2:
df1 =pd.read_csv("timeSpent.csv")
df2 =pd.read_csv("cost.csv")

# 3.	Examine the first few rows of the first DataFrame using the head() function:
df1.head()

# 4. Next, look at the first few rows of the second dataset:
df2.head()

# 5. Do a left join of df1 with df2 and store the output in a DataFrame, df. Use a left join as we are only interested in users who are spending time on the website. Specify the joining key as "users":
df = df1.merge(df2, on="users", how="left")
df.head()

# 6. You'll observe some missing values (NaN) in the preceding output. These types of scenarios are very common as you may fail to capture some details pertaining to the users. This can be attributed to the fact that some users visited the website organically and hence, the cost of acquisition is zero. 
# These missing values can be replaced with the value 0. Use the following code:
df=df.fillna(0)
df.head()

# Now, the DataFrame has no missing values and you can compute the average cost of acquisition along with the average time spent. To compute the average value, you will be using the built-in function describe, which gives the statistics of the numerical columns in the DataFrame. Run the following command:
df.describe()

# Based on the traffic you want to attract for the forthcoming holiday season, you can now compute the marketing budget using the following formula:
Marketing Budget = Number of users * Cost of Acquisition 

```
### **Exercise 104: Applying Data Transformations**
_You will use the user_info.json file:_
-	What is the average age of the users?
-	Which is the favorite fruit of the users?
-	Do you have more female customers?
-	How many of the users are active?
-	
**Code:**

```python
#1.	Import the pandas module using the following command:
import pandas as pd

#2.	Read the user_info.json file into a pandas DataFrame, user_info, using the following command:
user_info = pd.read_json('user_info.json')

#3.	Now, examine whether your data is properly loaded by checking the first few values in the DataFrame. Do this using the head() command:
user_info.head()

#4. Now, look at the attributes and the data inside them using the following command:
user_info.info()

5. Now, let's start answering the questions:
# What is the average age of the users? To find the average age, use the following code:
user_info['age'].mean()

# Which is the favorite fruit among the users? To answer this question, you can use the groupby function on the favoriteFruit column and get a count of users with the following code:
user_info.groupby('favoriteFruit')['_id'].count()

# Do you have more female customers? To answer this question, you need to count the number of male and female users. You can find this count with the help of the groupby function. Use the following code:
user_info.groupby('gender')['_id'].count()

# How many of the users are active? Similar to the preceding questions, you can use the groupby function on the isActive column to find out the answer.
# Use the following code:
user_info.groupby('isActive')['_id'].count()

```
### **Activity 101: Addressing Data Spilling**
You will start by loading **sales.csv**, which contains some historical sales data about different customer purchases in stores in the past few years. As you may recall, the data loaded in the DataFrame was not correct as the values of some columns were getting populated wrongly in other columns. The goal of this activity is to clean the DataFrame and make it into a usable form. 
You need to read the files into pandas DataFrames and prepare the output so that it can be used for further analysis. _Follow the steps given here:_
1.	Open a new Jupyter notebook and import the **pandas** module.
2.	Load the data from **sales.csv** into a separate DataFrame, named **sales**, and look at the first few rows of the generated DataFrame.
3.	Analyze the data type of the fields.
4.	Look at the first column. If the value in the column matches the expected values, move on to the next column or otherwise fix it with the correct value.
5.	Once you have fixed the first column, examine the other columns one by one and try to ascertain whether the values are right. 

## Bài tập tổng hợp
### **Bài tập 1: Làm sạch dữ liệu khách hàng**
Cho file customers.csv với cấu trúc sau:

- *customer_id, name, email, phone, age, city, registration_date*

Yêu cầu:

Đọc dữ liệu từ file CSV
- Kiểm tra và xử lý missing values
- Chuẩn hóa định dạng email và phone
- Xóa các bản ghi trùng lặp
- Chuyển đổi registration_date sang datetime

```python
# Template giải
import pandas as pd

# 1. Đọc dữ liệu
df = pd.read_csv('customers.csv')

# 2. Kiểm tra missing values
print(df.isnull().sum())

# 3. Xử lý missing values
# Điền age bằng median
df['age'].fillna(df['age'].median(), inplace=True)
# Xóa hàng thiếu email (quan trọng)
df.dropna(subset=['email'], inplace=True)

# 4. Chuẩn hóa dữ liệu
df['email'] = df['email'].str.lower().str.strip()
df['phone'] = df['phone'].str.replace(r'[^0-9]', '', regex=True)
df['name'] = df['name'].str.title().str.strip()

# 5. Xóa trùng lặp
df.drop_duplicates(subset=['email'], inplace=True)

# 6. Chuyển đổi kiểu dữ liệu
df['registration_date'] = pd.to_datetime(df['registration_date'])

print("Dữ liệu sau khi làm sạch:")
print(df.info())
```
### **Bài tập 2: Phân tích dữ liệu bán hàng**
Cho 2 file:

- *sales.csv: order_id, customer_id, product_id, quantity, order_date*
- *products.json: product_id, product_name, category, price*

Yêu cầu:

- Đọc dữ liệu từ cả 2 file
- Kết hợp dữ liệu từ 2 nguồn
- Tính tổng doanh thu theo category
- Tìm top 5 sản phẩm bán chạy nhất
- Phân tích xu hướng bán hàng theo tháng

```python
# Template giải
import pandas as pd
import json

# 1. Đọc dữ liệu
sales_df = pd.read_csv('sales.csv')
with open('products.json', 'r') as f:
    products_data = json.load(f)
products_df = pd.DataFrame(products_data)

# 2. Kết hợp dữ liệu
merged_df = pd.merge(sales_df, products_df, on='product_id', how='inner')

# Tính revenue
merged_df['revenue'] = merged_df['quantity'] * merged_df['price']

# 3. Tổng doanh thu theo category
revenue_by_category = merged_df.groupby('category')['revenue'].sum().sort_values(ascending=False)
print("Doanh thu theo category:")
print(revenue_by_category)

# 4. Top 5 sản phẩm bán chạy
top_products = merged_df.groupby('product_name')['quantity'].sum().sort_values(ascending=False).head(5)
print("\nTop 5 sản phẩm bán chạy:")
print(top_products)

# 5. Xu hướng theo tháng
merged_df['order_date'] = pd.to_datetime(merged_df['order_date'])
merged_df['month'] = merged_df['order_date'].dt.to_period('M')
monthly_trend = merged_df.groupby('month')['revenue'].sum()
print("\nXu hướng doanh thu theo tháng:")
print(monthly_trend)
```
#### **Bài tập 3: Xử lý dữ liệu từ nhiều nguồn**
Scenario: Bạn có dữ liệu nhân viên từ 3 nguồn:

- *employees_hr.csv: employee_id, name, department, hire_date*
- *salaries.json: employee_id, base_salary, bonus*
- *performance.csv: employee_id, performance_score, last_review_date*

Yêu cầu:

- Đọc và làm sạch dữ liệu từ cả 3 nguồn
- Kết hợp tất cả dữ liệu thành một DataFrame duy nhất
- Xử lý missing values một cách phù hợp
- Tính tổng lương (base + bonus) cho mỗi nhân viên
- Phân tích mức lương trung bình theo department
- Tìm nhân viên có performance cao nhất trong mỗi department
  ướng dẫn:
```python
# Template giải (học viên tự hoàn thành)
# Gợi ý:
# - Sử dụng pd.merge() để kết hợp multiple DataFrames
# - Chú ý xử lý missing values phù hợp với từng trường
# - Sử dụng groupby() cho các phân tích theo nhóm
```
## Tổng kết và Best Practices
1. Quy trình làm sạch dữ liệu chuẩn:

- Khám phá dữ liệu: Sử dụng .info(), .describe(), .head()
- Kiểm tra chất lượng: Missing values, duplicates, outliers
- Chuẩn hóa: Data types, string formatting, date parsing
- Xử lý missing data: Drop, fill, interpolate
- Kết hợp dữ liệu: Merge, join, concatenate
- Validation: Kiểm tra logic và consistency

2. Lưu ý quan trọng:

- Luôn backup dữ liệu gốc trước khi làm sạch
- Ghi chép lại các bước xử lý để có thể reproduce
- Ghi chép lại các bước xử lý để có thể reproduce
- Kiểm tra kết quả sau mỗi bước xử lý
- Hiểu domain knowledge để xử lý missing values đúng cách
- Sử dụng .copy() khi cần tạo bản sao DataFrame
