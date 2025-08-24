# Chapter 01 - Data Preparation and Cleaning
## **Mục tiêu học tập**
Sau khi hoàn thành bài học này, học viên sẽ có khả năng:
- Đọc và import dữ liệu từ các định dạng file khác nhau (CSV, JSON) vào DataFrame
- Thực hiện các thao tác slicing, aggregation và filtering trên DataFrame
- Kết hợp các DataFrame, xử lý dữ liệu thiếu và làm sạch dữ liệu từ nhiều nguồn khác nhau
## **Lý thuyết**
### Phần 1: Đọc và Import Dữ liệu
1. **Cài đặt và Import thư viện**
```python
import pandas as pd
import numpy as np
import json
```
2. **Đọc dữ liệu từ file CSV**
```python
# Đọc file CSV cơ bản
df_csv = pd.read_csv('data.csv')

# Đọc CSV với các tùy chọn nâng cao
df_csv = pd.read_csv('data.csv', 
                     sep=',',           # Ký tự phân cách
                     encoding='utf-8',  # Encoding
                     index_col=0,       # Cột làm index
                     na_values=['N/A', 'NULL', ''])  # Giá trị coi là NaN
```
3. **Đọc dữ liệu từ file JSON**
```python
# Đọc JSON đơn giản
df_json = pd.read_json('data.json')

# Đọc JSON với cấu trúc phức tạp
with open('complex_data.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)
    df_json = pd.json_normalize(json_data)

# Đọc JSON từ API
df_api = pd.read_json('https://api.example.com/data')
```
4. **Khám phá dữ liệu ban đầu**
```python
# Xem thông tin cơ bản về DataFrame
print(df.info())
print(df.head())
print(df.describe())
print(df.shape)
print(df.columns.tolist())
```
### Phần 2: Slicing, Aggregation và Filtering
1. **Slicing (Cắt lát dữ liệu)**
```python
# Chọn cột
df['column_name']                    # Một cột
df[['col1', 'col2']]                # Nhiều cột

# Chọn hàng
df.loc[0]                           # Hàng theo label
df.iloc[0]                          # Hàng theo vị trí
df.loc[0:5]                         # Slice theo label
df.iloc[0:5]                        # Slice theo vị trí

# Chọn cả hàng và cột
df.loc[0:5, 'col1':'col3']          # Slice hàng và cột
df.iloc[0:5, 0:3]                   # Slice theo vị trí
```
2. **Filtering (Lọc dữ liệu)**
```python
# Filtering cơ bản
df[df['age'] > 25]                  # Điều kiện đơn
df[df['name'].str.contains('John')] # Chứa chuỗi
df[df['date'] > '2023-01-01']       # Điều kiện ngày

# Filtering với nhiều điều kiện
df[(df['age'] > 25) & (df['salary'] > 50000)]  # AND
df[(df['city'] == 'Hanoi') | (df['city'] == 'HCMC')]  # OR

# Filtering với isin()
df[df['category'].isin(['A', 'B', 'C'])]

# Filtering null values
df[df['column'].notna()]            # Không null
df[df['column'].isna()]             # Là null
```
3. **Aggregation (Tổng hợp dữ liệu)**
```python
# Aggregation cơ bản
df['salary'].sum()                  # Tổng
df['age'].mean()                    # Trung bình
df['score'].max()                   # Giá trị lớn nhất
df['score'].min()                   # Giá trị nhỏ nhất
df['category'].count()              # Đếm

# Group by và aggregation
df.groupby('department')['salary'].mean()
df.groupby(['department', 'level']).agg({
    'salary': ['mean', 'sum'],
    'age': 'mean',
    'score': ['min', 'max']
})

# Pivot table
df.pivot_table(values='salary', 
               index='department', 
               columns='level', 
               aggfunc='mean')
```
### **Phần 3: Làm sạch và Kết hợp Dữ liệu**
1. **Xử lý Missing Values (Dữ liệu thiếu)**
```python
# Kiểm tra missing values
df.isnull().sum()                   # Đếm null theo cột
df.isnull().sum().sum()             # Tổng số null
df.info()                           # Thông tin tổng quan

# Xử lý missing values
df.dropna()                         # Xóa hàng có null
df.dropna(axis=1)                   # Xóa cột có null
df.dropna(subset=['important_col']) # Xóa null ở cột cụ thể

# Điền missing values
df.fillna(0)                        # Điền bằng 0
df.fillna(df.mean())               # Điền bằng mean
df.fillna(method='ffill')          # Forward fill
df.fillna(method='bfill')          # Backward fill

# Điền theo nhóm
df['salary'].fillna(df.groupby('department')['salary'].transform('mean'))
```
2. **Xử lý Duplicate Values (Dữ liệu trùng lặp)**
```python
# Kiểm tra và xử lý duplicate
df.duplicated().sum()               # Đếm số hàng trùng
df.drop_duplicates()                # Xóa trùng lặp
df.drop_duplicates(subset=['name', 'email'])  # Xóa trùng theo cột
```
3. **Data Type Conversion (Chuyển đổi kiểu dữ liệu)**
```python
# Chuyển đổi kiểu dữ liệu
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
```
4. **String Cleaning (Làm sạch chuỗi)**
```python
# Làm sạch text data
df['name'] = df['name'].str.strip()           # Xóa khoảng trắng
df['name'] = df['name'].str.upper()           # Chuyển hoa
df['name'] = df['name'].str.lower()           # Chuyển thường
df['phone'] = df['phone'].str.replace('-', '') # Thay thế ký tự
```
5. **Joining DataFrames (Kết hợp DataFrame)**
```python
# Merge DataFrames
df_merged = pd.merge(df1, df2, on='id', how='inner')  # Inner join
df_merged = pd.merge(df1, df2, on='id', how='left')   # Left join
df_merged = pd.merge(df1, df2, on='id', how='right')  # Right join
df_merged = pd.merge(df1, df2, on='id', how='outer')  # Outer join

# Merge với key khác nhau
df_merged = pd.merge(df1, df2, left_on='customer_id', right_on='id')

# Concatenate DataFrames
df_concat = pd.concat([df1, df2], axis=0)     # Theo hàng
df_concat = pd.concat([df1, df2], axis=1)     # Theo cột
```
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

### **Exercise 103: Combining DataFrames and Handling**
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
