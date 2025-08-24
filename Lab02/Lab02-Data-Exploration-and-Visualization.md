# Lab 02 - **Data Exploration and Visualization**

## **Mục tiêu học tập**

_Sau khi hoàn thành bài học này, học viên sẽ có thể:_

- Khám phá, phân tích và biến đổi dữ liệu một cách hiệu quả
- Sử dụng các hàm thống kê mô tả để hiểu dữ liệu
- Xây dựng pivot table và thực hiện phân tích so sánh
- Tạo các biểu đồ trực quan hóa bằng Matplotlib và Seaborn
- Áp dụng các kỹ thuật nâng cao trong phân tích dữ liệu
## **Bài tập Thực hành**
### Bài tập cơ bản
#### **Exercise 2.01: Exploring the Attributes in Sales Data**
You and your team are creating a marketing campaign for a client. All they've handed you is a file called sales.csv, which as they explained, contains the company's historical sales records. Apart from that, you know nothing about this dataset. 
Using this data, you'll need to derive insights that will be used to create a comprehensive marketing campaign. Not all insights may be useful to the business, but since you will be presenting your findings to various teams first, an insight that's useful for one team may not matter much for the other team. So, your approach would be to gather as many actionable insights as possible and present those to the stakeholders. 


```python
# 1.	Open a new Jupyter Notebook to implement this exercise. Save the file as Exercise2-01.ipnyb.
#     In a new Jupyter Notebook cell, import the pandas library as follows
import pandas as pd

2.	Create a new pandas DataFrame named sales and read the sales.csv file into it.
#     Examine if your data is properly loaded by checking the first few values in
#     the DataFrame by using the head() command:
sales = pd.read_csv('sales.csv') sales.head()

#     Examine the columns of the DataFrame using the following code:
sales.columns

# 4. Use the info function to print the datatypes of columns of sales DataFrame using the following code:
sales.info()

# 5.	To check the time frame of the data, use the unique function on the Year column:
sales['Year'].unique()

# 6.	Use the unique function again to find out the types of products that the company is selling:
sales['Product line'].unique()

# 7.	Now, check the Product type column:
sales['Product type'].unique()

# 8.	Check the Product column to find the unique categories present in it:
sales['Product'].unique()

# 9.	Now, check the Order method type column to find out the ways through which the customer can place an order:
sales['Order method type'].unique()

# 10. Use the same function again on the Retailer country column to find out
#     the countries where the client has a presence:
sales['Retailer country'].unique()

# 11. Now that you have analyzed the categorical values, get a quick summary of
#     the numerical fields using the describe function:
sales.describe()

# 12. Analyze the spread of the categorical columns in the data using the  value_counts() function.
#     This would shed light on how the data is distributed. Start with the Year column:
sales['Year'].value_counts()

# 13. Use the same function on the Product line column:
sales['Product line'].value_counts()

# 14. Now, check for the Product type column:
sales['Product type'].value_counts()

# 15. Now, find out the most popular order method using the following code:
sales['Order method type'].value_counts()

# 16. Finally, check for the Retailer country column along with their respective counts:
sales['Retailer country'].value_counts()

# 17. Get insights into country-wide statistics now. Group attributes such as Revenue,
#     Planned revenue, Product cost, Quantity, and Gross profit by their countries,
#     and sum their corresponding values. Use the following code:
sales.groupby('Retailer country')[['Revenue','Planned revenue', 'Product cost', 'Quantity', 'Gross profit']].sum()

# 18. Now find out the country whose product performance was affected the worst when sales dipped.
#     Use the following code to group data by Retailer country:
sales.dropna().groupby('Retailer country')[['Revenue', 'Planned revenue', Product cost', 'Quantity',
                        'Unit cost', 'Unit price', 'Gross profit', 'Unit sale price']].min()

# 19. Similarly, generate statistics for other categorical variables, such as Year,
#     Product line, Product type, and Product. Use the following code for the Year variable:
sales.groupby('Year')[['Revenue', 'Planned revenue', 'Product cost', Quantity', 'Unit cost',
                       'Unit price', 'Gross profit', 'Unit sale price']].sum()

# 20. Use the following code for the Product line variable:
sales.groupby('Product line')[['Revenue', 'Planned revenue', 'Product cost', 'Quantity', 'Unit cost', 
                               'Unit price', 'Gross profit', 'Unit sale price']].sum()

# 21. Now, find out which order method contributes to the maximum revenue:
sales.groupby('Order method type')[['Revenue', 'Planned revenue', 'Product cost', 'Quantity',
                                    'Gross profit']].sum()

```
#### **Exercise 2.02: Calculating Conversion Ratios for Website Ads.**
You are the owner of a website that randomly shows advertisements A or B to users each time a page is loaded. The performance of these advertisements is captured in a simple file called conversion_rates.csv. The file contains two columns: converted and group. If an advertisement succeeds in getting a user to click on it, the converted field gets the value 1, otherwise, it gets 0 by default; the group field denotes which ad was clicked – A or B. 
As you can see, comparing the performance of these two ads is not that easy when the data is stored in this format. Use the skills you've learned so far, store this data in a data frame and modify it to show, in one table, information about:
1.	The number of views ads in each group got.
2.	The number of ads converted in each group.
3.	The conversion ratio for each group.

Code:

```python
# 1.	Import the pandas library using the import command, as follows:
import pandas as pd

# 2.	Create a new pandas DataFrame named data and read the  conversion_rates.csv file into it.
#     Examine if your data is properly loaded by checking the first few values in the DataFrame
#     by using the head() command:
data = pd.read_csv('conversion_rates.csv')
data.head()

# 3. Group the data by the group column and count the number of conversions. 
#    Store the result in a DataFrame named converted_df:
converted_df = data.groupby('group').sum() converted_df

# 4. We would like to find out how many people have viewed the advertisement.
#    For that use the groupby function to group the data and the count() function
#    to count the number of times each advertisement was displayed.
#    Store the result in a DataFrame viewed_df. Also, make sure you change the column name from
#     converted to viewed:
viewed_df = data.groupby('group').count().rename({'converted':'viewed'},axis = 'columns')
viewed_df

# 5. Combine the converted_df and viewed_df datasets in a new DataFrame, named stats using the following commands:
stats = converted_df.merge(viewed_df, on = 'group') stats

# 6. Create a new column called conversion_ratio that displays the ratio of converted ads to
#     the number of views the ads received:
stats['conversion_ratio'] = stats['converted'] / stats['viewed']
stats

# 7. Create a DataFrame df where group A's conversion ratio is accessed as df['A'] ['conversion_ratio'].
#    Use the stack function for this operation:
df = stats.stack()
df

# 8.	Check the conversion ratio of group A using the following code:
df['A']['conversion_ratio']

# 9.	To bring back the data to its original form we can reverse the rows with the columns in
#    the stats DataFrame with the unstack() function twice:
stats.unstack().unstack()

```

To use the header parameter to make sure that the entries in the first row are read as column names. 
The header = 1 parameter reads the first row as the header:

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

#### **Exercise 2.03: Visualizing Data With pandas**
In this exercise, you'll be revisiting the sales.csv file you worked on in _Exercise 2.01, Exploring the Attributes in Sales Data_. This time, you'll need to visualize the sales data to answer the following two questions:
1.	Which mode of order generates the most revenue?
2.	How have the following parameters varied over four years: Revenue, Planned revenue, and Gross profit?
You will make use of bar plots and box plots to explore the distribution of the Revenue column.
 
_Perform the following steps to achieve the aim of this exercise:_

Code:

```python
# 1.	Import the pandas library using the import command as follows:
import pandas as pd

# 2.	Create a new panda DataFrame named sales and load the sales.csv file into it.
#     Examine if your data is properly loaded by checking the first few values in the DataFrame by using the head() command:
sales = pd.read_csv("sales.csv")
sales.head()

# 3.	Group the Revenue by Order method type and create a bar plot:
sales.groupby('Order method type').sum().plot(kind = 'bar', y = 'Revenue', color='gray')

# 4.	Now group the columns by year and create boxplots to get an idea on a relative scale:
sales.groupby('Year')[['Revenue', 'Planned revenue', 'Gross profit']].plot(kind= 'box', color='gray')

```
#### **Activity 201: Analyzing Advertisements**
Your company has collated data on the advertisement views through various mediums in a file called Advertising.csv. 
The advert campaign ran through radio, TV, web, and newspaper and you need to mine the data to answer the following questions:
1.	What are the unique values present in the Products column? 
2.	How many data points belong to each category in the Products column?
3.	What are the total views across each category in the Products column?
4.	Which product has the highest viewership on TV?
5.	Which product has the lowest viewership on the web?
Follow the following steps to achieve the aim of this activity:
1.	Open a new Jupyter Notebook and load pandas and the visualization libraries that you will need.
2.	Load the data into a pandas DataFrame named ads and look at the first few rows. Your DataFrame should look as follows:
 ![Figure 2.63: The first few rows of Advertising.csv](Lab02/images/Activity2-02.jpg)
3.	Understand the distribution of numerical variables in the dataset using the describe function. 
4.	Plot the relationship between the variables in the dataset with the help of pair plots. You can use the hue parameter as Products. The hue parameter determines which column can be used for color encoding. Using Products as a hue parameter will show the different products in various shades of gray.
 ![Figure 2.64: The first few rows of Advertising.csv](Lab02/images/Activity2-01.jpg)

### Bài tập tổng hợp

#### Bài tập 1: Phân tích dữ liệu bán hàng

**Đề bài:** Cho dataset về doanh số bán hàng của một cửa hàng, hãy thực hiện phân tích toàn diện.

```python
# Tạo dữ liệu bán hàng mẫu
np.random.seed(123)
sales_data = {
    'order_id': range(1, 2001),
    'customer_id': np.random.randint(1, 500, 2000),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], 2000),
    'sales_amount': np.random.gamma(2, 50, 2000).astype(int),
    'quantity': np.random.randint(1, 10, 2000),
    'discount': np.random.uniform(0, 0.3, 2000),
    'order_date': pd.date_range('2023-01-01', periods=2000, freq='D')[:2000],
    'customer_age': np.random.randint(18, 70, 2000),
    'city': np.random.choice(['Hà Nội', 'TP.HCM', 'Đà Nẵng', 'Cần Thơ', 'Hải Phòng'], 2000),
    'payment_method': np.random.choice(['Credit Card', 'Cash', 'Bank Transfer'], 2000, p=[0.5, 0.3, 0.2])
}

sales_df = pd.DataFrame(sales_data)
sales_df['total_revenue'] = sales_df['sales_amount'] * sales_df['quantity'] * (1 - sales_df['discount'])

print("Dataset bán hàng đã tạo:")
print(sales_df.head())
print(f"Shape: {sales_df.shape}")
```

**Yêu cầu:**

1. **Khám phá dữ liệu cơ bản:**
   - Thống kê mô tả cho tất cả các biến
   - Kiểm tra dữ liệu thiếu và outliers
   - Phân tích phân bố của các biến chính

2. **Phân tích theo thời gian:**
   - Xu hướng doanh thu theo thời gian
   - Phân tích seasonal patterns
   - Tính growth rate

3. **Phân tích theo nhóm:**
   - So sánh doanh thu giữa các danh mục sản phẩm
   - Phân tích khách hàng theo độ tuổi
   - So sánh hiệu quả các phương thức thanh toán

4. **Trực quan hóa:**
   - Tạo dashboard tổng quan
   - Sử dụng cả Matplotlib và Seaborn
   - Tối thiểu 6 biểu đồ khác nhau

**Hướng dẫn giải:**

```python
# 1. Khám phá dữ liệu cơ bản
print("=== KHÁM PHÁ DỮ LIỆU CỞ BẢN ===")
print("\n1.1. Thống kê mô tả:")
print(sales_df.describe())

print("\n1.2. Thông tin dataset:")
print(sales_df.info())

print("\n1.3. Kiểm tra dữ liệu thiếu:")
print(sales_df.isnull().sum())

# 1.4. Phát hiện outliers
def analyze_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return len(outliers), (lower, upper)

numeric_cols = ['sales_amount', 'quantity', 'total_revenue', 'customer_age']
print("\n1.4. Phân tích outliers:")
for col in numeric_cols:
    outlier_count, bounds = analyze_outliers(sales_df, col)
    print(f"{col}: {outlier_count} outliers (bounds: {bounds[0]:.2f} - {bounds[1]:.2f})")

# 2. Phân tích theo thời gian
print("\n=== PHÂN TÍCH THEO THỜI GIAN ===")

# Tạo dữ liệu theo ngày
daily_sales = sales_df.groupby('order_date').agg({
    'total_revenue': 'sum',
    'order_id': 'count'
}).rename(columns={'order_id': 'order_count'})

# Thêm moving averages
daily_sales['revenue_ma_7'] = daily_sales['total_revenue'].rolling(7).mean()
daily_sales['revenue_ma_30'] = daily_sales['total_revenue'].rolling(30).mean()

# Tính growth rate
daily_sales['revenue_growth'] = daily_sales['total_revenue'].pct_change() * 100

print("2.1. Thống kê doanh thu hàng ngày:")
print(daily_sales['total_revenue'].describe())

# 2.2. Seasonal analysis
sales_df['month'] = sales_df['order_date'].dt.month
sales_df['day_of_week'] = sales_df['order_date'].dt.day_name()

monthly_revenue = sales_df.groupby('month')['total_revenue'].sum()
weekly_revenue = sales_df.groupby('day_of_week')['total_revenue'].mean()

print("\n2.2. Doanh thu theo tháng:")
print(monthly_revenue)

# 3. Phân tích theo nhóm
print("\n=== PHÂN TÍCH THEO NHÓM ===")

# 3.1. Phân tích theo danh mục sản phẩm
category_analysis = sales_df.groupby('product_category').agg({
    'total_revenue': ['sum', 'mean', 'count'],
    'sales_amount': 'mean',
    'discount': 'mean'
}).round(2)

print("3.1. Phân tích theo danh mục sản phẩm:")
print(category_analysis)

# 3.2. Phân tích theo độ tuổi khách hàng
sales_df['age_group'] = pd.cut(sales_df['customer_age'], 
                              bins=[0, 25, 35, 50, 70], 
                              labels=['18-25', '26-35', '36-50', '51-70'])

age_analysis = sales_df.groupby('age_group').agg({
    'total_revenue': ['sum', 'mean'],
    'quantity': 'mean',
    'discount': 'mean'
}).round(2)

print("\n3.2. Phân tích theo nhóm tuổi:")
print(age_analysis)

# 3.3. Phân tích phương thức thanh toán
payment_analysis = sales_df.groupby('payment_method').agg({
    'total_revenue': ['sum', 'mean', 'count'],
    'discount': 'mean'
}).round(2)

print("\n3.3. Phân tích theo phương thức thanh toán:")
print(payment_analysis)

# 4. Trực quan hóa - Dashboard
fig = plt.figure(figsize=(20, 15))

# 4.1. Xu hướng doanh thu theo thời gian
plt.subplot(3, 3, 1)
plt.plot(daily_sales.index, daily_sales['total_revenue'], alpha=0.6, label='Daily Revenue')
plt.plot(daily_sales.index, daily_sales['revenue_ma_7'], linewidth=2, label='MA 7 days')
plt.plot(daily_sales.index, daily_sales['revenue_ma_30'], linewidth=2, label='MA 30 days')
plt.title('Xu hướng Doanh thu Theo Thời gian')
plt.xlabel('Ngày')
plt.ylabel('Doanh thu')
plt.legend()
plt.xticks(rotation=45)

# 4.2. Doanh thu theo danh mục
plt.subplot(3, 3, 2)
category_revenue = sales_df.groupby('product_category')['total_revenue'].sum()
plt.bar(category_revenue.index, category_revenue.values, color='skyblue', alpha=0.8)
plt.title('Doanh thu Theo Danh mục Sản phẩm')
plt.xlabel('Danh mục')
plt.ylabel('Tổng Doanh thu')
plt.xticks(rotation=45)

# 4.3. Phân bố doanh thu
plt.subplot(3, 3, 3)
plt.hist(sales_df['total_revenue'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('Phân bố Doanh thu')
plt.xlabel('Doanh thu')
plt.ylabel('Tần số')

# 4.4. Boxplot doanh thu theo danh mục
plt.subplot(3, 3, 4)
sns.boxplot(data=sales_df, x='product_category', y='total_revenue')
plt.title('Phân bố Doanh thu Theo Danh mục')
plt.xticks(rotation=45)

# 4.5. Heatmap correlation
plt.subplot(3, 3, 5)
numeric_columns = ['sales_amount', 'quantity', 'discount', 'total_revenue', 'customer_age']
correlation_matrix = sales_df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Ma trận Correlation')

# 4.6. Doanh thu theo thành phố
plt.subplot(3, 3, 6)
city_revenue = sales_df.groupby('city')['total_revenue'].sum()
plt.pie(city_revenue.values, labels=city_revenue.index, autopct='%1.1f%%', startangle=90)
plt.title('Phân bố Doanh thu Theo Thành phố')

# 4.7. Scatter plot: Age vs Revenue
plt.subplot(3, 3, 7)
plt.scatter(sales_df['customer_age'], sales_df['total_revenue'], alpha=0.6)
plt.title('Mối quan hệ Tuổi - Doanh thu')
plt.xlabel('Tuổi khách hàng')
plt.ylabel('Doanh thu')

# 4.8. Doanh thu theo ngày trong tuần
plt.subplot(3, 3, 8)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_revenue_ordered = sales_df.groupby('day_of_week')['total_revenue'].mean().reindex(day_order)
plt.bar(range(len(weekly_revenue_ordered)), weekly_revenue_ordered.values, color='orange', alpha=0.8)
plt.title('Doanh thu Trung bình Theo Ngày trong Tuần')
plt.xlabel('Ngày')
plt.ylabel('Doanh thu TB')
plt.xticks(range(len(day_order)), [d[:3] for d in day_order])

# 4.9. Violin plot cho discount theo payment method
plt.subplot(3, 3, 9)
sns.violinplot(data=sales_df, x='payment_method', y='discount')
plt.title('Phân bố Discount Theo Phương thức Thanh toán')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Insights và kết luận
print("\n=== INSIGHTS VÀ KẾT LUẬN ===")

print("1. Xu hướng doanh thu:")
total_revenue = sales_df['total_revenue'].sum()
avg_daily_revenue = daily_sales['total_revenue'].mean()
print(f"   - Tổng doanh thu: {total_revenue:,.0f}")
print(f"   - Doanh thu trung bình/ngày: {avg_daily_revenue:,.0f}")

print("\n2. Danh mục sản phẩm tốt nhất:")
best_category = category_revenue.idxmax()
best_category_revenue = category_revenue.max()
print(f"   - Danh mục: {best_category}")
print(f"   - Doanh thu: {best_category_revenue:,.0f}")

print("\n3. Thành phố có doanh thu cao nhất:")
best_city = city_revenue.idxmax()
best_city_revenue = city_revenue.max()
print(f"   - Thành phố: {best_city}")
print(f"   - Doanh thu: {best_city_revenue:,.0f}")

print("\n4. Ngày trong tuần tốt nhất:")
best_day = weekly_revenue_ordered.idxmax()
best_day_revenue = weekly_revenue_ordered.max()
print(f"   - Ngày: {best_day}")
print(f"   - Doanh thu TB: {best_day_revenue:,.0f}")
```

#### Bài tập 2: Phân tích dữ liệu khách sạn

**Đề bài:** Phân tích dữ liệu đặt phòng khách sạn để tối ưu hóa doanh thu.

```python
# Tạo dữ liệu khách sạn
np.random.seed(456)
hotel_data = {
    'booking_id': range(1, 5001),
    'hotel_type': np.random.choice(['Luxury', 'Business', 'Budget'], 5000, p=[0.2, 0.3, 0.5]),
    'room_type': np.random.choice(['Single', 'Double', 'Suite', 'Family'], 5000),
    'check_in_date': pd.date_range('2023-01-01', periods=5000, freq='D')[:5000],
    'nights_stayed': np.random.randint(1, 15, 5000),
    'guests_count': np.random.randint(1, 6, 5000),
    'room_rate': np.random.uniform(50, 500, 5000).round(2),
    'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], 5000),
    'booking_channel': np.random.choice(['Direct', 'Online', 'Travel Agent'], 5000, p=[0.3, 0.5, 0.2]),
    'customer_type': np.random.choice(['New', 'Returning'], 5000, p=[0.6, 0.4]),
    'cancellation': np.random.choice([0, 1], 5000, p=[0.85, 0.15]),
    'special_requests': np.random.randint(0, 5, 5000)
}

hotel_df = pd.DataFrame(hotel_data)
hotel_df['total_revenue'] = hotel_df['room_rate'] * hotel_df['nights_stayed'] * (1 - hotel_df['cancellation'])
hotel_df['check_out_date'] = hotel_df['check_in_date'] + pd.to_timedelta(hotel_df['nights_stayed'], unit='D')

print("Dataset khách sạn:")
print(hotel_df.head())
```

**Yêu cầu thực hiện:**

1. **Revenue Analysis:**
   - Doanh thu theo loại khách sạn và phòng
   - Tỷ lệ hủy phòng và tác động
   - ADR (Average Daily Rate) analysis

2. **Customer Segmentation:**
   - Phân tích theo loại khách hàng
   - Booking patterns theo channel
   - Guest behavior analysis

3. **Seasonal Analysis:**
   - Occupancy rate theo mùa
   - Pricing strategy theo thời gian
   - Demand forecasting

4. **Advanced Analytics:**
   - Clustering khách hàng
   - Prediction models
   - Recommendation engine

#### Bài tập 3: Phân tích dữ liệu tài chính

**Đề bài:** Phân tích portfolio đầu tư và rủi ro.

```python
# Tạo dữ liệu tài chính
np.random.seed(789)
stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX']
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')

# Tạo dữ liệu giá cổ phiếu với random walk
stock_data = {}
for stock in stocks:
    # Tạo returns ngẫu nhiên
    returns = np.random.normal(0.001, 0.02, len(dates))
    # Tạo giá từ returns (cumulative product)
    prices = 100 * np.exp(np.cumsum(returns))
    stock_data[stock] = prices

stock_df = pd.DataFrame(stock_data, index=dates)

# Thêm volume data
volume_data = {}
for stock in stocks:
    volume_data[f'{stock}_volume'] = np.random.randint(1000000, 10000000, len(dates))

volume_df = pd.DataFrame(volume_data, index=dates)

print("Sample stock data:")
print(stock_df.head())
```

**Yêu cầu:**

1. **Return Analysis:**
   - Daily, weekly, monthly returns
   - Risk metrics (volatility, VaR, Sharpe ratio)
   - Correlation analysis

2. **Portfolio Optimization:**
   - Efficient frontier
   - Risk-return analysis
   - Diversification benefits

3. **Technical Analysis:**
   - Moving averages
   - Bollinger Bands
   - RSI indicators

4. **Risk Management:**
   - Monte Carlo simulation
   - Stress testing
   - Portfolio rebalancing
