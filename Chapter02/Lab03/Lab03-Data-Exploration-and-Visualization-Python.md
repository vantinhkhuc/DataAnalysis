# Lab 03: Data Exploration and Visualization với Python


## Giới thiệu
Bộ bài tập này được thiết kế để học viên làm chủ các kỹ năng khám phá, phân tích và trực quan hóa dữ liệu bằng Python. Các bài tập được chia thành 4 phần chính, từ cơ bản đến nâng cao.

**Yêu cầu tiên quyết:**
- Python cơ bản (biến, vòng lặp, hàm)
- Kiến thức cơ bản về pandas và numpy
- Jupyter Notebook hoặc IDE Python

---

## PHẦN 1: LEARN TO EXPLORE, ANALYZE, AND RESHAPE YOUR DATA

### Mô tả lý thuyết
Khám phá dữ liệu (Data Exploration) là bước đầu tiên và quan trọng nhất trong quy trình phân tích dữ liệu. Giai đoạn này giúp chúng ta:
- Hiểu cấu trúc và đặc điểm của dữ liệu
- Phát hiện các vấn đề như dữ liệu thiếu, outliers
- Xác định các mối quan hệ tiềm năng giữa các biến
- Định hình hướng phân tích tiếp theo

### Các kỹ thuật chính:
1. **Data Loading & Initial Inspection**: Tải dữ liệu và khảo sát ban đầu
2. **Data Cleaning**: Làm sạch dữ liệu
3. **Data Reshaping**: Biến đổi cấu trúc dữ liệu
4. **Data Filtering & Selection**: Lọc và chọn dữ liệu

### Bài tập 1.1: Khám phá dữ liệu bán hàng cơ bản

```python
import pandas as pd
import numpy as np

# Tạo dataset mẫu về bán hàng
np.random.seed(42)
data = {
    'order_id': range(1, 1001),
    'customer_id': np.random.randint(1, 201, 1000),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 1000),
    'sales_amount': np.random.normal(100, 30, 1000).round(2),
    'order_date': pd.date_range('2023-01-01', periods=1000, freq='D')[:1000],
    'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
    'customer_age': np.random.randint(18, 70, 1000)
}
df = pd.DataFrame(data)

# Nhiệm vụ của học viên:
print("=== BÀI TẬP 1.1: KHÁM PHÁ DỮ LIỆU CƠ BẢN ===")
print("1. Hiển thị thông tin tổng quan về dataset (shape, dtypes, memory usage)")
print("2. Xem 5 dòng đầu và 5 dòng cuối")
print("3. Kiểm tra dữ liệu thiếu (null values)")
print("4. Hiển thị các giá trị unique của từng cột categorical")
print("5. Tính toán các thống kê cơ bản cho cột sales_amount")
```

**Đáp án mẫu:**
```python
# 1. Thông tin tổng quan
print("Thông tin dataset:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nData types:")
print(df.dtypes)
print(f"\nMemory usage: {df.memory_usage(deep=True).sum()} bytes")

# 2. Xem dữ liệu mẫu
print("\n5 dòng đầu:")
print(df.head())
print("\n5 dòng cuối:")
print(df.tail())

# 3. Kiểm tra dữ liệu thiếu
print("\nDữ liệu thiếu:")
print(df.isnull().sum())

# 4. Giá trị unique
print("\nGiá trị unique:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"{col}: {df[col].unique()}")

# 5. Thống kê cơ bản sales_amount
print("\nThống kê sales_amount:")
print(df['sales_amount'].describe())
```

### Bài tập 1.2: Làm sạch và biến đổi dữ liệu

```python
# Tạo dữ liệu có vấn đề để thực hành
df_dirty = df.copy()

# Thêm một số vấn đề vào dữ liệu
df_dirty.loc[10:15, 'sales_amount'] = np.nan  # Dữ liệu thiếu
df_dirty.loc[20:25, 'customer_age'] = -1      # Dữ liệu không hợp lệ
df_dirty.loc[30:35, 'sales_amount'] = 1000    # Outliers

print("=== BÀI TẬP 1.2: LÀM SẠCH DỮ LIỆU ===")
print("1. Tìm và xử lý dữ liệu thiếu trong cột sales_amount")
print("2. Tìm và xử lý giá trị âm trong cột customer_age")
print("3. Phát hiện và xử lý outliers trong sales_amount (sử dụng IQR method)")
print("4. Tạo cột mới 'age_group' dựa trên customer_age")
print("5. Chuyển đổi order_date sang định dạng datetime và tách thành year, month")
```

**Đáp án mẫu:**
```python
# 1. Xử lý dữ liệu thiếu
print("Dữ liệu thiếu trước xử lý:", df_dirty['sales_amount'].isnull().sum())
df_dirty['sales_amount'].fillna(df_dirty['sales_amount'].median(), inplace=True)
print("Dữ liệu thiếu sau xử lý:", df_dirty['sales_amount'].isnull().sum())

# 2. Xử lý giá trị âm
print("\nGiá trị âm customer_age:", (df_dirty['customer_age'] < 0).sum())
df_dirty.loc[df_dirty['customer_age'] < 0, 'customer_age'] = df_dirty['customer_age'].median()

# 3. Xử lý outliers bằng IQR
Q1 = df_dirty['sales_amount'].quantile(0.25)
Q3 = df_dirty['sales_amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nOutliers trước xử lý: {((df_dirty['sales_amount'] < lower_bound) | (df_dirty['sales_amount'] > upper_bound)).sum()}")
df_dirty.loc[df_dirty['sales_amount'] > upper_bound, 'sales_amount'] = upper_bound
df_dirty.loc[df_dirty['sales_amount'] < lower_bound, 'sales_amount'] = lower_bound

# 4. Tạo age_group
df_dirty['age_group'] = pd.cut(df_dirty['customer_age'], 
                               bins=[0, 25, 35, 50, 100], 
                               labels=['18-25', '26-35', '36-50', '50+'])

# 5. Xử lý datetime
df_dirty['order_date'] = pd.to_datetime(df_dirty['order_date'])
df_dirty['year'] = df_dirty['order_date'].dt.year
df_dirty['month'] = df_dirty['order_date'].dt.month

print(f"\nDataset sau làm sạch: {df_dirty.shape}")
print(df_dirty.head())
```

### Bài tập 1.3: Biến đổi cấu trúc dữ liệu (Reshaping)

```python
print("=== BÀI TẬP 1.3: BIẾN ĐỔI CẤU TRÚC DỮ LIỆU ===")
print("1. Tạo pivot table: doanh thu theo region và product_category")
print("2. Melt dữ liệu để chuyển từ wide format sang long format")
print("3. Group by và aggregate dữ liệu theo nhiều tiêu chí")
print("4. Sắp xếp dữ liệu theo multiple columns")
```

**Đáp án mẫu:**
```python
# 1. Pivot table
pivot_sales = df_dirty.pivot_table(
    values='sales_amount',
    index='region',
    columns='product_category',
    aggfunc='sum',
    fill_value=0
)
print("Pivot table doanh thu:")
print(pivot_sales)

# 2. Melt data
melted_data = pivot_sales.reset_index().melt(
    id_vars='region',
    var_name='category',
    value_name='total_sales'
)
print("\nDữ liệu sau khi melt:")
print(melted_data.head())

# 3. Group by với multiple aggregations
grouped_stats = df_dirty.groupby(['region', 'product_category']).agg({
    'sales_amount': ['sum', 'mean', 'count'],
    'customer_age': 'mean'
}).round(2)
print("\nGroup by statistics:")
print(grouped_stats.head())

# 4. Sorting
sorted_data = df_dirty.sort_values(['region', 'sales_amount'], ascending=[True, False])
print("\nTop 5 doanh thu cao nhất mỗi region:")
print(sorted_data.groupby('region').head(2)[['region', 'product_category', 'sales_amount']])
```

---

## PHẦN 2: DISCOVER FUNCTIONS FOR SUMMARY AND DESCRIPTIVE STATISTICS

### Mô tả lý thuyết
Thống kê mô tả (Descriptive Statistics) giúp chúng ta hiểu được đặc điểm cơ bản của dữ liệu thông qua các chỉ số như:
- **Xu hướng trung tâm**: Mean, Median, Mode
- **Độ phân tán**: Standard deviation, Variance, Range, IQR
- **Phân phối**: Skewness, Kurtosis, Percentiles
- **Mối quan hệ**: Correlation, Covariance

### Các hàm quan trọng trong pandas/numpy:
- `describe()`, `info()`, `value_counts()`
- `mean()`, `median()`, `std()`, `var()`
- `quantile()`, `corr()`, `cov()`
- `skew()`, `kurtosis()`

### Bài tập 2.1: Thống kê mô tả cơ bản

```python
print("=== BÀI TẬP 2.1: THỐNG KÊ MÔ TẢ CƠ BẢN ===")
print("1. Tính các thống kê cơ bản cho tất cả biến số")
print("2. Tính mode cho các biến categorical")
print("3. Tính percentiles (10%, 25%, 50%, 75%, 90%) cho sales_amount")
print("4. So sánh mean vs median để đánh giá độ lệch phân phối")
print("5. Tính coefficient of variation cho sales_amount theo từng region")
```

**Đáp án mẫu:**
```python
# 1. Thống kê cơ bản
print("Thống kê tổng quan:")
print(df_dirty.describe())

print("\nThống kê cho biến categorical:")
print(df_dirty.describe(include=['object', 'category']))

# 2. Mode cho biến categorical
print("\nMode của các biến categorical:")
categorical_cols = ['product_category', 'region', 'age_group']
for col in categorical_cols:
    mode_value = df_dirty[col].mode()[0]
    print(f"{col}: {mode_value}")

# 3. Percentiles
print("\nPercentiles của sales_amount:")
percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
for p in percentiles:
    value = df_dirty['sales_amount'].quantile(p)
    print(f"{p*100}%: {value:.2f}")

# 4. So sánh mean vs median
mean_sales = df_dirty['sales_amount'].mean()
median_sales = df_dirty['sales_amount'].median()
print(f"\nMean: {mean_sales:.2f}")
print(f"Median: {median_sales:.2f}")
print(f"Skewness: {df_dirty['sales_amount'].skew():.2f}")

if mean_sales > median_sales:
    print("Phân phối lệch phải (right-skewed)")
elif mean_sales < median_sales:
    print("Phân phối lệch trái (left-skewed)")
else:
    print("Phân phối đối xứng")

# 5. Coefficient of variation theo region
print("\nCoefficient of Variation theo region:")
cv_by_region = df_dirty.groupby('region')['sales_amount'].agg(['mean', 'std'])
cv_by_region['cv'] = (cv_by_region['std'] / cv_by_region['mean']) * 100
print(cv_by_region[['cv']].round(2))
```

### Bài tập 2.2: Phân tích phân phối và outliers

```python
print("=== BÀI TẬP 2.2: PHÂN TÍCH PHÂN PHỐI VÀ OUTLIERS ===")
print("1. Tính skewness và kurtosis cho sales_amount")
print("2. Tìm outliers bằng Z-score method")
print("3. Tạo frequency table cho product_category")
print("4. Tính cross-tabulation giữa region và age_group")
print("5. Phân tích phân phối sales_amount theo từng category")
```

**Đáp án mẫu:**
```python
from scipy import stats

# 1. Skewness và Kurtosis
skewness = df_dirty['sales_amount'].skew()
kurtosis = df_dirty['sales_amount'].kurtosis()
print(f"Skewness: {skewness:.3f}")
print(f"Kurtosis: {kurtosis:.3f}")

# Giải thích kết quả
if abs(skewness) < 0.5:
    skew_desc = "gần như đối xứng"
elif abs(skewness) < 1:
    skew_desc = "lệch vừa phải"
else:
    skew_desc = "lệch mạnh"
print(f"Phân phối {skew_desc}")

# 2. Z-score outliers
z_scores = np.abs(stats.zscore(df_dirty['sales_amount']))
outliers_zscore = df_dirty[z_scores > 3]
print(f"\nOutliers (Z-score > 3): {len(outliers_zscore)}")

# 3. Frequency table
print("\nFrequency table - Product Category:")
freq_table = df_dirty['product_category'].value_counts()
freq_percentage = df_dirty['product_category'].value_counts(normalize=True) * 100
freq_df = pd.DataFrame({
    'Count': freq_table,
    'Percentage': freq_percentage.round(2)
})
print(freq_df)

# 4. Cross-tabulation
print("\nCross-tabulation: Region vs Age Group:")
cross_tab = pd.crosstab(df_dirty['region'], df_dirty['age_group'], margins=True)
print(cross_tab)

# 5. Phân phối sales_amount theo category
print("\nPhân phối sales_amount theo Product Category:")
category_stats = df_dirty.groupby('product_category')['sales_amount'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max'
]).round(2)
print(category_stats)
```

### Bài tập 2.3: Ma trận tương quan và mối quan hệ

```python
print("=== BÀI TẬP 2.3: MA TRẬN TƯƠNG QUAN VÀ MỐI QUAN HỆ ===")
print("1. Tạo ma trận tương quan cho các biến số")
print("2. Tìm cặp biến có tương quan mạnh nhất")
print("3. Tính tương quan điểm (point-biserial) giữa biến liên tục và categorical")
print("4. Phân tích mối quan hệ giữa customer_age và sales_amount")
```

**Đáp án mẫu:**
```python
# 1. Ma trận tương quan
numeric_cols = ['sales_amount', 'customer_age', 'year', 'month']
corr_matrix = df_dirty[numeric_cols].corr()
print("Ma trận tương quan:")
print(corr_matrix.round(3))

# 2. Tìm tương quan mạnh nhất
# Loại bỏ tự tương quan (diagonal)
corr_matrix_masked = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool))
max_corr_idx = corr_matrix_masked.abs().stack().idxmax()
max_corr_value = corr_matrix_masked.stack()[max_corr_idx]
print(f"\nTương quan mạnh nhất: {max_corr_idx[0]} vs {max_corr_idx[1]} = {max_corr_value:.3f}")

# 3. Point-biserial correlation (tương quan giữa biến liên tục và categorical)
from scipy.stats import pointbiserialr

# Encode categorical variable to binary
df_dirty['is_electronics'] = (df_dirty['product_category'] == 'Electronics').astype(int)
corr_coef, p_value = pointbiserialr(df_dirty['is_electronics'], df_dirty['sales_amount'])
print(f"\nTương quan điểm giữa Electronics và Sales Amount: {corr_coef:.3f} (p-value: {p_value:.3f})")

# 4. Phân tích chi tiết age vs sales_amount
print("\nPhân tích mối quan hệ Age vs Sales Amount:")
age_groups = pd.cut(df_dirty['customer_age'], bins=5)
age_sales_analysis = df_dirty.groupby(age_groups)['sales_amount'].agg([
    'count', 'mean', 'median', 'std'
]).round(2)
print(age_sales_analysis)

# Tính tương quan Spearman (không tham số)
from scipy.stats import spearmanr
spearman_corr, spearman_p = spearmanr(df_dirty['customer_age'], df_dirty['sales_amount'])
print(f"\nSpearman correlation: {spearman_corr:.3f} (p-value: {spearman_p:.3f})")
```

---

## PHẦN 3: BUILD PIVOT TABLES AND COMPARATIVE ANALYSES

### Mô tả lý thuyết
Pivot tables là công cụ mạnh mẽ để tóm tắt, phân tích và so sánh dữ liệu theo nhiều chiều. Chúng giúp:
- Tóm tắt dữ liệu lớn thành thông tin có ý nghĩa
- So sánh giữa các nhóm khác nhau
- Phát hiện patterns và trends
- Thực hiện các phép kiểm định thống kê

### Các kỹ thuật chính:
1. **Basic Pivot Tables**: Tạo bảng pivot cơ bản
2. **Multi-level Pivots**: Pivot tables nhiều cấp
3. **Statistical Tests**: T-test, Chi-square, ANOVA
4. **Comparative Analysis**: So sánh giữa các nhóm

### Bài tập 3.1: Pivot Tables cơ bản và nâng cao

```python
print("=== BÀI TẬP 3.1: PIVOT TABLES CƠ BẢN VÀ NÂNG CAO ===")
print("1. Tạo pivot table đơn giản: doanh thu theo region")
print("2. Pivot table nhiều chiều: region vs product_category vs age_group")
print("3. Pivot table với multiple aggregations")
print("4. Tính tỷ lệ phần trăm trong pivot table")
print("5. Pivot table với custom aggregation functions")
```

**Đáp án mẫu:**
```python
# 1. Pivot table đơn giản
print("1. Doanh thu theo Region:")
pivot_simple = df_dirty.pivot_table(
    values='sales_amount',
    index='region',
    aggfunc=['sum', 'mean', 'count']
).round(2)
print(pivot_simple)

# 2. Pivot table nhiều chiều
print("\n2. Pivot table nhiều chiều:")
pivot_multi = df_dirty.pivot_table(
    values='sales_amount',
    index=['region', 'product_category'],
    columns='age_group',
    aggfunc='mean',
    fill_value=0
).round(2)
print(pivot_multi.head(10))

# 3. Multiple aggregations
print("\n3. Multiple aggregations:")
pivot_agg = df_dirty.pivot_table(
    values=['sales_amount', 'customer_age'],
    index='product_category',
    aggfunc={
        'sales_amount': ['sum', 'mean', 'std'],
        'customer_age': ['mean', 'median']
    }
).round(2)
print(pivot_agg)

# 4. Tỷ lệ phần trăm
print("\n4. Tỷ lệ phần trăm:")
pivot_pct = pd.crosstab(df_dirty['region'], df_dirty['product_category'], normalize='index') * 100
print(pivot_pct.round(1))

# 5. Custom aggregation
def coefficient_variation(x):
    return (x.std() / x.mean()) * 100 if x.mean() != 0 else 0

print("\n5. Custom aggregation (CV):")
pivot_custom = df_dirty.pivot_table(
    values='sales_amount',
    index='region',
    columns='product_category',
    aggfunc=coefficient_variation
).round(2)
print(pivot_custom)
```

### Bài tập 3.2: Phân tích so sánh và kiểm định thống kê

```python
print("=== BÀI TẬP 3.2: PHÂN TÍCH SO SÁNH VÀ KIỂM ĐỊNH ===")
print("1. So sánh doanh thu trung bình giữa các regions bằng ANOVA")
print("2. T-test so sánh doanh thu giữa Electronics vs Non-Electronics")
print("3. Chi-square test cho mối quan hệ giữa region và product_category")
print("4. Phân tích seasonal trends theo month")
print("5. Cohort analysis đơn giản")
```

**Đáp án mẫu:**
```python
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 1. ANOVA test
print("1. ANOVA Test - So sánh doanh thu giữa các regions:")
regions = [group['sales_amount'].values for name, group in df_dirty.groupby('region')]
f_stat, p_value = stats.f_oneway(*regions)
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Kết luận: {'Có sự khác biệt' if p_value < 0.05 else 'Không có sự khác biệt'} giữa các regions")

# Post-hoc analysis với Tukey HSD
from scipy.stats import tukey_hsd
region_names = df_dirty['region'].unique()
region_data = [df_dirty[df_dirty['region']==r]['sales_amount'] for r in region_names]
tukey_result = tukey_hsd(*region_data)
print("\nTukey HSD Post-hoc Analysis:")
print(tukey_result)

# 2. T-test
print("\n2. T-test - Electronics vs Non-Electronics:")
electronics = df_dirty[df_dirty['product_category'] == 'Electronics']['sales_amount']
non_electronics = df_dirty[df_dirty['product_category'] != 'Electronics']['sales_amount']

t_stat, t_p_value = stats.ttest_ind(electronics, non_electronics)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {t_p_value:.4f}")
print(f"Electronics mean: {electronics.mean():.2f}")
print(f"Non-Electronics mean: {non_electronics.mean():.2f}")

# 3. Chi-square test
print("\n3. Chi-square test - Region vs Product Category:")
contingency_table = pd.crosstab(df_dirty['region'], df_dirty['product_category'])
chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square: {chi2:.4f}")
print(f"P-value: {chi2_p:.4f}")
print(f"Degrees of freedom: {dof}")
print("Contingency Table:")
print(contingency_table)

# 4. Seasonal analysis
print("\n4. Seasonal Trends:")
monthly_sales = df_dirty.groupby('month')['sales_amount'].agg(['sum', 'mean', 'count'])
monthly_sales['month_name'] = pd.to_datetime(monthly_sales.index, format='%m').strftime('%B')
monthly_sales = monthly_sales.round(2)
print(monthly_sales)

# Seasonal trend test
months = df_dirty['month'].values
sales = df_dirty['sales_amount'].values
seasonal_corr, seasonal_p = stats.pearsonr(months, sales)
print(f"\nSeasonal correlation: {seasonal_corr:.4f} (p-value: {seasonal_p:.4f})")

# 5. Cohort analysis đơn giản
print("\n5. Simple Cohort Analysis:")
df_dirty['order_period'] = df_dirty['order_date'].dt.to_period('M')
cohort_table = df_dirty.groupby(['customer_id', 'order_period']).size().reset_index(name='orders')
cohort_summary = cohort_table.groupby('order_period').agg({
    'customer_id': 'nunique',
    'orders': 'sum'
}).rename(columns={'customer_id': 'unique_customers', 'orders': 'total_orders'})
cohort_summary['avg_orders_per_customer'] = (cohort_summary['total_orders'] / 
                                            cohort_summary['unique_customers']).round(2)
print(cohort_summary.head())
```

### Bài tập 3.3: Advanced Pivot Analysis

```python
print("=== BÀI TẬP 3.3: PHÂN TÍCH PIVOT NÂNG CAO ===")
print("1. Rolling window analysis")
print("2. Pivot với conditional formatting logic")
print("3. Multi-index pivot với percentage calculations")
print("4. Time-based pivot analysis")
print("5. Customer segmentation analysis")
```

**Đáp án mẫu:**
```python
# 1. Rolling window analysis
print("1. Rolling Window Analysis (7-day moving average):")
daily_sales = df_dirty.groupby(df_dirty['order_date'].dt.date)['sales_amount'].sum().reset_index()
daily_sales['sales_7day_ma'] = daily_sales['sales_amount'].rolling(window=7).mean()
print(daily_sales.head(10))

# 2. Conditional formatting logic trong pivot
print("\n2. Pivot với conditional analysis:")
pivot_performance = df_dirty.pivot_table(
    values='sales_amount',
    index='region',
    columns='product_category',
    aggfunc='mean'
).round(2)

# Thêm performance indicators
overall_mean = df_dirty['sales_amount'].mean()
performance_matrix = pivot_performance.applymap(lambda x: 'Above Average' if x > overall_mean else 'Below Average')
print("Performance Matrix:")
print(performance_matrix)

# 3. Multi-index với percentage
print("\n3. Multi-index Pivot với Percentage:")
multi_pivot = df_dirty.pivot_table(
    values='sales_amount',
    index=['region', 'age_group'],
    columns='product_category',
    aggfunc=['sum', 'count'],
    fill_value=0
)

# Tính percentage của tổng
total_sales = multi_pivot['sum'].sum().sum()
pct_pivot = (multi_pivot['sum'] / total_sales * 100).round(2)
print("Percentage of Total Sales:")
print(pct_pivot.head())

# 4. Time-based analysis
print("\n4. Time-based Pivot Analysis:")
df_dirty['quarter'] = df_dirty['order_date'].dt.quarter
time_pivot = df_dirty.pivot_table(
    values='sales_amount',
    index='quarter',
    columns=['region', 'product_category'],
```
---

## PHẦN 4: CREATE IMPACTFUL VISUALIZATIONS WITH MATPLOTLIB AND SEABORN

### Mô tả lý thuyết
Data Visualization là nghệ thuật và khoa học biến dữ liệu thành hình ảnh có ý nghĩa. Visualization hiệu quả giúp:
- Truyền đạt insights một cách rõ ràng và thuyết phục
- Phát hiện patterns, trends, và outliers
- So sánh và đối chiếu dữ liệu
- Kể câu chuyện với dữ liệu (data storytelling)

### Matplotlib vs Seaborn:
- **Matplotlib**: Thư viện visualization cơ bản, linh hoạt, control chi tiết
- **Seaborn**: Built trên Matplotlib, syntax đơn giản, đẹp mắt hơn, tích hợp tốt với pandas

### Các loại biểu đồ chính:
1. **Distribution plots**: Histogram, KDE, Box plot, Violin plot
2. **Relationship plots**: Scatter plot, Line plot, Correlation heatmap
3. **Categorical plots**: Bar plot, Count plot, Point plot
4. **Multi-dimensional plots**: Pair plot, Facet grid

### Bài tập 4.1: Biểu đồ phân phối (Distribution Plots)

```python
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== BÀI TẬP 4.1: BIỂU ĐỒ PHÂN PHỐI ===")
print("1. Histogram với KDE overlay")
print("2. Box plot so sánh phân phối sales_amount theo region")
print("3. Violin plot kết hợp với strip plot")
print("4. Distribution plot với multiple groups")
print("5. QQ plot để kiểm tra tính chuẩn của phân phối")
```

**Đáp án mẫu:**
```python
# Setup figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Distribution Analysis', fontsize=16, y=1.02)

# 1. Histogram với KDE
ax1 = axes[0, 0]
sns.histplot(data=df_dirty, x='sales_amount', kde=True, ax=ax1)
ax1.set_title('Sales Amount Distribution')
ax1.axvline(df_dirty['sales_amount'].mean(), color='red', linestyle='--', label='Mean')
ax1.axvline(df_dirty['sales_amount'].median(), color='green', linestyle='--', label='Median')
ax1.legend()

# 2. Box plot theo region
ax2 = axes[0, 1]
sns.boxplot(data=df_dirty, x='region', y='sales_amount', ax=ax2)
ax2.set_title('Sales Distribution by Region')
ax2.tick_params(axis='x', rotation=45)

# 3. Violin plot với strip plot
ax3 = axes[0, 2]
sns.violinplot(data=df_dirty, x='product_category', y='sales_amount', ax=ax3)
sns.stripplot(data=df_dirty, x='product_category', y='sales_amount', 
              size=3, alpha=0.3, ax=ax3)
ax3.set_title('Sales Distribution by Category')
ax3.tick_params(axis='x', rotation=45)

# 4. Multiple group distribution
ax4 = axes[1, 0]
for region in df_dirty['region'].unique():
    data = df_dirty[df_dirty['region'] == region]['sales_amount']
    sns.kdeplot(data=data, label=region, ax=ax4)
ax4.set_title('Sales Distribution by Region (KDE)')
ax4.legend()

# 5. QQ plot
ax5 = axes[1, 1]
from scipy.stats import probplot
probplot(df_dirty['sales_amount'], dist="norm", plot=ax5)
ax5.set_title('Q-Q Plot (Normal Distribution)')

# 6. Cumulative distribution
ax6 = axes[1, 2]
for category in df_dirty['product_category'].unique():
    data = df_dirty[df_dirty['product_category'] == category]['sales_amount']
    ax6.hist(data, bins=30, alpha=0.5, label=category, cumulative=True, density=True)
ax6.set_title('Cumulative Distribution by Category')
ax6.legend()

plt.tight_layout()
plt.show()

# Statistical summary của distributions
print("\nStatistical Summary by Region:")
distribution_summary = df_dirty.groupby('region')['sales_amount'].agg([
    'count', 'mean', 'median', 'std', 'skew'
]).round(3)
print(distribution_summary)
```

### Bài tập 4.2: Biểu đồ mối quan hệ (Relationship Plots)

```python
print("=== BÀI TẬP 4.2: BIỂU ĐỒ MỐI QUAN HỆ ===")
print("1. Scatter plot với regression line")
print("2. Correlation heatmap")
print("3. Pair plot cho multiple variables")
print("4. Line plot cho time series")
print("5. Joint plot với marginal distributions")
```

**Đáp án mẫu:**
```python
# Setup figure
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Relationship Analysis', fontsize=16, y=1.02)

# 1. Scatter plot với regression
ax1 = axes[0, 0]
sns.scatterplot(data=df_dirty, x='customer_age', y='sales_amount', 
                hue='region', alpha=0.6, ax=ax1)
sns.regplot(data=df_dirty, x='customer_age', y='sales_amount', 
            scatter=False, color='black', ax=ax1)
ax1.set_title('Age vs Sales Amount')

# 2. Correlation heatmap
ax2 = axes[0, 1]
numeric_cols = ['sales_amount', 'customer_age', 'month', 'year']
corr_matrix = df_dirty[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f', ax=ax2)
ax2.set_title('Correlation Matrix')

# 3. Scatter matrix plot (manual pair plot)
ax3 = axes[0, 2]
from pandas.plotting import scatter_matrix
scatter_matrix(df_dirty[['sales_amount', 'customer_age']], 
               alpha=0.5, figsize=(6, 6), ax=ax3)
ax3.set_title('Scatter Matrix')

# 4. Time series line plot
ax4 = axes[1, 0]
daily_sales = df_dirty.groupby(df_dirty['order_date'].dt.date)['sales_amount'].sum()
daily_sales_ma = daily_sales.rolling(window=7).mean()

ax4.plot(daily_sales.index, daily_sales.values, alpha=0.3, label='Daily Sales')
ax4.plot(daily_sales_ma.index, daily_sales_ma.values, label='7-day MA', linewidth=2)
ax4.set_title('Daily Sales Trend')
ax4.legend()
ax4.tick_params(axis='x', rotation=45)

# 5. Regression plot by category
ax5 = axes[1, 1]
sns.lmplot(data=df_dirty, x='customer_age', y='sales_amount', 
           hue='product_category', col='region', col_wrap=2,
           height=4, aspect=0.8)
plt.show()

# 6. Joint plot (separate figure)
print("\n6. Joint Plot với Marginal Distributions:")
g = sns.jointplot(data=df_dirty, x='customer_age', y='sales_amount', 
                  kind='reg', height=8)
g.plot_joint(sns.scatterplot, alpha=0.5)
plt.show()

plt.tight_layout()
plt.show()

# Advanced correlation analysis
print("\nAdvanced Correlation Analysis:")
# Partial correlation (controlling for age)
from scipy.stats import pearsonr

# Correlation matrix với significance levels
def correlation_significance(df, cols):
    corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    p_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
    for i in cols:
        for j in cols:
            if i != j:
                corr, p_val = pearsonr(df[i], df[j])
                corr_matrix.loc[i, j] = corr
                p_matrix.loc[i, j] = p_val
            else:
                corr_matrix.loc[i, j] = 1.0
                p_matrix.loc[i, j] = 0.0
    
    return corr_matrix, p_matrix

corr_vals, p_vals = correlation_significance(df_dirty, numeric_cols)
print("Correlation with p-values:")
print(corr_vals.round(3))
print("\nP-values:")
print(p_vals.round(4))
```

### Bài tập 4.3: Biểu đồ categorical và advanced plots

```python
print("=== BÀI TẬP 4.3: BIỂU ĐỒ CATEGORICAL VÀ ADVANCED PLOTS ===")
print("1. Bar plot với error bars")
print("2. Count plot với percentages")
print("3. Grouped bar chart")
print("4. Stacked bar chart")
print("5. Radar chart cho multi-dimensional comparison")
print("6. Sankey diagram cho flow analysis")
```

**Đáp án mẫu:**
```python
# Setup figure
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('Categorical and Advanced Analysis', fontsize=16, y=0.98)

# 1. Bar plot với error bars
ax1 = axes[0, 0]
region_stats = df_dirty.groupby('region')['sales_amount'].agg(['mean', 'std', 'count'])
region_stats.plot(kind='bar', y='mean', yerr='std', ax=ax1, capsize=4)
ax1.set_title('Average Sales by Region (with Error Bars)')
ax1.tick_params(axis='x', rotation=45)

# 2. Count plot với percentages
ax2 = axes[0, 1]
category_counts = df_dirty['product_category'].value_counts()
wedges, texts, autotexts = ax2.pie(category_counts.values, labels=category_counts.index, 
                                   autopct='%1.1f%%', startangle=90)
ax2.set_title('Product Category Distribution')

# 3. Grouped bar chart
ax3 = axes[1, 0]
grouped_data = df_dirty.pivot_table(values='sales_amount', 
                                   index='region', 
                                   columns='product_category', 
                                   aggfunc='mean')
grouped_data.plot(kind='bar', ax=ax3)
ax3.set_title('Average Sales: Region vs Category')
ax3.tick_params(axis='x', rotation=45)
ax3.legend(title='Product Category', bbox_to_anchor=(1.05, 1))

# 4. Stacked bar chart
ax4 = axes[1, 1]
stacked_data = df_dirty.groupby(['region', 'product_category']).size().unstack(fill_value=0)
stacked_data.plot(kind='bar', stacked=True, ax=ax4)
ax4.set_title('Order Count: Stacked by Region and Category')
ax4.tick_params(axis='x', rotation=45)

# 5. Heatmap với annotations
ax5 = axes[2, 0]
pivot_heatmap = df_dirty.pivot_table(values='sales_amount', 
                                    index='region', 
                                    columns='age_group', 
                                    aggfunc='mean')
sns.heatmap(pivot_heatmap, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax5)
ax5.set_title('Sales Heatmap: Region vs Age Group')

# 6. Advanced subplot - Multiple metrics
ax6 = axes[2, 1]
metrics_by_region = df_dirty.groupby('region').agg({
    'sales_amount': ['mean', 'median'],
    'customer_age': 'mean',
    'order_id': 'count'
})
metrics_by_region.columns = ['Sales_Mean', 'Sales_Median', 'Avg_Age', 'Order_Count']

# Normalize data for radar chart visualization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
metrics_normalized = pd.DataFrame(
    scaler.fit_transform(metrics_by_region),
    columns=metrics_by_region.columns,
    index=metrics_by_region.index
)

# Create a simple multi-metric comparison
x = range(len(metrics_by_region.columns))
width = 0.2
regions = metrics_normalized.index

for i, region in enumerate(regions):
    ax6.bar([j + i*width for j in x], metrics_normalized.loc[region], 
            width=width, label=region, alpha=0.8)

ax6.set_xlabel('Metrics')
ax6.set_ylabel('Normalized Values')
ax6.set_title('Multi-Metric Comparison by Region')
ax6.set_xticks([j + width*1.5 for j in x])
ax6.set_xticklabels(metrics_normalized.columns, rotation=45)
ax6.legend()

plt.tight_layout()
plt.show()

# Advanced visualization - Subplots với different chart types
print("\nAdvanced Multi-Chart Dashboard:")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Top-left: Time series với multiple lines
monthly_sales_by_region = df_dirty.groupby([df_dirty['order_date'].dt.month, 'region'])['sales_amount'].sum().unstack()
monthly_sales_by_region.plot(ax=ax1, marker='o')
ax1.set_title('Monthly Sales Trend by Region')
ax1.set_xlabel('Month')
ax1.legend(title='Region')

# Top-right: Box plot comparison
sns.boxplot(data=df_dirty, x='age_group', y='sales_amount', hue='product_category', ax=ax2)
ax2.set_title('Sales Distribution by Age Group and Category')
ax2.legend(bbox_to_anchor=(1.05, 1))

# Bottom-left: Correlation network (simplified)
corr_matrix = df_dirty[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', 
            vmin=-1, vmax=1, center=0, square=True, ax=ax3)
ax3.set_title('Correlation Matrix (Lower Triangle)')

# Bottom-right: Customer analysis
customer_metrics = df_dirty.groupby('customer_id').agg({
    'sales_amount': 'sum',
    'order_id': 'count'
}).rename(columns={'order_id': 'order_frequency'})

ax4.scatter(customer_metrics['order_frequency'], customer_metrics['sales_amount'], alpha=0.6)
ax4.set_xlabel('Order Frequency')
ax4.set_ylabel('Total Sales Amount')
ax4.set_title('Customer Value Analysis')

# Add trend line
z = np.polyfit(customer_metrics['order_frequency'], customer_metrics['sales_amount'], 1)
p = np.poly1d(z)
ax4.plot(customer_metrics['order_frequency'], p(customer_metrics['order_frequency']), "r--", alpha=0.8)

plt.tight_layout()
plt.show()
```

### Bài tập 4.4: Dashboard và Storytelling

```python
print("=== BÀI TẬP 4.4: DASHBOARD VÀ DATA STORYTELLING ===")
print("1. Tạo executive dashboard tổng hợp")
print("2. Before/After comparison visualization")
print("3. Performance metrics dashboard")
print("4. Interactive-style filtering simulation")
```

**Đáp án mẫu:**
```python
# 1. Executive Dashboard
def create_executive_dashboard(data):
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # KPI Cards (top row)
    # Total Sales
    ax1 = fig.add_subplot(gs[0, 0])
    total_sales = data['sales_amount'].sum()
    ax1.text(0.5, 0.5, f'${total_sales:,.0f}', ha='center', va='center', 
             fontsize=24, fontweight='bold', color='darkblue')
    ax1.text(0.5, 0.2, 'Total Sales', ha='center', va='center', fontsize=12)
    ax1.axis('off')
    ax1.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='darkblue', lw=2))
    
    # Average Order Value
    ax2 = fig.add_subplot(gs[0, 1])
    avg_order = data['sales_amount'].mean()
    ax2.text(0.5, 0.5, f'${avg_order:.0f}', ha='center', va='center', 
             fontsize=24, fontweight='bold', color='darkgreen')
    ax2.text(0.5, 0.2, 'Avg Order Value', ha='center', va='center', fontsize=12)
    ax2.axis('off')
    ax2.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='darkgreen', lw=2))
    
    # Total Customers
    ax3 = fig.add_subplot(gs[0, 2])
    total_customers = data['customer_id'].nunique()
    ax3.text(0.5, 0.5, f'{total_customers:,}', ha='center', va='center', 
             fontsize=24, fontweight='bold', color='darkorange')
    ax3.text(0.5, 0.2, 'Total Customers', ha='center', va='center', fontsize=12)
    ax3.axis('off')
    ax3.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='darkorange', lw=2))
    
    # Total Orders
    ax4 = fig.add_subplot(gs[0, 3])
    total_orders = len(data)
    ax4.text(0.5, 0.5, f'{total_orders:,}', ha='center', va='center', 
             fontsize=24, fontweight='bold', color='darkred')
    ax4.text(0.5, 0.2, 'Total Orders', ha='center', va='center', fontsize=12)
    ax4.axis('off')
    ax4.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='darkred', lw=2))
    
    # Sales trend (second row, span 2 columns)
    ax5 = fig.add_subplot(gs[1, :2])
    daily_sales = data.groupby(data['order_date'].dt.date)['sales_amount'].sum()
    daily_sales_ma = daily_sales.rolling(window=7).mean()
    ax5.plot(daily_sales.index, daily_sales.values, alpha=0.3, color='lightblue')
    ax5.plot(daily_sales_ma.index, daily_sales_ma.values, color='darkblue', linewidth=2)
    ax5.set_title('Daily Sales Trend (7-day Moving Average)', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Regional performance
    ax6 = fig.add_subplot(gs[1, 2:])
    region_performance = data.groupby('region')['sales_amount'].agg(['sum', 'count', 'mean'])
    region_performance['sum'].plot(kind='bar', ax=ax6, color='skyblue', edgecolor='navy')
    ax6.set_title('Sales Performance by Region', fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    
    # Category breakdown (third row)
    ax7 = fig.add_subplot(gs[2, :2])
    category_sales = data.groupby('product_category')['sales_amount'].sum().sort_values(ascending=True)
    category_sales.plot(kind='barh', ax=ax7, color='lightgreen', edgecolor='darkgreen')
    ax7.set_title('Sales by Product Category', fontweight='bold')
    
    # Age group analysis
    ax8 = fig.add_subplot(gs[2, 2:])
    age_analysis = data.groupby('age_group')['sales_amount'].agg(['mean', 'count'])
    ax8_twin = ax8.twinx()
    
    bars = ax8.bar(age_analysis.index, age_analysis['mean'], alpha=0.7, color='orange', label='Avg Sales')
    line = ax8_twin.plot(age_analysis.index, age_analysis['count'], color='red', marker='o', linewidth=2, label='Order Count')
    
    ax8.set_title('Sales and Order Count by Age Group', fontweight='bold')
    ax8.set_ylabel('Average Sales', color='orange')
    ax8_twin.set_ylabel('Order Count', color='red')
    ax8.legend(loc='upper left')
    ax8_twin.legend(loc='upper right')
    
    # Bottom insights section
    ax9 = fig.add_subplot(gs[3, :])
    insights_text = f"""
    KEY INSIGHTS:
    • Top performing region: {data.groupby('region')['sales_amount'].sum().idxmax()} (${data.groupby('region')['sales_amount'].sum().max():,.0f})
    • Most popular category: {data['product_category'].value_counts().index[0]} ({data['product_category'].value_counts().iloc[0]:,} orders)
    • Peak sales month: Month {data.groupby('month')['sales_amount'].sum().idxmax()} (${data.groupby('month')['sales_amount'].sum().max():,.0f})
    • Customer retention: {(data.groupby('customer_id').size() > 1).mean()*100:.1f}% of customers made multiple orders
    • Sales distribution: {((data['sales_amount'] > data['sales_amount'].mean()).mean()*100):.1f}% of orders above average value
    """
    ax9.text(0.02, 0.8, insights_text, fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    ax9.axis('off')
    
    plt.suptitle('EXECUTIVE SALES DASHBOARD', fontsize=20, fontweight='bold', y=0.98)
    return fig

# Create dashboard
dashboard = create_executive_dashboard(df_dirty)
plt.show()

# 2. Performance Comparison Analysis
print("\n2. Performance Analysis Dashboard:")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Performance metrics by region
ax1 = axes[0, 0]
region_metrics = df_dirty.groupby('region').agg({
    'sales_amount': ['mean', 'std'],
    'customer_age': 'mean',
    'order_id': 'count'
})
region_metrics.columns = ['Avg_Sales', 'Sales_StdDev', 'Avg_Age', 'Order_Count']

# Performance score calculation
region_metrics['Performance_Score'] = (
    (region_metrics['Avg_Sales'] - region_metrics['Avg_Sales'].min()) / 
    (region_metrics['Avg_Sales'].max() - region_metrics['Avg_Sales'].min()) * 50 +
    (region_metrics['Order_Count'] - region_metrics['Order_Count'].min()) / 
    (region_metrics['Order_Count'].max() - region_metrics['Order_Count'].min()) * 50
)

colors = ['red' if score < 50 else 'green' for score in region_metrics['Performance_Score']]
bars = ax1.bar(region_metrics.index, region_metrics['Performance_Score'], color=colors, alpha=0.7)
ax1.set_title('Regional Performance Score')
ax1.set_ylabel('Performance Score (0-100)')
ax1.axhline(y=50, color='black', linestyle='--', alpha=0.5)

# Add value labels on bars
for bar, score in zip(bars, region_metrics['Performance_Score']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{score:.1f}', ha='center', va='bottom')

# Customer value distribution
ax2 = axes[0, 1]
customer_value = df_dirty.groupby('customer_id')['sales_amount'].sum()
ax2.hist(customer_value, bins=30, alpha=0.7, color='skyblue', edgecolor='navy')
ax2.axvline(customer_value.mean(), color='red', linestyle='--', label=f'Mean: ${customer_value.mean():.0f}')
ax2.axvline(customer_value.quantile(0.8), color='green', linestyle='--', label=f'80th percentile: ${customer_value.quantile(0.8):.0f}')
ax2.set_title('Customer Lifetime Value Distribution')
ax2.set_xlabel('Total Sales per Customer')
ax2.legend()

# Monthly trends with forecasting
ax3 = axes[1, 0]
monthly_trend = df_dirty.groupby('month')['sales_amount'].agg(['sum', 'mean'])
ax3.plot(monthly_trend.index, monthly_trend['sum'], marker='o', linewidth=2, label='Total Sales')
ax3_twin = ax3.twinx()
ax3_twin.plot(monthly_trend.index, monthly_trend['mean'], marker='s', color='orange', linewidth=2, label='Average Sales')

# Simple linear trend
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(monthly_trend.index, monthly_trend['sum'])
trend_line = slope * monthly_trend.index + intercept
ax3.plot(monthly_trend.index, trend_line, '--', color='red', alpha=0.7, label=f'Trend (R²={r_value**2:.3f})')

ax3.set_title('Monthly Sales Trend Analysis')
ax3.set_xlabel('Month')
ax3.set_ylabel('Total Sales', color='blue')
ax3_twin.set_ylabel('Average Sales', color='orange')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')

# Category performance matrix
ax4 = axes[1, 1]
category_performance = df_dirty.groupby('product_category').agg({
    'sales_amount': ['mean', 'count'],
    'customer_id': 'nunique'
})
category_performance.columns = ['Avg_Sales', 'Order_Count', 'Unique_Customers']
category_performance['Revenue_per_Customer'] = category_performance['Avg_Sales'] * category_performance['Order_Count'] / category_performance['Unique_Customers']

scatter = ax4.scatter(category_performance['Order_Count'], 
                     category_performance['Avg_Sales'],
                     s=category_performance['Unique_Customers']*2,
                     alpha=0.7, c=range(len(category_performance)), cmap='viridis')

# Add labels for each point
for i, (idx, row) in enumerate(category_performance.iterrows()):
    ax4.annotate(idx, (row['Order_Count'], row['Avg_Sales']), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax4.set_xlabel('Order Count')
ax4.set_ylabel('Average Sales')
ax4.set_title('Category Performance Matrix\n(Size = Unique Customers)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 3. Advanced Storytelling Visualization
print("\n3. Data Storytelling: Customer Journey Analysis")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Customer acquisition over time
ax1 = axes[0, 0]
customer_first_order = df_dirty.groupby('customer_id')['order_date'].min().reset_index()
customer_first_order['first_order_month'] = customer_first_order['order_date'].dt.to_period('M')
monthly_acquisitions = customer_first_order['first_order_month'].value_counts().sort_index()

ax1.plot(monthly_acquisitions.index.astype(str), monthly_acquisitions.values, 
         marker='o', linewidth=2, markersize=8, color='darkblue')
ax1.fill_between(range(len(monthly_acquisitions)), monthly_acquisitions.values, alpha=0.3, color='lightblue')
ax1.set_title('Customer Acquisition Over Time', fontsize=14, fontweight='bold')
ax1.set_ylabel('New Customers')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# Customer lifecycle analysis
ax2 = axes[0, 1]
customer_lifecycle = df_dirty.groupby('customer_id').agg({
    'order_date': ['min', 'max', 'count'],
    'sales_amount': ['sum', 'mean']
}).round(2)
customer_lifecycle.columns = ['First_Order', 'Last_Order', 'Order_Count', 'Total_Spent', 'Avg_Order']
customer_lifecycle['Days_Active'] = (customer_lifecycle['Last_Order'] - customer_lifecycle['First_Order']).dt.days

# Customer segments based on behavior
customer_lifecycle['Segment'] = pd.cut(customer_lifecycle['Total_Spent'], 
                                      bins=[0, 50, 100, 200, float('inf')], 
                                      labels=['Low Value', 'Medium Value', 'High Value', 'VIP'])

segment_counts = customer_lifecycle['Segment'].value_counts()
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
wedges, texts, autotexts = ax2.pie(segment_counts.values, labels=segment_counts.index, 
                                   autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title('Customer Segmentation by Total Spending', fontsize=14, fontweight='bold')

# Purchase behavior patterns
ax3 = axes[1, 0]
# Create purchase frequency distribution
purchase_frequency = customer_lifecycle['Order_Count'].value_counts().sort_index()
ax3.bar(purchase_frequency.index, purchase_frequency.values, 
        color='lightcoral', edgecolor='darkred', alpha=0.8)
ax3.set_title('Purchase Frequency Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Number of Orders per Customer')
ax3.set_ylabel('Number of Customers')

# Add statistics
mean_orders = customer_lifecycle['Order_Count'].mean()
ax3.axvline(mean_orders, color='darkred', linestyle='--', 
           label=f'Mean: {mean_orders:.1f} orders')
ax3.legend()

# Customer value vs engagement
ax4 = axes[1, 1]
scatter = ax4.scatter(customer_lifecycle['Order_Count'], 
                     customer_lifecycle['Total_Spent'],
                     c=customer_lifecycle['Days_Active'], 
                     cmap='plasma', alpha=0.7, s=60)
ax4.set_xlabel('Order Frequency')
ax4.set_ylabel('Total Spending')
ax4.set_title('Customer Value vs Engagement', fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Days Active')

# Add trend line
z = np.polyfit(customer_lifecycle['Order_Count'], customer_lifecycle['Total_Spent'], 1)
p = np.poly1d(z)
ax4.plot(customer_lifecycle['Order_Count'], 
         p(customer_lifecycle['Order_Count']), 
         "r--", alpha=0.8, linewidth=2)

plt.tight_layout()
plt.show()

# 4. Interactive-style Analysis (Simulated filtering)
print("\n4. Simulated Interactive Analysis:")

def analyze_subset(data, filters):
    """Simulate interactive filtering"""
    filtered_data = data.copy()
    
    for column, values in filters.items():
        if isinstance(values, list):
            filtered_data = filtered_data[filtered_data[column].isin(values)]
        else:
            filtered_data = filtered_data[filtered_data[column] == values]
    
    return filtered_data

# Define different filter scenarios
scenarios = {
    'High-Value Customers': {'customer_id': customer_lifecycle[customer_lifecycle['Total_Spent'] > 150].index.tolist()[:50]},
    'Electronics Only': {'product_category': 'Electronics'},
    'Q1 Orders': {'month': [1, 2, 3]},
    'Young Adults': {'customer_age': list(range(18, 35))}
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, (scenario_name, filters) in enumerate(scenarios.items()):
    filtered_df = analyze_subset(df_dirty, filters)
    
    if len(filtered_df) > 0:
        # Create summary visualization for each scenario
        if scenario_name == 'High-Value Customers':
            # Sales distribution
            axes[i].hist(filtered_df['sales_amount'], bins=20, alpha=0.7, 
                        color='gold', edgecolor='darkorange')
            axes[i].set_title(f'{scenario_name}\nSales Distribution (n={len(filtered_df)})')
            
        elif scenario_name == 'Electronics Only':
            # Regional breakdown
            region_sales = filtered_df.groupby('region')['sales_amount'].sum()
            region_sales.plot(kind='bar', ax=axes[i], color='skyblue', edgecolor='navy')
            axes[i].set_title(f'{scenario_name}\nSales by Region (n={len(filtered_df)})')
            axes[i].tick_params(axis='x', rotation=45)
            
        elif scenario_name == 'Q1 Orders':
            # Daily trend
            daily_sales = filtered_df.groupby(filtered_df['order_date'].dt.date)['sales_amount'].sum()
            axes[i].plot(daily_sales.index, daily_sales.values, 
                        color='green', linewidth=2, marker='o', markersize=4)
            axes[i].set_title(f'{scenario_name}\nDaily Sales Trend (n={len(filtered_df)})')
            axes[i].tick_params(axis='x', rotation=45)
            
        else:  # Young Adults
            # Category preferences
            category_pref = filtered_df['product_category'].value_counts()
            axes[i].pie(category_pref.values, labels=category_pref.index, autopct='%1.1f%%')
            axes[i].set_title(f'{scenario_name}\nCategory Preferences (n={len(filtered_df)})')
        
        # Add summary statistics
        avg_sales = filtered_df['sales_amount'].mean()
        total_sales = filtered_df['sales_amount'].sum()
        axes[i].text(0.02, 0.98, f'Avg: ${avg_sales:.0f}\nTotal: ${total_sales:,.0f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.suptitle('Scenario-Based Analysis Dashboard', fontsize=16, fontweight='bold', y=1.02)
plt.show()

print("\n=== SUMMARY: KEY VISUALIZATION INSIGHTS ===")
print(f"1. Dataset Overview: {len(df_dirty):,} orders from {df_dirty['customer_id'].nunique():,} customers")
print(f"2. Revenue Distribution: Mean ${df_dirty['sales_amount'].mean():.0f}, Median ${df_dirty['sales_amount'].median():.0f}")
print(f"3. Regional Leaders: {df_dirty.groupby('region')['sales_amount'].sum().idxmax()} region leads with ${df_dirty.groupby('region')['sales_amount'].sum().max():,.0f}")
print(f"4. Customer Loyalty: {(df_dirty.groupby('customer_id').size() > 1).mean()*100:.1f}% of customers are repeat buyers")
print(f"5. Category Performance: {df_dirty['product_category'].value_counts().index[0]} has highest order volume ({df_dirty['product_category'].value_counts().iloc[0]:,} orders)")
print(f"6. Seasonal Trends: Month {df_dirty.groupby('month')['sales_amount'].sum().idxmax()} shows peak sales (${df_dirty.groupby('month')['sales_amount'].sum().max():,.0f})")
```

---

## PHẦN 5: BÀI TẬP TỔNG HỢP VÀ DỰ ÁN THỰC HÀNH

### Dự án thực hành: Phân tích dữ liệu E-commerce

**Mô tả dự án**: Học viên sẽ áp dụng tất cả các kỹ năng đã học để phân tích một bộ dữ liệu e-commerce hoàn chỉnh và tạo báo cáo insights.

### Yêu cầu dự án:

```python
print("=== DỰ ÁN THỰC HÀNH: PHÂN TÍCH E-COMMERCE ===")
print("Học viên cần hoàn thành các nhiệm vụ sau:")
print("1. Data Exploration & Cleaning (25%)")
print("2. Statistical Analysis (25%)")
print("3. Pivot Analysis & Comparisons (25%)")
print("4. Visualization Dashboard (25%)")
print("\nThời gian: 4-6 tiếng")
print("Kết quả: Jupyter Notebook + Presentation slides")
```

### Rubric đánh giá:

| Tiêu chí | Xuất sắc (4) | Tốt (3) | Đạt (2) | Cần cải thiện (1) |
|----------|-------------|---------|---------|-------------------|
| Data Exploration | Phân tích toàn diện, xử lý outliers, missing values chuyên nghiệp | Khám phá tốt, xử lý cơ bản các vấn đề | Khám phá cơ bản, một số vấn đề chưa được xử lý | Thiếu nhiều bước quan trọng |
| Statistical Analysis | Sử dụng đúng các test, giải thích ý nghĩa thống kê | Áp dụng đúng phương pháp, giải thích cơ bản | Sử dụng một số phương pháp phù hợp | Áp dụng sai hoặc thiếu phương pháp |
| Pivot & Comparisons | Insights sâu sắc, so sánh đa chiều | Phân tích tốt, một số insights có giá trị | Phân tích cơ bản, insights đơn giản | Thiếu phân tích so sánh |
| Visualization | Dashboard chuyên nghiệp, story-telling tốt | Biểu đồ đẹp, truyền tải thông tin rõ | Biểu đồ cơ bản, dễ hiểu | Biểu đồ kém chất lượng |

---

## TÀI LIỆU THAM KHẢO VÀ RESOURCES

### Thư viện chính:
```python
# Essential imports cho tất cả bài tập
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Cấu hình hiển thị
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

### Cheat Sheet - Các hàm quan trọng:

**Data Exploration:**
- `df.info()`, `df.describe()`, `df.head()`, `df.tail()`
- `df.isnull().sum()`, `df.value_counts()`, `df.nunique()`
- `df.groupby().agg()`, `pd.pivot_table()`

**Statistical Analysis:**
- `df.corr()`, `df.cov()`, `df.skew()`, `df.kurtosis()`
- `stats.ttest_ind()`, `stats.f_oneway()`, `stats.chi2_contingency()`
- `stats.pearsonr()`, `stats.spearmanr()`

**Visualization:**
- **Matplotlib**: `plt.plot()`, `plt.bar()`, `plt.hist()`, `plt.scatter()`
- **Seaborn**: `sns.boxplot()`, `sns.heatmap()`, `sns.pairplot()`, `sns.violinplot()`

### Best Practices:
1. **Always start with data exploration** - Hiểu dữ liệu trước khi phân tích
2. **Document your analysis** - Comment code và giải thích insights
3. **Validate assumptions** - Kiểm tra assumptions của statistical tests
4. **Tell a story with data** - Visualization phải có mục đích rõ ràng
5. **Iterate and refine** - Cải thiện analysis dựa trên findings

### Recommended Reading:
- "Python for Data Analysis" by Wes McKinney
- "Fundamentals of Data Visualization" by Claus O. Wilke
- Pandas Documentation: https://pandas.pydata.org/docs/
- Seaborn Tutorial: https://seaborn.pydata.org/tutorial.html# Bài Tập Ứng Dụng: Data Exploration and Visualization với Python

