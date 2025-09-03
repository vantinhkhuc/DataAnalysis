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
