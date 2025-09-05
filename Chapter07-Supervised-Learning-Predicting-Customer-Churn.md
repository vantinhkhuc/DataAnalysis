# Chapter 07 Supervised Learning: Predicting Customer Churn

**Môn học:** Phân tích dữ liệu  
**Ngôn ngữ:** Python  
**Thời lượng:** 4-5 tiếng  
**Prerequisite:** Lập trình Python căn bản

## Mục tiêu học tập

Sau khi hoàn thành bài học này, học viên sẽ có khả năng:

1. **Hiểu và áp dụng Supervised Learning** cho bài toán classification
2. **Thực hiện classification tasks** sử dụng logistic regression
3. **Triển khai pipeline OSEMN** (Obtain, Scrub, Explore, Model, iNterpret) một cách hoàn chỉnh
4. **Phân tích mối quan hệ** giữa biến target và explanatory variables thông qua data exploration
5. **Lựa chọn features hiệu quả** để xây dựng predictive models
6. **Xây dựng và đánh giá churn model** sử dụng logistic regression làm baseline
7. **Giải thích business insights** từ kết quả mô hình

---

## Phần 1: Giới thiệu về Supervised Learning và Customer Churn

### 1.1 Supervised Learning là gì?

**Supervised Learning** là một nhánh của Machine Learning trong đó mô hình được huấn luyện trên dữ liệu có nhãn (labeled data) để dự đoán kết quả cho dữ liệu mới.

**Đặc điểm:**
- Có target variable (biến phụ thuộc) rõ ràng
- Chia thành 2 loại chính: Classification và Regression
- Đánh giá được performance thông qua so sánh prediction vs actual

**Classification vs Regression:**
- **Classification**: Dự đoán category/class (Yes/No, High/Medium/Low)
- **Regression**: Dự đoán continuous value (giá nhà, doanh số)

### 1.2 Customer Churn Problem

**Customer Churn** (Khách hàng rời bỏ) là hiện tượng khách hàng ngừng sử dụng sản phẩm/dịch vụ của công ty.

**Tại sao quan trọng:**
- Chi phí giữ chân khách hàng cũ < Chi phí tìm khách hàng mới (5-25 lần)
- Tác động trực tiếp đến revenue và growth
- Có thể predict và prevent được

**Business Impact:**
- Định identify khách hàng có risk cao
- Thiết kế retention campaigns
- Optimize customer lifetime value

### 1.3 Pipeline OSEMN Overview

**OSEMN** là framework được sử dụng rộng rãi trong Data Science:

1. **O**btain: Thu thập dữ liệu
2. **S**crub: Làm sạch dữ liệu  
3. **E**xplore: Khám phá dữ liệu (EDA)
4. **M**odel: Xây dựng mô hình
5. **iN**terpret: Giải thích kết quả

---

## Phần 2: OBTAIN - Thu thập và Load dữ liệu

### 2.1 Setup Environment

```python
# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           accuracy_score, precision_score, recall_score, f1_score)

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Environment setup completed!")
```

### 2.2 Data Generation và Understanding

```python
# Tạo realistic customer churn dataset
np.random.seed(42)
n_customers = 5000

# Customer demographics
customer_data = {
    'CustomerID': range(1, n_customers + 1),
    'Gender': np.random.choice(['Male', 'Female'], n_customers),
    'Age': np.random.normal(40, 12, n_customers).clip(18, 80).astype(int),
    'SeniorCitizen': np.random.choice([0, 1], n_customers, p=[0.84, 0.16]),
    
    # Account information  
    'Tenure': np.random.exponential(scale=24, size=n_customers).clip(1, 72).astype(int),
    'MonthlyCharges': np.random.normal(65, 20, n_customers).clip(18.25, 118.75),
    
    # Services
    'PhoneService': np.random.choice(['Yes', 'No'], n_customers, p=[0.91, 0.09]),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_customers, p=[0.42, 0.49, 0.09]),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.34, 0.44, 0.22]),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.29, 0.49, 0.22]),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.29, 0.49, 0.22]),
    
    # Contract details
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers, p=[0.55, 0.21, 0.24]),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n_customers, p=[0.59, 0.41]),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 
                                    n_customers, p=[0.34, 0.19, 0.22, 0.25])
}

# Create DataFrame
df = pd.DataFrame(customer_data)

# Calculate TotalCharges based on Tenure and MonthlyCharges
df['TotalCharges'] = df['Tenure'] * df['MonthlyCharges'] + np.random.normal(0, 100, n_customers)
df['TotalCharges'] = df['TotalCharges'].clip(lower=df['MonthlyCharges'])

print("Dataset created successfully!")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
```

### 2.3 Tạo Target Variable với Business Logic

```python
def create_churn_target(df):
    """
    Tạo target variable 'Churn' dựa trên business logic thực tế
    """
    # Initialize churn probability
    churn_prob = np.zeros(len(df))
    
    # Age factor: Younger customers more likely to switch
    age_factor = np.where(df['Age'] < 30, 0.15, 
                 np.where(df['Age'] > 60, 0.10, 0.05))
    
    # Tenure factor: New customers more likely to churn
    tenure_factor = np.where(df['Tenure'] <= 6, 0.40,
                    np.where(df['Tenure'] <= 12, 0.25,
                    np.where(df['Tenure'] <= 24, 0.15, 0.05)))
    
    # Contract factor: Month-to-month highest risk
    contract_factor = np.where(df['Contract'] == 'Month-to-month', 0.35,
                      np.where(df['Contract'] == 'One year', 0.15, 0.05))
    
    # Payment method factor: Electronic check higher risk  
    payment_factor = np.where(df['PaymentMethod'] == 'Electronic check', 0.15, 0.05)
    
    # Service factor: No internet service lower risk
    service_factor = np.where(df['InternetService'] == 'No', 0.02,
                     np.where(df['InternetService'] == 'Fiber optic', 0.12, 0.08))
    
    # Monthly charges factor: Higher charges = higher risk
    charges_factor = np.where(df['MonthlyCharges'] > 80, 0.20,
                     np.where(df['MonthlyCharges'] > 60, 0.10, 0.05))
    
    # Tech support factor: No tech support = higher risk
    tech_factor = np.where(df['TechSupport'] == 'No', 0.10, 0.02)
    
    # Combine all factors
    churn_prob = (age_factor + tenure_factor + contract_factor + 
                  payment_factor + service_factor + charges_factor + tech_factor)
    
    # Add some randomness and ensure probability bounds
    churn_prob = np.clip(churn_prob + np.random.normal(0, 0.05, len(df)), 0, 1)
    
    # Generate actual churn based on probability
    df['Churn'] = np.random.binomial(1, churn_prob)
    
    return df

# Apply churn generation
df = create_churn_target(df)

# Display basic statistics
print("Churn Distribution:")
print(df['Churn'].value_counts())
print(f"Churn Rate: {df['Churn'].mean():.2%}")
```

---

## Phần 3: SCRUB - Data Cleaning và Preprocessing

### 3.1 Data Quality Assessment

```python
def assess_data_quality(df):
    """
    Đánh giá chất lượng dữ liệu toàn diện
    """
    print("="*50)
    print("DATA QUALITY ASSESSMENT")
    print("="*50)
    
    # Basic info
    print(f"\nDataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    if missing_data.sum() > 0:
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        }).sort_values('Missing Count', ascending=False)
        print("\nMISSING VALUES:")
        print(missing_df[missing_df['Missing Count'] > 0])
    else:
        print("\n✓ No missing values found")
    
    # Data types
    print("\nDATA TYPES:")
    print(df.dtypes.value_counts())
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    # Numerical columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNUMERICAL COLUMNS STATISTICS:")
        print(df[numeric_cols].describe().round(2))
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"\nCATEGORICAL COLUMNS:")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            if unique_count <= 10:  # Show values if not too many
                print(f"    Values: {list(df[col].unique())}")

# Assess our dataset
assess_data_quality(df)
```

### 3.2 Data Cleaning Steps

```python
def clean_data(df):
    """
    Thực hiện các bước làm sạch dữ liệu
    """
    df_clean = df.copy()
    
    print("Starting data cleaning process...")
    
    # 1. Handle data type issues
    # Convert TotalCharges to numeric (if it contains spaces/empty strings)
    if df_clean['TotalCharges'].dtype == 'object':
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # 2. Handle missing values
    # Fill missing TotalCharges with Tenure * MonthlyCharges
    missing_total_charges = df_clean['TotalCharges'].isnull()
    if missing_total_charges.sum() > 0:
        df_clean.loc[missing_total_charges, 'TotalCharges'] = (
            df_clean.loc[missing_total_charges, 'Tenure'] * 
            df_clean.loc[missing_total_charges, 'MonthlyCharges']
        )
        print(f"✓ Filled {missing_total_charges.sum()} missing TotalCharges values")
    
    # 3. Handle outliers
    def remove_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
        return data[~outliers_mask], outliers_mask.sum()
    
    # Remove outliers from numerical columns
    original_size = len(df_clean)
    numerical_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
    
    for col in numerical_cols:
        df_clean, outliers_removed = remove_outliers_iqr(df_clean, col)
        if outliers_removed > 0:
            print(f"✓ Removed {outliers_removed} outliers from {col}")
    
    print(f"✓ Dataset size after cleaning: {len(df_clean)} (removed {original_size - len(df_clean)} rows)")
    
    # 4. Feature consistency checks
    # Ensure MultipleLines consistency with PhoneService
    inconsistent_lines = ((df_clean['PhoneService'] == 'No') & 
                         (df_clean['MultipleLines'] != 'No phone service'))
    if inconsistent_lines.sum() > 0:
        df_clean.loc[inconsistent_lines, 'MultipleLines'] = 'No phone service'
        print(f"✓ Fixed {inconsistent_lines.sum()} inconsistent MultipleLines values")
    
    # Ensure internet-dependent services consistency
    internet_services = ['OnlineSecurity', 'TechSupport']
    no_internet_mask = df_clean['InternetService'] == 'No'
    
    for service in internet_services:
        inconsistent_mask = no_internet_mask & (df_clean[service] != 'No internet service')
        if inconsistent_mask.sum() > 0:
            df_clean.loc[inconsistent_mask, service] = 'No internet service'
            print(f"✓ Fixed {inconsistent_mask.sum()} inconsistent {service} values")
    
    return df_clean

# Clean the data
df_clean = clean_data(df)
print("\nData cleaning completed successfully!")
```

---

## Phần 4: EXPLORE - Data Exploration và Analysis

### 4.1 Univariate Analysis

```python
def perform_univariate_analysis(df):
    """
    Phân tích từng biến riêng lẻ
    """
    print("="*50)
    print("UNIVARIATE ANALYSIS")
    print("="*50)
    
    # Target variable analysis
    print("\n1. TARGET VARIABLE ANALYSIS:")
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True)
    
    print("Churn Distribution:")
    for i, (count, pct) in enumerate(zip(churn_counts, churn_pct)):
        label = "No Churn" if i == 0 else "Churn"
        print(f"  {label}: {count:,} ({pct:.2%})")
    
    # Visualization setup
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Target distribution
    plt.subplot(3, 4, 1)
    churn_counts.plot(kind='pie', autopct='%1.1f%%', labels=['No Churn', 'Churn'])
    plt.title('Churn Distribution')
    plt.ylabel('')
    
    # 2. Numerical variables
    numerical_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
    
    for i, col in enumerate(numerical_cols):
        plt.subplot(3, 4, i+2)
        df[col].hist(bins=30, alpha=0.7, edgecolor='black')
        plt.title(f'{col} Distribution')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        
        # Add statistics
        mean_val = df[col].mean()
        median_val = df[col].median()
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
        plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.1f}')
        plt.legend()
    
    # 3. Categorical variables
    categorical_cols = ['Gender', 'SeniorCitizen', 'Contract', 'InternetService', 'PaymentMethod', 'PaperlessBilling']
    
    for i, col in enumerate(categorical_cols):
        plt.subplot(3, 4, i+6)
        value_counts = df[col].value_counts()
        value_counts.plot(kind='bar')
        plt.title(f'{col} Distribution')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        
        # Add percentages
        total = len(df)
        for j, v in enumerate(value_counts.values):
            plt.text(j, v + total*0.01, f'{v/total:.1%}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Print numerical statistics
    print("\n2. NUMERICAL VARIABLES STATISTICS:")
    print(df[numerical_cols].describe().round(2))

# Perform univariate analysis
perform_univariate_analysis(df_clean)
```

### 4.2 Bivariate Analysis - Mối quan hệ với Target Variable

```python
def analyze_target_relationships(df):
    """
    Phân tích mối quan hệ giữa features và target variable
    Đây là phần quan trọng để hiểu data và select features
    """
    print("="*50)
    print("BIVARIATE ANALYSIS - RELATIONSHIP WITH TARGET")
    print("="*50)
    
    # Numerical variables vs Churn
    numerical_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
    
    print("\n1. NUMERICAL VARIABLES vs CHURN:")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(numerical_cols):
        # Box plot
        df.boxplot(column=col, by='Churn', ax=axes[i])
        axes[i].set_title(f'{col} by Churn Status')
        
        # Statistical test
        churn_group = df[df['Churn'] == 1][col]
        no_churn_group = df[df['Churn'] == 0][col]
        
        # T-test
        t_stat, p_value = stats.ttest_ind(churn_group, no_churn_group)
        
        # Statistics summary
        churn_mean = churn_group.mean()
        no_churn_mean = no_churn_group.mean()
        
        print(f"\n{col}:")
        print(f"  No Churn Mean: {no_churn_mean:.2f}")
        print(f"  Churn Mean: {churn_mean:.2f}")
        print(f"  Difference: {churn_mean - no_churn_mean:.2f}")
        print(f"  T-test p-value: {p_value:.6f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    plt.suptitle('Numerical Variables vs Churn', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Categorical variables vs Churn
    print("\n2. CATEGORICAL VARIABLES vs CHURN:")
    categorical_cols = ['Gender', 'SeniorCitizen', 'Contract', 'InternetService', 
                       'PaymentMethod', 'PaperlessBilling', 'PhoneService']
    
    # Calculate churn rates for each category
    churn_analysis = {}
    
    for col in categorical_cols:
        churn_rate = df.groupby(col)['Churn'].agg(['count', 'sum', 'mean']).round(3)
        churn_rate.columns = ['Total_Customers', 'Churned_Customers', 'Churn_Rate']
        churn_analysis[col] = churn_rate
        
        print(f"\n{col}:")
        print(churn_rate)
        
        # Chi-square test
        contingency_table = pd.crosstab(df[col], df['Churn'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"  Chi-square p-value: {p_value:.6f}")
        print(f"  Significant association: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Visualization of categorical variables
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.ravel()
    
    for i, col in enumerate(categorical_cols):
        if i < len(axes):
            # Churn rate by category
            churn_rate = df.groupby(col)['Churn'].mean()
            churn_rate.plot(kind='bar', ax=axes[i], color='skyblue', edgecolor='black')
            axes[i].set_title(f'Churn Rate by {col}')
            axes[i].set_ylabel('Churn Rate')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add percentage labels on bars
            for j, v in enumerate(churn_rate.values):
                axes[i].text(j, v + 0.01, f'{v:.2%}', ha='center')
    
    # Remove empty subplots
    for i in range(len(categorical_cols), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()
    
    return churn_analysis

# Analyze relationships with target
churn_analysis_results = analyze_target_relationships(df_clean)
```

### 4.3 Feature Correlation Analysis

```python
def analyze_feature_correlations(df):
    """
    Phân tích correlation giữa các features
    """
    print("="*50)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*50)
    
    # Encode categorical variables for correlation analysis
    df_encoded = df.copy()
    
    # Label encoding for categorical variables
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df_encoded[col] = le.fit_transform(df[col])
    
    # Calculate correlation matrix
    correlation_matrix = df_encoded.corr()
    
    # Correlation with target variable
    target_corr = correlation_matrix['Churn'].abs().sort_values(ascending=False)
    print("CORRELATION WITH TARGET VARIABLE (absolute values):")
    print(target_corr.round(3))
    
    # Visualization
    plt.figure(figsize=(12, 10))
    
    # Full correlation matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                fmt='.2f', square=True)
    plt.title('Full Correlation Matrix')
    
    # Target correlations
    plt.subplot(1, 2, 2)
    target_corr_df = target_corr.drop('Churn').to_frame()
    target_corr_df.columns = ['Correlation']
    
    # Color mapping for positive/negative correlations
    colors = ['red' if x < 0 else 'blue' for x in correlation_matrix['Churn'].drop('Churn')]
    
    plt.barh(range(len(target_corr_df)), target_corr_df['Correlation'], color=colors, alpha=0.7)
    plt.yticks(range(len(target_corr_df)), target_corr_df.index)
    plt.xlabel('Absolute Correlation with Churn')
    plt.title('Feature Importance by Correlation')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    # High correlation pairs (potential multicollinearity)
    print("\nHIGH CORRELATION PAIRS (|correlation| > 0.7):")
    high_corr_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                high_corr_pairs.append({
                    'Feature1': correlation_matrix.columns[i],
                    'Feature2': correlation_matrix.columns[j],
                    'Correlation': corr_value
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        print(high_corr_df.round(3))
    else:
        print("No high correlation pairs found.")
    
    return correlation_matrix, target_corr

# Analyze correlations
correlation_matrix, target_correlations = analyze_feature_correlations(df_clean)
```

---

## Phần 5: Feature Selection và Engineering

### 5.1 Feature Selection dựa trên EDA Results

```python
def select_features_based_on_analysis(df, target_correlations, churn_analysis_results):
    """
    Lựa chọn features dựa trên kết quả phân tích EDA
    """
    print("="*50)
    print("FEATURE SELECTION BASED ON EDA")
    print("="*50)
    
    # 1. Features with high correlation with target
    high_corr_features = target_correlations[target_correlations.abs() > 0.1].index.tolist()
    if 'Churn' in high_corr_features:
        high_corr_features.remove('Churn')
    
    print("1. HIGH CORRELATION FEATURES (|correlation| > 0.1):")
    for feature in high_corr_features:
        corr_value = target_correlations[feature]
        print(f"  {feature}: {corr_value:.3f}")
    
    # 2. Categorical features with significant churn rate differences
    significant_categorical = []
    
    print("\n2. CATEGORICAL FEATURES WITH SIGNIFICANT CHURN DIFFERENCES:")
    for col, analysis in churn_analysis_results.items():
        churn_rates = analysis['Churn_Rate'].values
        if len(churn_rates) > 1:
            # Check if there's substantial difference in churn rates
            churn_range = churn_rates.max() - churn_rates.min()
            if churn_range > 0.1:  # 10% difference threshold
                significant_categorical.append(col)
                print(f"  {col}: Range = {churn_range:.3f} (Max: {churn_rates.max():.3f}, Min: {churn_rates.min():.3f})")
    
    # 3. Business importance (domain knowledge)
    business_important = ['Tenure', 'Contract', 'MonthlyCharges', 'PaymentMethod']
    
    print("\n3. BUSINESS IMPORTANT FEATURES:")
    for feature in business_important:
        print(f"  {feature}")
    
    # Combine all selected features
    selected_features = list(set(high_corr_features + significant_categorical + business_important))
    selected_features = [f for f in selected_features if f in df.columns and f != 'Churn']
    
    print(f"\n4. FINAL SELECTED FEATURES ({len(selected_features)}):")
    for i, feature in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feature}")
    
    return selected_features

# Select features
selected_features = select_features_based_on_analysis(df_clean, target_correlations, churn_analysis_results)
```

### 5.2 Feature Engineering

```python
def engineer_features(df, selected_features):
    """
    Tạo các features mới từ dữ liệu hiện có
    """
    print("="*50)
    print("FEATURE ENGINEERING")
    print("="*50)
    
    df_eng = df.copy()
    
    # 1. Tenure-based features
    print("1. TENURE-BASED FEATURES:")
    
    # Tenure groups
    df_eng['TenureGroup'] = pd.cut(df_eng['Tenure'], 
                                  bins=[0, 12, 24, 48, 100], 
                                  labels=['0-12', '12-24', '24-48', '48+'])
    print("  ✓ TenureGroup created")
    
    # New customer indicator
    df_eng['IsNewCustomer'] = (df_eng['Tenure'] <= 6).astype(int)
    print("  ✓ IsNewCustomer created")
    
    # 2. Charges-based features
    print("\n2. CHARGES-BASED FEATURES:")
    
    # Average monthly spending
    df_eng['AvgMonthlySpending'] = df_eng['TotalCharges'] / (df_eng['Tenure'] + 1)
    print("  ✓ AvgMonthlySpending created")
    
    # High value customer
    monthly_charges_75th = df_eng['MonthlyCharges'].quantile(0.75)
    df_eng['IsHighValueCustomer'] = (df_eng['MonthlyCharges'] > monthly_charges_75th).astype(int)
    print(f"  ✓ IsHighValueCustomer created (threshold: ${monthly_charges_75th:.2f})")
    
    # Charges ratio
    df_eng['ChargesRatio'] = df_eng['MonthlyCharges'] / (df_eng['AvgMonthlySpending'] + 1)
    print("  ✓ ChargesRatio created")
    
    # 3. Contract-based features
    print("\n3. CONTRACT-BASED FEATURES:")
    
    # Contract risk score
    contract_risk = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
    df_eng['ContractRisk'] = df_eng['Contract'].map(contract_risk)
    print("  ✓ ContractRisk created")
    
    # 4. Service-based features
    print("\n4. SERVICE-BASED FEATURES:")
    
    # Count of additional services
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'TechSupport']
    df_eng['ServiceCount'] = 0
    
    for col in service_cols:
        if col in df_eng.columns:
            df_eng['ServiceCount'] += (df_eng[col] == 'Yes').astype(int)
    
    print("  ✓ ServiceCount created")
    
    # Internet service score
    internet_score = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
    df_eng['InternetServiceScore'] = df_eng['InternetService'].map(internet_score)
    print("  ✓ InternetServiceScore created")
    
    # 5. Payment-based features
    print("\n5. PAYMENT-BASED FEATURES:")
    
    # Payment risk score (based on EDA results)
    payment_risk = {
        'Electronic check': 3,
        'Mailed check': 2,
        'Bank transfer (automatic)': 1,
        'Credit card (automatic)': 1
    }
    df_eng['PaymentRisk'] = df_eng['PaymentMethod'].map(payment_risk)
    print("  ✓ PaymentRisk created")
    
    # 6. Demographic features
    print("\n6. DEMOGRAPHIC FEATURES:")
    
    # Age groups
    df_eng['AgeGroup'] = pd.cut(df_eng['Age'], 
                               bins=[0, 30, 50, 65, 100], 
                               labels=['18-30', '30-50', '50-65', '65+'])
    print("  ✓ AgeGroup created")
    
    # Senior citizen flag (if not already binary)
    if df_eng['SeniorCitizen'].dtype == 'object':
        df_eng['SeniorCitizen'] = (df_eng['SeniorCitizen'] == 'Yes').astype(int)
    
    # List all engineered features
    engineered_features = [
        'TenureGroup', 'IsNewCustomer', 'AvgMonthlySpending', 'IsHighValueCustomer',
        'ChargesRatio', 'ContractRisk', 'ServiceCount', 'InternetServiceScore',
        'PaymentRisk', 'AgeGroup'
    ]
    
    print(f"\n✓ Total engineered features: {len(engineered_features)}")
    
    # Update selected features with engineered ones
    updated_features = selected_features + engineered_features
    updated_features = [f for f in updated_features if f in df_eng.columns and f != 'Churn']
    
    return df_eng, updated_features

# Engineer features
df_engineered, final_features = engineer_features(df_clean, selected_features)
```

---

## Phần 6: MODEL - Xây dựng Logistic Regression Model

### 6.1 Data Preparation cho Modeling

```python
def prepare_modeling_data(df, features):
    """
    Chuẩn bị dữ liệu cho quá trình modeling
    """
    print("="*50)
    print("DATA PREPARATION FOR MODELING")
    print("="*50)
    
    # Create feature matrix
    df_model = df.copy()
    
    # Separate numerical and categorical features
    numerical_features = []
    categorical_features = []
    
    for feature in features:
        if df_model[feature].dtype in ['int64', 'float64']:
            numerical_features.append(feature)
        else:
            categorical_features.append(feature)
    
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Create feature matrix X
    X = pd.DataFrame()
    
    # Add numerical features
    for feature in numerical_features:
        X[feature] = df_model[feature]
    
    # One-hot encode categorical features
    for feature in categorical_features:
        # Get dummies
        dummies = pd.get_dummies(df_model[feature], prefix=feature, drop_first=True)
        X = pd.concat([X, dummies], axis=1)
    
    # Target variable
    y = df_model['Churn']
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    print(f"Final features: {list(X.columns)}")
    
    # Check for any remaining missing values
    if X.isnull().sum().sum() > 0:
        print("\nWarning: Missing values detected!")
        print(X.isnull().sum()[X.isnull().sum() > 0])
        X = X.fillna(X.mean())  # Simple imputation
        print("Missing values filled with mean.")
    
    return X, y, numerical_features, categorical_features

# Prepare data for modeling
X, y, num_features, cat_features = prepare_modeling_data(df_engineered, final_features)
```

### 6.2 Train-Test Split và Data Scaling

```python
def split_and_scale_data(X, y):
    """
    Chia dữ liệu và chuẩn hóa
    """
    print("="*50)
    print("TRAIN-TEST SPLIT AND SCALING")
    print("="*50)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Check target distribution
    print(f"\nTraining set churn rate: {y_train.mean():.3f}")
    print(f"Test set churn rate: {y_test.mean():.3f}")
    
    # Scale the features
    scaler = StandardScaler()
    
    # Fit on training data and transform both sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    print("✓ Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Split and scale data
X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
```

### 6.3 Baseline Logistic Regression Model

```python
def train_baseline_logistic_regression(X_train, y_train):
    """
    Huấn luyện baseline Logistic Regression model
    """
    print("="*50)
    print("BASELINE LOGISTIC REGRESSION MODEL")
    print("="*50)
    
    # Initialize the model
    lr_baseline = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='liblinear'  # Good for small datasets
    )
    
    # Train the model
    print("Training logistic regression model...")
    lr_baseline.fit(X_train, y_train)
    
    # Model parameters
    print(f"✓ Model trained successfully!")
    print(f"Number of features: {len(lr_baseline.coef_[0])}")
    print(f"Intercept: {lr_baseline.intercept_[0]:.4f}")
    
    # Training accuracy
    train_accuracy = lr_baseline.score(X_train, y_train)
    print(f"Training accuracy: {train_accuracy:.4f}")
    
    return lr_baseline

# Train baseline model
baseline_model = train_baseline_logistic_regression(X_train, y_train)
```

### 6.4 Model Predictions và Probabilities

```python
def generate_predictions(model, X_train, X_test, y_train, y_test):
    """
    Tạo predictions và probabilities
    """
    print("="*50)
    print("GENERATING PREDICTIONS")
    print("="*50)
    
    # Generate predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Generate prediction probabilities
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    print("✓ Predictions generated for training and test sets")
    
    # Quick accuracy check
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Prediction distribution
    print(f"\nPrediction distribution:")
    print(f"Training - No Churn: {(y_train_pred == 0).sum()}, Churn: {(y_train_pred == 1).sum()}")
    print(f"Test - No Churn: {(y_test_pred == 0).sum()}, Churn: {(y_test_pred == 1).sum()}")
    
    return y_train_pred, y_test_pred, y_train_prob, y_test_prob

# Generate predictions
y_train_pred, y_test_pred, y_train_prob, y_test_prob = generate_predictions(
    baseline_model, X_train, X_test, y_train, y_test
)
```

---

## Phần 7: INTERPRET - Model Evaluation và Business Insights

### 7.1 Comprehensive Model Evaluation

```python
def comprehensive_model_evaluation(model, X_train, X_test, y_train, y_test, 
                                 y_train_pred, y_test_pred, y_train_prob, y_test_prob):
    """
    Đánh giá toàn diện hiệu suất mô hình
    """
    print("="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # 1. Classification Metrics
    def print_classification_metrics(y_true, y_pred, y_prob, dataset_name):
        print(f"\n{dataset_name.upper()} SET METRICS:")
        print("-" * 40)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        
        return {
            'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 
            'F1-Score': f1, 'AUC-ROC': auc
        }
    
    # Calculate metrics for both sets
    train_metrics = print_classification_metrics(y_train, y_train_pred, y_train_prob, "training")
    test_metrics = print_classification_metrics(y_test, y_test_pred, y_test_prob, "test")
    
    # 2. Confusion Matrices
    print(f"\nCONFUSION MATRICES:")
    print("-" * 40)
    
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    print("Training Set:")
    print(cm_train)
    print("\nTest Set:")
    print(cm_test)
    
    # 3. Detailed Classification Reports
    print(f"\nDETAILED CLASSIFICATION REPORTS:")
    print("-" * 40)
    print("Training Set:")
    print(classification_report(y_train, y_train_pred, target_names=['No Churn', 'Churn']))
    
    print("Test Set:")
    print(classification_report(y_test, y_test_pred, target_names=['No Churn', 'Churn']))
    
    # 4. Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Confusion Matrices
    sns.heatmap(cm_train, annot=True, fmt='d', ax=axes[0,0], cmap='Blues')
    axes[0,0].set_title('Training Set - Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    sns.heatmap(cm_test, annot=True, fmt='d', ax=axes[0,1], cmap='Blues')
    axes[0,1].set_title('Test Set - Confusion Matrix')
    axes[0,1].set_xlabel('Predicted')
    axes[0,1].set_ylabel('Actual')
    
    # ROC Curves
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
    
    axes[0,2].plot(fpr_train, tpr_train, label=f'Training AUC = {train_metrics["AUC-ROC"]:.3f}')
    axes[0,2].plot(fpr_test, tpr_test, label=f'Test AUC = {test_metrics["AUC-ROC"]:.3f}')
    axes[0,2].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    axes[0,2].set_xlabel('False Positive Rate')
    axes[0,2].set_ylabel('True Positive Rate')
    axes[0,2].set_title('ROC Curve')
    axes[0,2].legend()
    axes[0,2].grid(True)
    
    # Precision-Recall Curves
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_prob)
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_prob)
    
    axes[1,0].plot(recall_train, precision_train, label='Training')
    axes[1,0].plot(recall_test, precision_test, label='Test')
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].set_title('Precision-Recall Curve')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Prediction Probability Distributions
    axes[1,1].hist(y_train_prob[y_train==0], alpha=0.7, bins=30, label='No Churn (Train)', density=True)
    axes[1,1].hist(y_train_prob[y_train==1], alpha=0.7, bins=30, label='Churn (Train)', density=True)
    axes[1,1].set_xlabel('Predicted Probability')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Training Set - Probability Distribution')
    axes[1,1].legend()
    
    axes[1,2].hist(y_test_prob[y_test==0], alpha=0.7, bins=30, label='No Churn (Test)', density=True)
    axes[1,2].hist(y_test_prob[y_test==1], alpha=0.7, bins=30, label='Churn (Test)', density=True)
    axes[1,2].set_xlabel('Predicted Probability')
    axes[1,2].set_ylabel('Density')
    axes[1,2].set_title('Test Set - Probability Distribution')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 5. Model Performance Summary
    metrics_df = pd.DataFrame([train_metrics, test_metrics], 
                             index=['Training', 'Test']).round(4)
    
    print(f"\nMODEL PERFORMANCE SUMMARY:")
    print("-" * 40)
    print(metrics_df)
    
    # 6. Overfitting Check
    print(f"\nOVERFITTING ANALYSIS:")
    print("-" * 40)
    
    train_test_diff = train_metrics['Accuracy'] - test_metrics['Accuracy']
    if train_test_diff > 0.05:
        print(f"⚠️  Potential overfitting detected!")
        print(f"   Training accuracy ({train_metrics['Accuracy']:.4f}) > Test accuracy ({test_metrics['Accuracy']:.4f})")
        print(f"   Difference: {train_test_diff:.4f}")
    else:
        print(f"✓ Model generalizes well")
        print(f"  Training-Test accuracy difference: {train_test_diff:.4f}")
    
    return train_metrics, test_metrics, metrics_df

# Perform comprehensive evaluation
train_metrics, test_metrics, performance_summary = comprehensive_model_evaluation(
    baseline_model, X_train, X_test, y_train, y_test,
    y_train_pred, y_test_pred, y_train_prob, y_test_prob
)
```

### 7.2 Feature Importance Analysis

```python
def analyze_feature_importance(model, feature_names):
    """
    Phân tích tầm quan trọng của các features
    """
    print("="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get coefficients from the logistic regression model
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    
    # Create feature importance DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients),
        'Odds_Ratio': np.exp(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print(f"Model Intercept: {intercept:.4f}")
    print(f"\nTOP 15 MOST IMPORTANT FEATURES:")
    print("-" * 60)
    print(feature_importance_df.head(15).round(4))
    
    # Interpretation guide
    print(f"\nINTERPRETATION GUIDE:")
    print("-" * 40)
    print("• Positive coefficient: Increases churn probability")
    print("• Negative coefficient: Decreases churn probability")
    print("• Larger |coefficient|: Stronger impact")
    print("• Odds ratio > 1: Increases odds of churn")
    print("• Odds ratio < 1: Decreases odds of churn")
    
    # Visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top 10 features by absolute coefficient
    top_10 = feature_importance_df.head(10)
    
    colors = ['red' if coef < 0 else 'blue' for coef in top_10['Coefficient']]
    ax1.barh(range(len(top_10)), top_10['Coefficient'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels(top_10['Feature'])
    ax1.set_xlabel('Coefficient Value')
    ax1.set_title('Top 10 Features by Coefficient')
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # Odds ratios for top 10 features
    ax2.barh(range(len(top_10)), top_10['Odds_Ratio'], color='green', alpha=0.7)
    ax2.set_yticks(range(len(top_10)))
    ax2.set_yticklabels(top_10['Feature'])
    ax2.set_xlabel('Odds Ratio')
    ax2.set_title('Top 10 Features by Odds Ratio')
    ax2.axvline(x=1, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    # Distribution of coefficients
    ax3.hist(coefficients, bins=20, alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', label='Zero coefficient')
    ax3.set_xlabel('Coefficient Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Feature Coefficients')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Positive vs Negative impact features
    positive_features = (coefficients > 0).sum()
    negative_features = (coefficients < 0).sum()
    zero_features = (coefficients == 0).sum()
    
    impact_data = [positive_features, negative_features, zero_features]
    impact_labels = ['Positive Impact\n(Increase Churn)', 'Negative Impact\n(Decrease Churn)', 'No Impact']
    colors = ['red', 'blue', 'gray']
    
    ax4.pie(impact_data, labels=impact_labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Feature Impact Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Business insights from top features
    print(f"\nBUSINESS INSIGHTS FROM TOP FEATURES:")
    print("-" * 50)
    
    # Features that increase churn
    increase_churn = feature_importance_df[feature_importance_df['Coefficient'] > 0].head(5)
    print("Features that INCREASE churn probability:")
    for _, row in increase_churn.iterrows():
        print(f"  • {row['Feature']}: {row['Coefficient']:.3f} (OR: {row['Odds_Ratio']:.3f})")
    
    # Features that decrease churn  
    decrease_churn = feature_importance_df[feature_importance_df['Coefficient'] < 0].head(5)
    print(f"\nFeatures that DECREASE churn probability:")
    for _, row in decrease_churn.iterrows():
        print(f"  • {row['Feature']}: {row['Coefficient']:.3f} (OR: {row['Odds_Ratio']:.3f})")
    
    return feature_importance_df

# Analyze feature importance
feature_importance = analyze_feature_importance(baseline_model, X.columns)
```

### 7.3 Customer Segmentation và Risk Analysis

```python
def customer_risk_segmentation(df_original, model, scaler, X_columns):
    """
    Phân khúc khách hàng theo risk level
    """
    print("="*60)
    print("CUSTOMER RISK SEGMENTATION")
    print("="*60)
    
    # Prepare data for prediction
    df_segment = df_original.copy()
    
    # Get the same features used in modeling
    X_all, _, _, _ = prepare_modeling_data(df_segment, final_features)
    X_all = X_all.reindex(columns=X_columns, fill_value=0)  # Ensure same column order
    
    # Scale the features
    X_all_scaled = scaler.transform(X_all)
    
    # Get churn probabilities
    churn_probabilities = model.predict_proba(X_all_scaled)[:, 1]
    df_segment['ChurnProbability'] = churn_probabilities
    df_segment['PredictedChurn'] = model.predict(X_all_scaled)
    
    # Create risk segments
    df_segment['RiskSegment'] = pd.cut(df_segment['ChurnProbability'], 
                                      bins=[0, 0.3, 0.7, 1.0], 
                                      labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    # Segment analysis
    segment_analysis = df_segment.groupby('RiskSegment').agg({
        'CustomerID': 'count',
        'ChurnProbability': ['mean', 'min', 'max'],
        'Tenure': 'mean',
        'MonthlyCharges': 'mean',
        'TotalCharges': 'mean',
        'Churn': 'mean'  # Actual churn rate
    }).round(3)
    
    # Flatten column names
    segment_analysis.columns = ['_'.join(col).strip() if col[1] else col[0] for col in segment_analysis.columns.values]
    segment_analysis = segment_analysis.rename(columns={
        'CustomerID_count': 'Customer_Count',
        'ChurnProbability_mean': 'Avg_Churn_Prob',
        'ChurnProbability_min': 'Min_Churn_Prob',
        'ChurnProbability_max': 'Max_Churn_Prob',
        'Tenure_mean': 'Avg_Tenure',
        'MonthlyCharges_mean': 'Avg_Monthly_Charges',
        'TotalCharges_mean': 'Avg_Total_Charges',
        'Churn_mean': 'Actual_Churn_Rate'
    })
    
    print("CUSTOMER RISK SEGMENTS:")
    print("-" * 40)
    print(segment_analysis)
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Risk segment distribution
    risk_counts = df_segment['RiskSegment'].value_counts()
    axes[0,0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                  colors=['green', 'orange', 'red'])
    axes[0,0].set_title('Customer Risk Distribution')
    
    # Churn probability distribution by segment
    for segment in ['Low Risk', 'Medium Risk', 'High Risk']:
        data = df_segment[df_segment['RiskSegment'] == segment]['ChurnProbability']
        axes[0,1].hist(data, alpha=0.7, label=segment, bins=20)
    axes[0,1].set_xlabel('Churn Probability')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Churn Probability Distribution by Risk Segment')
    axes[0,1].legend()
    
    # Actual vs Predicted churn rate by segment
    segment_comparison = df_segment.groupby('RiskSegment').agg({
        'Churn': 'mean',
        'ChurnProbability': 'mean'
    })
    
    x_pos = np.arange(len(segment_comparison.index))
    width = 0.35
    
    axes[0,2].bar(x_pos - width/2, segment_comparison['Churn'], width, 
                  label='Actual Churn Rate', alpha=0.7)
    axes[0,2].bar(x_pos + width/2, segment_comparison['ChurnProbability'], width, 
                  label='Predicted Churn Prob', alpha=0.7)
    axes[0,2].set_xlabel('Risk Segment')
    axes[0,2].set_ylabel('Rate')
    axes[0,2].set_title('Actual vs Predicted Churn by Segment')
    axes[0,2].set_xticks(x_pos)
    axes[0,2].set_xticklabels(segment_comparison.index)
    axes[0,2].legend()
    
    # Business metrics by segment
    metrics = ['Avg_Tenure', 'Avg_Monthly_Charges', 'Avg_Total_Charges']
    segment_metrics = segment_analysis[metrics]
    
    for i, metric in enumerate(metrics):
        axes[1,i].bar(segment_metrics.index, segment_metrics[metric], 
                      color=['green', 'orange', 'red'], alpha=0.7)
        axes[1,i].set_title(f'{metric} by Risk Segment')
        axes[1,i].set_ylabel(metric.replace('_', ' '))
        axes[1,i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # High-risk customer analysis
    high_risk_customers = df_segment[df_segment['RiskSegment'] == 'High Risk']
    
    print(f"\nHIGH RISK CUSTOMERS ANALYSIS:")
    print("-" * 40)
    print(f"Total high-risk customers: {len(high_risk_customers):,}")
    print(f"Average churn probability: {high_risk_customers['ChurnProbability'].mean():.3f}")
    print(f"Actual churn rate: {high_risk_customers['Churn'].mean():.3f}")
    
    # Top characteristics of high-risk customers
    print(f"\nTOP CHARACTERISTICS OF HIGH-RISK CUSTOMERS:")
    categorical_cols = ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport']
    
    for col in categorical_cols:
        if col in high_risk_customers.columns:
            top_category = high_risk_customers[col].mode().iloc[0]
            percentage = (high_risk_customers[col] == top_category).mean()
            print(f"  • {col}: {top_category} ({percentage:.1%})")
    
    return df_segment, segment_analysis

# Perform customer risk segmentation
df_with_risk, risk_analysis = customer_risk_segmentation(df_engineered, baseline_model, scaler, X.columns)
```

### 7.4 Business Recommendations và Action Plan

```python
def generate_business_recommendations(feature_importance, risk_analysis, performance_metrics):
    """
    Tạo business recommendations dựa trên kết quả phân tích
    """
    print("="*60)
    print("BUSINESS RECOMMENDATIONS & ACTION PLAN")
    print("="*60)
    
    # Model performance summary
    print("1. MODEL PERFORMANCE OVERVIEW:")
    print("-" * 40)
    test_auc = performance_metrics['AUC-ROC']
    test_accuracy = performance_metrics['Accuracy']
    
    if test_auc >= 0.8:
        performance_level = "Excellent"
    elif test_auc >= 0.7:
        performance_level = "Good"
    elif test_auc >= 0.6:
        performance_level = "Fair"
    else:
        performance_level = "Poor"
    
    print(f"  • Model Performance: {performance_level}")
    print(f"  • AUC-ROC Score: {test_auc:.3f}")
    print(f"  • Accuracy: {test_accuracy:.3f}")
    print(f"  • Business Readiness: {'Ready for deployment' if test_auc >= 0.7 else 'Needs improvement'}")
    
    # Key risk factors
    print(f"\n2. KEY CHURN RISK FACTORS:")
    print("-" * 40)
    top_risk_factors = feature_importance[feature_importance['Coefficient'] > 0].head(5)
    
    for i, (_, row) in enumerate(top_risk_factors.iterrows(), 1):
        impact = "High" if row['Abs_Coefficient'] > 0.5 else "Medium" if row['Abs_Coefficient'] > 0.2 else "Low"
        print(f"  {i}. {row['Feature']}")
        print(f"     Impact: {impact} (Coefficient: {row['Coefficient']:.3f})")
        print(f"     Odds Ratio: {row['Odds_Ratio']:.3f}")
    
    # Protective factors
    print(f"\n3. CHURN PROTECTION FACTORS:")
    print("-" * 40)
    protection_factors = feature_importance[feature_importance['Coefficient'] < 0].head(5)
    
    for i, (_, row) in enumerate(protection_factors.iterrows(), 1):
        impact = "High" if row['Abs_Coefficient'] > 0.5 else "Medium" if row['Abs_Coefficient'] > 0.2 else "Low"
        print(f"  {i}. {row['Feature']}")
        print(f"     Protection Level: {impact} (Coefficient: {row['Coefficient']:.3f})")
        print(f"     Odds Ratio: {row['Odds_Ratio']:.3f}")
    
    # Customer segment insights
    print(f"\n4. CUSTOMER SEGMENT INSIGHTS:")
    print("-" * 40)
    
    total_customers = risk_analysis['Customer_Count'].sum()
    high_risk_customers = risk_analysis.loc['High Risk', 'Customer_Count']
    high_risk_percentage = (high_risk_customers / total_customers) * 100
    
    print(f"  • Total Customers: {total_customers:,}")
    print(f"  • High Risk Customers: {high_risk_customers:,} ({high_risk_percentage:.1f}%)")
    print(f"  • Estimated Revenue at Risk: ${high_risk_customers * risk_analysis.loc['High Risk', 'Avg_Monthly_Charges'] * 12:,.0f}")
    
    # Actionable recommendations
    print(f"\n5. ACTIONABLE RECOMMENDATIONS:")
    print("-" * 40)
    
    print("A. IMMEDIATE ACTIONS (High Priority):")
    print("   1. HIGH-RISK CUSTOMER INTERVENTION:")
    print("      • Implement proactive retention campaigns for high-risk customers")
    print("      • Offer contract upgrades with attractive incentives")
    print("      • Provide dedicated customer success management")
    print("      • Consider temporary discounts or service upgrades")
    
    print("   2. CONTRACT OPTIMIZATION:")
    print("      • Focus on converting month-to-month customers to longer contracts")
    print("      • Develop attractive annual/biennial contract packages")
    print("      • Implement early renewal incentives")
    
    print("   3. PAYMENT METHOD IMPROVEMENTS:")
    print("      • Encourage automatic payment methods")
    print("      • Provide incentives for credit card/bank transfer payments")
    print("      • Simplify payment processes")
    
    print("\nB. SHORT-TERM INITIATIVES (3-6 months):")
    print("   1. SERVICE ENHANCEMENT:")
    print("      • Improve technical support quality and accessibility")
    print("      • Develop self-service options for common issues")
    print("      • Implement proactive service monitoring")
    
    print("   2. NEW CUSTOMER EXPERIENCE:")
    print("      • Enhance onboarding process for new customers")
    print("      • Implement early engagement programs")
    print("      • Provide extra support during first 6 months")
    
    print("   3. PRICING STRATEGY:")
    print("      • Review pricing structure for high-risk segments")
    print("      • Develop value-based pricing packages")
    print("      • Consider loyalty rewards program")
    
    print("\nC. LONG-TERM STRATEGIC INITIATIVES (6+ months):")
    print("   1. PREDICTIVE ANALYTICS:")
    print("      • Deploy real-time churn prediction system")
    print("      • Implement automated intervention triggers")
    print("      • Develop customer lifetime value optimization")
    
    print("   2. PRODUCT DEVELOPMENT:")
    print("      • Enhance service offerings based on churn factors")
    print("      • Develop sticky features that increase switching costs")
    print("      • Create customer community and engagement platforms")
    
    print("   3. ORGANIZATIONAL CAPABILITIES:")
    print("      • Train customer service teams on retention techniques")
    print("      • Implement customer success metrics and KPIs")
    print("      • Develop data-driven decision making culture")
    
    # ROI estimation
    print(f"\n6. ESTIMATED ROI OF RETENTION EFFORTS:")
    print("-" * 40)
    
    avg_monthly_revenue = risk_analysis.loc['High Risk', 'Avg_Monthly_Charges']
    customer_lifetime_months = 24  # Assume 2 years average
    
    # Conservative estimates
    retention_program_cost_per_customer = 50
    retention_success_rate = 0.3  # 30% success rate
    
    revenue_saved = high_risk_customers * retention_success_rate * avg_monthly_revenue * customer_lifetime_months
    program_cost = high_risk_customers * retention_program_cost_per_customer
    net_roi = revenue_saved - program_cost
    roi_percentage = (net_roi / program_cost) * 100
    
    print(f"  • Revenue at Risk: ${high_risk_customers * avg_monthly_revenue * customer_lifetime_months:,.0f}")
    print(f"  • Estimated Program Cost: ${program_cost:,.0f}")
    print(f"  • Expected Revenue Saved: ${revenue_saved:,.0f}")
    print(f"  • Net ROI: ${net_roi:,.0f}")
    print(f"  • ROI Percentage: {roi_percentage:.1f}%")
    
    # Implementation timeline
    print(f"\n7. IMPLEMENTATION TIMELINE:")
    print("-" * 40)
    print("  Week 1-2:  Setup high-risk customer identification system")
    print("  Week 3-4:  Launch immediate retention campaigns")
    print("  Month 2:   Implement contract optimization programs")
    print("  Month 3:   Deploy payment method improvement initiatives")
    print("  Month 4-6: Execute service enhancement projects")
    print("  Month 6+:  Monitor results and iterate strategies")
    
    # KPIs to track
    print(f"\n8. KEY PERFORMANCE INDICATORS TO TRACK:")
    print("-" * 40)
    print("  • Monthly churn rate by risk segment")
    print("  • Model accuracy and prediction stability")
    print("  • Customer retention rate after interventions")
    print("  • Revenue impact of retention programs")
    print("  • Customer satisfaction scores")
    print("  • Contract conversion rates")
    print("  • Payment method adoption rates")

# Generate comprehensive business recommendations
generate_business_recommendations(feature_importance, risk_analysis, test_metrics)
```

---

## Phần 8: Model Improvement và Advanced Techniques

### 8.1 Hyperparameter Tuning

```python
def hyperparameter_tuning(X_train, y_train):
    """
    Tối ưu hyperparameters cho Logistic Regression
    """
    print("="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    # Define parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000]
    }
    
    # Handle solver-penalty compatibility
    param_combinations = []
    for C in param_grid['C']:
        for penalty in param_grid['penalty']:
            for solver in param_grid['solver']:
                for max_iter in param_grid['max_iter']:
                    # Check compatibility
                    if penalty == 'elasticnet' and solver not in ['saga']:
                        continue
                    if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                        continue
                    if penalty == 'l2' and solver not in ['liblinear', 'saga']:
                        continue
                    
                    param_combinations.append({
                        'C': C,
                        'penalty': penalty,
                        'solver': solver,
                        'max_iter': max_iter
                    })
    
    print(f"Total parameter combinations to test: {len(param_combinations)}")
    
    # Grid search with cross-validation
    lr_tuned = LogisticRegression(random_state=42)
    
    grid_search = GridSearchCV(
        estimator=lr_tuned,
        param_grid=param_combinations,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    print("Performing grid search...")
    grid_search.fit(X_train, y_train)
    
    # Results
    print(f"✓ Grid search completed!")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Improvement over baseline: {grid_search.best_score_ - baseline_model.score(X_train, y_train):.4f}")
    
    # Top 5 parameter combinations
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_results = results_df.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
    
    print(f"\nTOP 5 PARAMETER COMBINATIONS:")
    print("-" * 50)
    for i, (_, row) in enumerate(top_results.iterrows(), 1):
        print(f"{i}. Score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
        print(f"   Parameters: {row['params']}")
    
    return grid_search.best_estimator_, grid_search.best_params_

# Perform hyperparameter tuning
best_model, best_params = hyperparameter_tuning(X_train, y_train)
```

### 8.2 Cross-Validation Analysis

```python
def cross_validation_analysis(model, X_train, y_train):
    """
    Phân tích hiệu suất mô hình qua cross-validation
    """
    print("="*60)
    print("CROSS-VALIDATION ANALYSIS")
    print("="*60)
    
    # Perform cross-validation with multiple metrics
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    cv_results = {}
    for metric in scoring_metrics:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=metric)
        cv_results[metric] = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
        
        print(f"{metric.upper()}:")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Std:  {scores.std():.4f}")
        print(f"  Range: [{scores.min():.4f}, {scores.max():.4f}]")
        print()
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot of cross-validation scores
    cv_scores_df = pd.DataFrame({metric: cv_results[metric]['scores'] for metric in scoring_metrics})
    cv_scores_df.boxplot(ax=ax1)
    ax1.set_title('Cross-Validation Scores Distribution')
    ax1.set_ylabel('Score')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Mean scores with error bars
    means = [cv_results[metric]['mean'] for metric in scoring_metrics]
    stds = [cv_results[metric]['std'] for metric in scoring_metrics]
    
    ax2.bar(scoring_metrics, means, yerr=stds, alpha=0.7, capsize=5)
    ax2.set_title('Cross-Validation Mean Scores with Standard Deviation')
    ax2.set_ylabel('Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Model stability analysis
    print("MODEL STABILITY ANALYSIS:")
    print("-" * 40)
    
    stability_threshold = 0.05  # 5% threshold for stability
    
    for metric in scoring_metrics:
        std_score = cv_results[metric]['std']
        if std_score < stability_threshold:
            stability = "Stable"
        elif std_score < stability_threshold * 2:
            stability = "Moderately Stable"
        else:
            stability = "Unstable"
        
        print(f"  {metric}: {stability} (std: {std_score:.4f})")
    
    return cv_results

# Perform cross-validation analysis
cv_results = cross_validation_analysis(best_model, X_train, y_train)
```

---

## Phần 9: Bài tập thực hành

### Bài tập 1: Model Comparison và Ensemble

**Mục tiêu:** So sánh Logistic Regression với các thuật toán khác và tạo ensemble model

```python
# Hướng dẫn thực hiện:

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def compare_models_exercise():
    """
    Bài tập: So sánh multiple algorithms
    """
    print("="*60)
    print("BÀI TẬP 1: MODEL COMPARISON")
    print("="*60)
    
    # TODO: Implement các models sau
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    # TODO: Train each model và evaluate
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        # TODO: Fit model
        # TODO: Make predictions  
        # TODO: Calculate metrics
        # TODO: Store results
    
    # TODO: Create comparison DataFrame
    # TODO: Visualize results
    # TODO: Identify best performing model
    
    print("Exercise 1 completed!")

# Gợi ý thực hiện:
# 1. Train từng model trên training set
# 2. Evaluate trên test set với multiple metrics
# 3. So sánh AUC-ROC, Precision, Recall, F1
# 4. Analyze trade-offs giữa interpretability và performance
# 5. Create ensemble prediction từ top 3 models
```

### Bài tập 2: Feature Engineering Advanced

**Mục tiêu:** Tạo thêm features mới và đánh giá impact

```python
def advanced_feature_engineering_exercise():
    """
    Bài tập: Advanced Feature Engineering
    """
    print("="*60)
    print("BÀI TẬP 2: ADVANCED FEATURE ENGINEERING")
    print("="*60)
    
    # TODO: Implement advanced feature engineering techniques
    
    # 1. Polynomial Features
    # TODO: Create polynomial combinations of numerical features
    
    # 2. Interaction Features  
    # TODO: Create interaction terms between important features
    
    # 3. Binning và Discretization
    # TODO: Create optimal bins for continuous variables
    
    # 4. Domain-specific Features
    # TODO: Create business logic based features như:
    #   - Customer loyalty score
    #   - Service utilization ratio
    #   - Payment behavior patterns
    
    # 5. Time-based Features (if available)
    # TODO: Seasonality, trends, recency features
    
    # 6. Feature Selection
    # TODO: Use statistical tests để select best features
    # TODO: Implement recursive feature elimination
    
    # 7. Evaluation
    # TODO: Compare performance before/after feature engineering
    
    print("Exercise 2 completed!")

# Gợi ý:
# - Sử dụng sklearn.preprocessing.PolynomialFeatures
# - Implement custom feature transformations
# - Use feature_selection modules
# - Validate với cross-validation
```

### Bài tập 3: Threshold Optimization

**Mục tiêu:** Tối ưu decision threshold cho business objectives

```python
def threshold_optimization_exercise():
    """
    Bài tập: Optimize Decision Threshold
    """
    print("="*60)
    print("BÀI TẬP 3: THRESHOLD OPTIMIZATION")
    print("="*60)
    
    # TODO: Implement threshold optimization
    
    # 1. Business Cost Analysis
    # TODO: Define costs of false positives và false negatives
    # Cost of wrongly predicting churn (false positive)
    # Cost of missing actual churn (false negative)
    
    # 2. Threshold Analysis
    # TODO: Test different thresholds từ 0.1 to 0.9
    # TODO: Calculate metrics for each threshold
    
    # 3. ROC Analysis
    # TODO: Find optimal threshold using ROC curve
    # TODO: Calculate Youden's Index
    
    # 4. Precision-Recall Analysis
    # TODO: Find optimal threshold using PR curve
    
    # 5. Business Impact Analysis
    # TODO: Calculate expected profit/loss for each threshold
    # TODO: Find threshold that maximizes business value
    
    # 6. Visualization
    # TODO: Plot metrics vs threshold
    # TODO: Highlight optimal thresholds
    
    print("Exercise 3 completed!")

# Business scenario:
# - Cost of retention campaign: $50 per customer
# - Value of retained customer: $500 (annual value)
# - Find threshold that maximizes ROI
```

### Bài tập 4: Real-time Prediction System

**Mục tiêu:** Xây dựng system để predict churn cho khách hàng mới

```python
def prediction_system_exercise():
    """
    Bài tập: Build Prediction System
    """
    print("="*60)
    print("BÀI TẬP 4: REAL-TIME PREDICTION SYSTEM")
    print("="*60)
    
    # TODO: Create production-ready prediction system
    
    class ChurnPredictor:
        def __init__(self):
            self.model = None
            self.scaler = None
            self.feature_columns = None
            self.is_trained = False
        
        def train(self, X_train, y_train):
            """Train the model"""
            # TODO: Implement training logic
            pass
        
        def predict_single_customer(self, customer_data):
            """Predict churn for single customer"""
            # TODO: Implement single prediction
            # TODO: Handle data preprocessing
            # TODO: Return probability và risk category
            pass
        
        def predict_batch(self, customers_data):
            """Predict churn for multiple customers"""
            # TODO: Implement batch prediction
            pass
        
        def get_feature_importance(self):
            """Return feature importance explanation"""
            # TODO: Return interpretable feature importance
            pass
        
        def save_model(self, filepath):
            """Save trained model"""
            # TODO: Implement model persistence
            pass
        
        def load_model(self, filepath):
            """Load trained model"""
            # TODO: Implement model loading
            pass
    
    # TODO: Test the system với sample data
    # TODO: Create API endpoint simulation
    # TODO: Implement input validation
    # TODO: Add logging và monitoring
    
    print("Exercise 4 completed!")

# Requirements:
# - Input validation
# - Error handling  
# - Model versioning
# - Performance monitoring
# - Explanation capability
```

### Bài tập 5: A/B Testing Framework

**Mục tiêu:** Thiết kế A/B test để validate retention strategies

```python
def ab_testing_framework_exercise():
    """
    Bài tập: A/B Testing for Retention Strategies
    """
    print("="*60)
    print("BÀI TẬP 5: A/B TESTING FRAMEWORK")
    print("="*60)
    
    # TODO: Design A/B testing framework
    
    # 1. Experimental Design
    # TODO: Define control và treatment groups
    # TODO: Calculate sample size requirements
    # TODO: Define success metrics
    
    # 2. Customer Segmentation for Testing
    # TODO: Stratify customers by risk level
    # TODO: Ensure balanced groups
    
    # 3. Treatment Strategies
    # TODO: Design different retention interventions:
    #   - Discount offers
    #   - Service upgrades  
    #   - Personalized communications
    #   - Contract incentives
    
    # 4. Statistical Testing
    # TODO: Implement hypothesis testing
    # TODO: Calculate statistical significance
    # TODO: Estimate effect sizes
    
    # 5. Simulation Framework
    # TODO: Simulate different scenarios
    # TODO: Test power analysis
    
    def design_experiment(treatment_effect=0.05, alpha=0.05, power=0.8):
        """Design A/B test experiment"""
        # TODO: Calculate required sample size
        # TODO: Define randomization strategy
        # TODO: Set up tracking metrics
        pass
    
    def analyze_results(control_group, treatment_group):
        """Analyze A/B test results"""
        # TODO: Statistical significance testing
        # TODO: Confidence intervals
        # TODO: Business impact estimation
        pass
    
    print("Exercise 5 completed!")

# Scenario:
# Test effectiveness of different retention campaigns
# Target: Reduce churn by 20% for high-risk customers
# Budget: $10,000 for testing
# Timeline: 3 months
```

---

**Lưu ý cho giáo viên:**

- Encourage students để explore data thoroughly trước khi jump vào modeling
- Emphasize business context và practical applications
- Use real examples và case studies khi possible
- Promote iterative approach - start simple, then enhance
- Focus on interpretability và actionable insights
- Assign different datasets cho students để practice independently

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
```

---

## Tóm tắt và Key Takeaways

### Những điểm quan trọng đã học:

1. **OSEMN Pipeline** là framework chuẩn cho data science projects
2. **Data Exploration** là bước quan trọng nhất để understand business problem
3. **Feature Engineering** có thể significantly improve model performance  
4. **Logistic Regression** là excellent baseline model cho classification
5. **Model Interpretation** quan trọng hơn raw performance trong business context
6. **Customer Segmentation** enables targeted business strategies
7. **ROI Analysis** essential để justify data science investments

### Best Practices:

- Always start với thorough EDA trước khi modeling
- Use business logic để create meaningful features
- Cross-validate để ensure model stability
- Focus on actionable insights, không chỉ model metrics
- Consider deployment và maintenance từ early stages
- Document everything để ensure reproducibility

### Next Steps:

1. Practice với real-world datasets
2. Learn advanced algorithms (XGBoost, Neural Networks)
3. Study MLOps và model deployment
4. Develop business acumen và domain expertise
5. Learn experiment design và causal inference

---

## Tài liệu tham khảo

### Libraries Documentation:
- **Pandas**: https://pandas.pydata.org/docs/
- **Scikit-learn**: https://scikit-learn.org/stable/
- **Matplotlib/Seaborn**: https://matplotlib.org/, https://seaborn.pydata.org/

### Books:
- "Hands-On Machine Learning" by Aurélien Géron
- "Python for Data Analysis" by Wes McKinney  
- "Feature Engineering for Machine Learning" by Alice Zheng

### Online Courses:
- Coursera Machine Learning Course
- Kaggle Learn Modules
- Fast.ai Practical Deep Learning
