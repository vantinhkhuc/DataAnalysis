# Chapter 08: Fine-Tuning Classification Algorithms

## Mục tiêu học tập
Sau khi hoàn thành bài học này, học viên sẽ có thể:
- Tối ưu hóa phân tích dự đoán sử dụng các thuật toán phân loại
- Triển khai và điều chỉnh Support Vector Machines, Decision Trees, và Random Forests
- Lựa chọn metrics phù hợp để đánh giá hiệu suất mô hình
- Giải quyết bài toán dự đoán customer churn
- Tối ưu hóa và đánh giá thuật toán phân loại tốt nhất

## Phần 1: Giới thiệu về Classification Algorithms

### 1.1 Tổng quan về Classification
Classification là một dạng supervised learning nhằm dự đoán nhãn (label) của dữ liệu dựa trên các đặc trung đã học từ training data.

### 1.2 Các thuật toán chính trong Scikit-learn

#### Support Vector Machines (SVM)
- **Nguyên lý**: Tìm hyperplane tối ưu để phân tách các class
- **Ưu điểm**: Hiệu quả với high-dimensional data, memory efficient
- **Nhược điểm**: Chậm với dataset lớn, nhạy cảm với feature scaling

#### Decision Trees
- **Nguyên lý**: Tạo cây quyết định dựa trên việc chia dữ liệu theo features
- **Ưu điểm**: Dễ hiểu và giải thích, không cần feature scaling
- **Nhược điểm**: Dễ overfitting, không ổn định

#### Random Forests
- **Nguyên lý**: Kết hợp nhiều decision trees (ensemble method)
- **Ưu điểm**: Giảm overfitting, robust, cung cấp feature importance
- **Nhược điểm**: Khó giải thích hơn single tree

```python
# Import các thư viện cần thiết
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')
```

## Phần 2: Triển khai Tree-based Classification Models

### 2.1 Decision Trees cho Classification

```python
# Tạo và huấn luyện Decision Tree
def create_decision_tree(X_train, y_train, max_depth=None, min_samples_split=2):
    """
    Tạo và huấn luyện Decision Tree Classifier
    """
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    dt.fit(X_train, y_train)
    return dt

# Visualization Decision Tree
from sklearn.tree import plot_tree

def visualize_tree(model, feature_names, class_names, max_depth=3):
    """
    Vẽ cây quyết định
    """
    plt.figure(figsize=(20, 10))
    plot_tree(model, 
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              max_depth=max_depth)
    plt.title("Decision Tree Visualization")
    plt.show()
```

### 2.2 Random Forest Implementation

```python
def create_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """
    Tạo và huấn luyện Random Forest Classifier
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    rf.fit(X_train, y_train)
    return rf

def plot_feature_importance(model, feature_names, top_n=15):
    """
    Vẽ biểu đồ feature importance
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance")
    plt.bar(range(top_n), importance[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()
```

### 2.3 So sánh với Regression Trees

**Điểm khác biệt chính:**
- **Classification**: Dự đoán class/category (discrete output)
- **Regression**: Dự đoán giá trị liên tục (continuous output)
- **Splitting criteria**: Classification sử dụng Gini impurity hoặc entropy, Regression sử dụng MSE
- **Leaf nodes**: Classification trả về class phổ biến nhất, Regression trả về giá trị trung bình

## Phần 3: Performance Metrics cho Classification

### 3.1 Confusion Matrix và các metrics cơ bản

```python
def evaluate_classification_model(model, X_test, y_test, model_name="Model"):
    """
    Đánh giá toàn diện mô hình classification
    """
    # Dự đoán
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Tính toán metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"=== {model_name} Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    if y_pred_proba is not None:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"AUC-ROC: {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc if y_pred_proba is not None else None
    }
```

### 3.2 ROC Curve và AUC

```python
def plot_roc_curves(models, X_test, y_test):
    """
    Vẽ ROC curves cho nhiều mô hình
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### 3.3 Lựa chọn Metrics phù hợp

**Khi nào sử dụng metric nào:**
- **Accuracy**: Dataset balanced, quan tâm tổng thể
- **Precision**: Quan trọng tránh False Positive (spam detection)
- **Recall**: Quan trọng tránh False Negative (medical diagnosis)
- **F1-Score**: Dataset imbalanced, cần balance precision và recall
- **AUC-ROC**: So sánh models, threshold-independent

## Phần 4: Customer Churn Prediction Problem

### 4.1 Hiểu về Customer Churn
Customer Churn là hiện tượng khách hàng ngưng sử dụng sản phẩm/dịch vụ. Việc dự đoán churn giúp:
- Giữ chân khách hàng có giá trị
- Tối ưu hóa chi phí marketing
- Cải thiện customer retention strategy

### 4.2 Data Preparation cho Churn Prediction

```python
def prepare_churn_data():
    """
    Tạo sample dataset cho customer churn prediction
    """
    np.random.seed(42)
    n_samples = 5000
    
    # Tạo features
    data = {
        'customer_id': range(1, n_samples + 1),
        'tenure': np.random.normal(24, 12, n_samples),
        'monthly_charges': np.random.normal(65, 20, n_samples),
        'total_charges': np.random.normal(1500, 800, n_samples),
        'contract_length': np.random.choice([1, 12, 24], n_samples, p=[0.4, 0.3, 0.3]),
        'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic Check'], 
                                         n_samples, p=[0.4, 0.3, 0.3]),
        'internet_service': np.random.choice(['DSL', 'Fiber', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
        'tech_support': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'streaming_tv': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'multiple_lines': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'partner': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'dependents': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Tạo target variable (churn) với logic
    churn_prob = (
        0.1 +  # base rate
        0.3 * (df['tenure'] < 12) +  # new customers more likely to churn
        0.2 * (df['monthly_charges'] > 80) +  # high charges increase churn
        0.15 * (df['contract_length'] == 1) +  # month-to-month more likely to churn
        0.1 * (df['tech_support'] == 0) +  # no tech support increases churn
        0.1 * (df['senior_citizen'] == 1)  # senior citizens more likely to churn
    )
    
    df['churn'] = np.random.binomial(1, np.clip(churn_prob, 0, 1), n_samples)
    
    return df

# Load và explore data
churn_data = prepare_churn_data()
print("Dataset Shape:", churn_data.shape)
print("\nChurn Distribution:")
print(churn_data['churn'].value_counts(normalize=True))
```

### 4.3 Exploratory Data Analysis

```python
def explore_churn_data(df):
    """
    Phân tích khám phá dữ liệu churn
    """
    # Correlation với churn
    numerical_features = ['tenure', 'monthly_charges', 'total_charges', 'contract_length']
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(2, 2, i)
        df.boxplot(column=feature, by='churn')
        plt.title(f'{feature} by Churn Status')
        plt.suptitle('')  # Remove default title
    
    plt.tight_layout()
    plt.show()
    
    # Categorical features analysis
    categorical_features = ['payment_method', 'internet_service', 'tech_support', 'senior_citizen']
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(categorical_features, 1):
        plt.subplot(2, 2, i)
        pd.crosstab(df[feature], df['churn'], normalize='index').plot(kind='bar', ax=plt.gca())
        plt.title(f'Churn Rate by {feature}')
        plt.legend(['No Churn', 'Churn'])
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
```

## Phần 5: Tối ưu hóa và So sánh Classification Algorithms

### 5.1 Data Preprocessing

```python
def preprocess_churn_data(df):
    """
    Tiền xử lý dữ liệu cho modeling
    """
    # Copy dataframe
    df_processed = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = ['payment_method', 'internet_service']
    
    for col in categorical_columns:
        df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
    
    # Select features
    feature_columns = ['tenure', 'monthly_charges', 'total_charges', 'contract_length',
                      'payment_method_encoded', 'internet_service_encoded', 'tech_support',
                      'streaming_tv', 'multiple_lines', 'senior_citizen', 'partner', 'dependents']
    
    X = df_processed[feature_columns]
    y = df_processed['churn']
    
    return X, y, feature_columns
```

### 5.2 Model Training và Hyperparameter Tuning

```python
def train_and_tune_models(X_train, y_train):
    """
    Huấn luyện và tune hyperparameters cho các models
    """
    models = {}
    
    # Decision Tree với GridSearch
    dt_params = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    dt_grid = GridSearchCV(dt, dt_params, cv=5, scoring='f1', n_jobs=-1)
    dt_grid.fit(X_train, y_train)
    models['Decision Tree'] = dt_grid.best_estimator_
    
    print("Best Decision Tree params:", dt_grid.best_params_)
    
    # Random Forest với GridSearch
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='f1', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    models['Random Forest'] = rf_grid.best_estimator_
    
    print("Best Random Forest params:", rf_grid.best_params_)
    
    # SVM với GridSearch
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    
    svm = SVC(random_state=42, probability=True)
    svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='f1', n_jobs=-1)
    svm_grid.fit(X_train, y_train)
    models['SVM'] = svm_grid.best_estimator_
    
    print("Best SVM params:", svm_grid.best_params_)
    
    return models
```

### 5.3 Model Comparison và Selection

```python
def compare_models(models, X_train, y_train, X_test, y_test):
    """
    So sánh hiệu suất của các models
    """
    results = {}
    
    # Cross-validation scores
    print("=== Cross-Validation Results ===")
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Test set evaluation
    print("\n=== Test Set Results ===")
    for name, model in models.items():
        test_results = evaluate_classification_model(model, X_test, y_test, name)
        results[name].update(test_results)
    
    # Plot ROC curves
    plot_roc_curves(models, X_test, y_test)
    
    # Results summary
    results_df = pd.DataFrame(results).T
    print("\n=== Model Comparison Summary ===")
    print(results_df.round(4))
    
    return results_df

def select_best_model(results_df, primary_metric='f1'):
    """
    Chọn model tốt nhất dựa trên metric
    """
    best_model_name = results_df[primary_metric].idxmax()
    print(f"\nBest model based on {primary_metric}: {best_model_name}")
    print(f"Score: {results_df.loc[best_model_name, primary_metric]:.4f}")
    
    return best_model_name
```

## Phần 6: Pipeline hoàn chỉnh

```python
def complete_churn_prediction_pipeline():
    """
    Pipeline hoàn chỉnh cho churn prediction
    """
    print("=== Customer Churn Prediction Pipeline ===\n")
    
    # 1. Data Preparation
    print("1. Preparing data...")
    df = prepare_churn_data()
    X, y, feature_names = preprocess_churn_data(df)
    
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Feature Scaling (for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Model Training và Tuning
    print("\n2. Training and tuning models...")
    models = train_and_tune_models(X_train_scaled, y_train)
    
    # 5. Model Comparison
    print("\n3. Comparing models...")
    results = compare_models(models, X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 6. Best Model Selection
    print("\n4. Selecting best model...")
    best_model_name = select_best_model(results)
    best_model = models[best_model_name]
    
    # 7. Feature Importance (if applicable)
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n5. Feature importance for {best_model_name}:")
        plot_feature_importance(best_model, feature_names)
    
    return best_model, scaler, feature_names

# Chạy pipeline
best_model, scaler, feature_names = complete_churn_prediction_pipeline()
```

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

## Tài Liệu Tham Khảo

### Thư viện Python
- **Scikit-learn**: Documentation và User Guide
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **Imbalanced-learn**: Xử lý imbalanced datasets

### Metrics và Evaluation
- **Classification Report**: Chi tiết về precision, recall, F1 cho từng class
- **ROC-AUC**: Threshold-independent performance measure
- **Precision-Recall Curve**: Tốt cho imbalanced datasets
- **Cross-validation**: K-fold, stratified CV

### Best Practices
1. **Always split data before any preprocessing**
2. **Use appropriate metrics cho business problem**
3. **Validate models với cross-validation**
4. **Document assumptions và limitations**
5. **Consider business constraints trong model selection**

### Câu hỏi Ôn tập
1. Khi nào nên sử dụng precision thay vì recall làm primary metric?
2. Tại sao Random Forest thường ít overfitting hơn single Decision Tree?
3. Làm thế nào để xử lý categorical features có nhiều categories?
4. SVM linear vs RBF kernel: khi nào dùng cái nào?
5. Làm thế nào để interpret feature importance trong business context?
