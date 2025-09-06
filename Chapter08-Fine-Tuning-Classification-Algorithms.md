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

### Bài Tập 1: Cơ bản
1. Tạo một dataset classification đơn giản với 3 features và 2 classes
2. Triển khai Decision Tree, Random Forest, và SVM
3. So sánh hiệu suất sử dụng accuracy và F1-score
4. Vẽ confusion matrix cho từng model

### Bài Tập 2: Trung bình
1. Sử dụng dataset Iris hoặc Wine từ sklearn
2. Thực hiện feature selection sử dụng Random Forest feature importance
3. Tune hyperparameters cho các models sử dụng GridSearchCV
4. Tạo ensemble model kết hợp 3 algorithms

### Bài Tập 3: Nâng cao - Customer Churn Prediction
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
