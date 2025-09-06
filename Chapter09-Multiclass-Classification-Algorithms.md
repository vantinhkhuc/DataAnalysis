# Chapter 09: THUẬT TOÁN PHÂN LOẠI ĐA LỚP (MULTICLASS CLASSIFICATION)

**Môn học:** Phân tích Dữ liệu  
**Thời gian:** 4 tiết (180 phút)  
**Công cụ:** Python, Scikit-learn, Pandas, Numpy, Matplotlib

---

## MỤC TIÊU BÀI HỌC

Sau khi hoàn thành bài học này, học viên sẽ:
1. Hiểu và triển khai các thuật toán giải quyết bài toán phân loại đa lớp trong marketing analytics
2. Thành thạo các loại classifier khác nhau sử dụng thư viện scikit-learn
3. Diễn giải các chỉ số đánh giá micro và macro performance cho bài toán multiclass
4. Áp dụng các kỹ thuật sampling để giải quyết vấn đề dữ liệu không cân bằng
5. Vận dụng thuật toán và metric phù hợp cho bài toán thực tế

---

## PHẦN 1: GIỚI THIỆU VÀ NHẬN DẠNG BÀI TOÁN MULTICLASS CLASSIFICATION

### 1.1. Định nghĩa và Khái niệm

**Multiclass Classification** là bài toán phân loại trong đó mỗi instance có thể thuộc về một trong nhiều lớp (≥3 lớp), khác với binary classification chỉ có 2 lớp.

**Ví dụ trong Marketing Analytics:**
- Phân loại khách hàng: VIP, Premium, Standard, Basic
- Dự đoán kênh marketing hiệu quả: Email, Social Media, TV, Radio, Print
- Phân loại sản phẩm theo mức độ quan tâm: Rất quan tâm, Quan tâm, Bình thường, Không quan tâm

### 1.2. Chiến lược tiếp cận Multiclass

#### One-vs-Rest (OvR) / One-vs-All (OvA)
```python
# Ví dụ với 3 lớp A, B, C
# Tạo 3 binary classifier:
# - Classifier 1: A vs (B+C)
# - Classifier 2: B vs (A+C) 
# - Classifier 3: C vs (A+B)
```

#### One-vs-One (OvO)
```python
# Với 3 lớp A, B, C tạo 3 binary classifier:
# - Classifier 1: A vs B
# - Classifier 2: A vs C
# - Classifier 3: B vs C
# Tổng quát: n*(n-1)/2 classifiers cho n lớp
```

### 1.3. Code Demo: Nhận dạng và chuẩn bị dữ liệu

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Tạo dữ liệu marketing giả lập
np.random.seed(42)
n_samples = 1000

# Features: tuổi, thu nhập, thời gian sử dụng app, số lần click
X, y = make_classification(n_samples=n_samples, n_features=4, n_classes=4, 
                          n_informative=3, n_redundant=1, n_clusters_per_class=1,
                          class_sep=0.8, random_state=42)

# Tạo DataFrame với tên cột có ý nghĩa
feature_names = ['Tuoi', 'Thu_nhap', 'Thoi_gian_app', 'So_lan_click']
class_names = ['Basic', 'Standard', 'Premium', 'VIP']

df = pd.DataFrame(X, columns=feature_names)
df['Customer_Segment'] = [class_names[i] for i in y]

print("Phân phối các lớp:")
print(df['Customer_Segment'].value_counts())

# Visualize
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
df['Customer_Segment'].value_counts().plot(kind='bar')
plt.title('Phân phối Customer Segments')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
plt.scatter(df['Tuoi'], df['Thu_nhap'], c=y, cmap='viridis', alpha=0.6)
plt.xlabel('Tuổi')
plt.ylabel('Thu nhập')
plt.title('Scatter plot: Tuổi vs Thu nhập')

plt.tight_layout()
plt.show()
```

---

## PHẦN 2: CÁC LOẠI CLASSIFIER VÀ TRIỂN KHAI BẰNG SCIKIT-LEARN

### 2.1. Logistic Regression cho Multiclass

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Chuẩn bị dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# 1. Logistic Regression với One-vs-Rest
lr_ovr = LogisticRegression(multi_class='ovr', max_iter=1000, random_state=42)
lr_ovr.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred_lr = lr_ovr.predict(X_test)
print("=== LOGISTIC REGRESSION (One-vs-Rest) ===")
print(classification_report(y_test, y_pred_lr, target_names=class_names))

# 2. Logistic Regression với Multinomial
lr_multi = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
lr_multi.fit(X_train, y_train)
y_pred_multi = lr_multi.predict(X_test)

print("\n=== LOGISTIC REGRESSION (Multinomial) ===")
print(classification_report(y_test, y_pred_multi, target_names=class_names))
```

### 2.2. Decision Tree và Random Forest

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# Decision Tree
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)

print("=== DECISION TREE ===")
print(classification_report(y_test, y_pred_dt, target_names=class_names))

# Visualize Decision Tree (chỉ hiển thị 3 tầng đầu)
plt.figure(figsize=(15, 8))
tree.plot_tree(dt_classifier, max_depth=3, feature_names=feature_names, 
               class_names=class_names, filled=True, fontsize=8)
plt.title('Decision Tree cho Customer Segmentation')
plt.show()

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

print("\n=== RANDOM FOREST ===")
print(classification_report(y_test, y_pred_rf, target_names=class_names))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance - Random Forest')
plt.show()
```

### 2.3. Support Vector Machine (SVM)

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Chuẩn hóa dữ liệu cho SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM với kernel khác nhau
svm_linear = SVC(kernel='linear', random_state=42)
svm_rbf = SVC(kernel='rbf', random_state=42)

# Training
svm_linear.fit(X_train_scaled, y_train)
svm_rbf.fit(X_train_scaled, y_train)

# Predictions
y_pred_svm_linear = svm_linear.predict(X_test_scaled)
y_pred_svm_rbf = svm_rbf.predict(X_test_scaled)

print("=== SVM LINEAR ===")
print(classification_report(y_test, y_pred_svm_linear, target_names=class_names))

print("\n=== SVM RBF ===")
print(classification_report(y_test, y_pred_svm_rbf, target_names=class_names))
```

### 2.4. k-Nearest Neighbors (kNN)

```python
from sklearn.neighbors import KNeighborsClassifier

# Tìm k tối ưu
k_range = range(1, 21)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

# Plot kết quả
plt.figure(figsize=(10, 5))
plt.plot(k_range, k_scores)
plt.xlabel('Giá trị K')
plt.ylabel('Cross-validation Accuracy')
plt.title('Tìm giá trị K tối ưu cho kNN')
plt.grid(True)
plt.show()

# Chọn k tốt nhất
best_k = k_range[np.argmax(k_scores)]
print(f"K tối ưu: {best_k}")

# Train model với k tốt nhất
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
y_pred_knn = knn_best.predict(X_test_scaled)

print(f"\n=== kNN (k={best_k}) ===")
print(classification_report(y_test, y_pred_knn, target_names=class_names))
```

---

## PHẦN 3: DIỄN GIẢI MICRO VÀ MACRO PERFORMANCE METRICS

### 3.1. Khái niệm cơ bản

**Macro Metrics:** Tính toán metric cho từng lớp riêng biệt, sau đó lấy trung bình.
**Micro Metrics:** Tính toán metric dựa trên tổng số TP, TN, FP, FN của tất cả các lớp.

```python
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

def detailed_multiclass_report(y_true, y_pred, class_names):
    """
    Tạo báo cáo chi tiết về performance metrics cho multiclass classification
    """
    
    # Tính các metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted')
    
    accuracy = accuracy_score(y_true, y_pred)
    
    print("=== CHI TIẾT PERFORMANCE METRICS ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nMACRO AVERAGE (Không trọng số theo số lượng sample):")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall: {recall_macro:.4f}")
    print(f"  F1-Score: {f1_macro:.4f}")
    
    print("\nMICRO AVERAGE (Trọng số theo số lượng sample):")
    print(f"  Precision: {precision_micro:.4f}")
    print(f"  Recall: {recall_micro:.4f}")
    print(f"  F1-Score: {f1_micro:.4f}")
    
    print("\nWEIGHTED AVERAGE (Trọng số theo support của mỗi lớp):")
    print(f"  Precision: {precision_weighted:.4f}")
    print(f"  Recall: {recall_weighted:.4f}")
    print(f"  F1-Score: {f1_weighted:.4f}")
    
    # Confusion Matrix heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'macro': {'precision': precision_macro, 'recall': recall_macro, 'f1': f1_macro},
        'micro': {'precision': precision_micro, 'recall': recall_micro, 'f1': f1_micro},
        'weighted': {'precision': precision_weighted, 'recall': recall_weighted, 'f1': f1_weighted}
    }

# Ví dụ sử dụng
print("PHÂN TÍCH CHI TIẾT - RANDOM FOREST:")
rf_metrics = detailed_multiclass_report(y_test, y_pred_rf, class_names)
```

### 3.2. Giải thích sự khác biệt Micro vs Macro

```python
# Tạo ví dụ minh họa với dữ liệu imbalanced
np.random.seed(42)

# Tạo dữ liệu không cân bằng: 70% lớp 0, 20% lớp 1, 10% lớp 2
y_imbalanced = np.concatenate([
    np.zeros(700),    # 700 samples lớp 0
    np.ones(200),     # 200 samples lớp 1  
    np.full(100, 2)   # 100 samples lớp 2
])

# Giả lập predictions (có bias về lớp đa số)
y_pred_biased = y_imbalanced.copy()
# Làm sai một số predictions
indices_to_change = np.random.choice(len(y_pred_biased), size=150, replace=False)
y_pred_biased[indices_to_change] = np.random.choice(3, size=150)

print("=== SO SÁNH MICRO VS MACRO TRÊN DỮ LIỆU IMBALANCED ===")
detailed_multiclass_report(y_imbalanced, y_pred_biased, ['Lớp 0', 'Lớp 1', 'Lớp 2'])

print("""\n=== GIẢI THÍCH ===
MACRO AVERAGE: 
- Tính precision/recall/f1 cho từng lớp riêng biệt
- Lấy trung bình số học (không quan tâm đến số lượng sample)
- Phù hợp khi muốn các lớp được đối xử công bằng

MICRO AVERAGE:
- Tính tổng TP, FP, FN của tất cả lớp
- Ưu tiên lớp có nhiều sample hơn  
- Phù hợp khi lớp đa số quan trọng hơn

WEIGHTED AVERAGE:
- Tương tự macro nhưng có trọng số theo support
- Cân bằng giữa macro và micro
""")
```

---

## PHẦN 4: KỸ THUẬT SAMPLING CHO DỮ LIỆU IMBALANCED

### 4.1. Tạo dữ liệu imbalanced và phân tích vấn đề

```python
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek
from collections import Counter

# Tạo dữ liệu marketing imbalanced
X_imb, y_imb = make_classification(n_samples=2000, n_features=4, n_classes=4,
                                   weights=[0.7, 0.2, 0.08, 0.02],  # Rất imbalanced
                                   n_informative=3, n_redundant=1,
                                   random_state=42)

print("Phân phối dữ liệu gốc:")
print(Counter(y_imb))

# Chia train/test
X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=42, stratify=y_imb)

print("\nPhân phối training set:")
print(Counter(y_train_imb))

# Train model trên dữ liệu imbalanced
rf_imbalanced = RandomForestClassifier(random_state=42)
rf_imbalanced.fit(X_train_imb, y_train_imb)
y_pred_imb = rf_imbalanced.predict(X_test_imb)

print("\n=== KẾT QUẢ TRÊN DỮ LIỆU IMBALANCED ===")
detailed_multiclass_report(y_test_imb, y_pred_imb, class_names)
```

### 4.2. Over-sampling techniques

```python
# 1. Random Over Sampling
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_train_imb, y_train_imb)
print("Sau Random Over Sampling:")
print(Counter(y_ros))

# 2. SMOTE (Synthetic Minority Oversampling Technique)
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train_imb, y_train_imb)
print("\nSau SMOTE:")
print(Counter(y_smote))

# 3. ADASYN (Adaptive Synthetic Sampling)
adasyn = ADASYN(random_state=42)
X_adasyn, y_adasyn = adasyn.fit_resample(X_train_imb, y_train_imb)
print("\nSau ADASYN:")
print(Counter(y_adasyn))

# So sánh hiệu quả các phương pháp
methods = {
    'Original': (X_train_imb, y_train_imb),
    'Random_OverSampling': (X_ros, y_ros),
    'SMOTE': (X_smote, y_smote),
    'ADASYN': (X_adasyn, y_adasyn)
}

results_comparison = {}

for method_name, (X_method, y_method) in methods.items():
    # Train model
    rf_method = RandomForestClassifier(random_state=42)
    rf_method.fit(X_method, y_method)
    y_pred_method = rf_method.predict(X_test_imb)
    
    # Tính metrics
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_test_imb, y_pred_method, average='macro')
    _, _, f1_micro, _ = precision_recall_fscore_support(
        y_test_imb, y_pred_method, average='micro')
    
    results_comparison[method_name] = {
        'F1_Macro': f1_macro,
        'F1_Micro': f1_micro
    }

# Visualize comparison
comparison_df = pd.DataFrame(results_comparison).T
comparison_df.plot(kind='bar', figsize=(10, 6))
plt.title('So sánh hiệu quả các phương pháp Over-sampling')
plt.ylabel('F1 Score')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

print("\n=== BẢNG SO SÁNH KẾT QUẢ ===")
print(comparison_df)
```

### 4.3. Under-sampling và Combined techniques

```python
# Under-sampling
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train_imb, y_train_imb)

# Tomek Links (loại bỏ outliers)
tomek = TomekLinks()
X_tomek, y_tomek = tomek.fit_resample(X_train_imb, y_train_imb)

# Combined: SMOTE + Tomek Links
smote_tomek = SMOTETomek(random_state=42)
X_combined, y_combined = smote_tomek.fit_resample(X_train_imb, y_train_imb)

print("Random Under Sampling:", Counter(y_rus))
print("Tomek Links:", Counter(y_tomek))  
print("SMOTE + Tomek:", Counter(y_combined))

# Đánh giá combined method
rf_combined = RandomForestClassifier(random_state=42)
rf_combined.fit(X_combined, y_combined)
y_pred_combined = rf_combined.predict(X_test_imb)

print("\n=== KẾT QUẢ SMOTE + TOMEK LINKS ===")
detailed_multiclass_report(y_test_imb, y_pred_combined, class_names)
```

---

## PHẦN 5: ỨNG DỤNG TỔNG HỢP - CHỌN THUẬT TOÁN VÀ METRIC TỐI ưU

### 5.1. Framework đánh giá tổng hợp

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def comprehensive_multiclass_evaluation(X, y, test_size=0.2):
    """
    Framework đánh giá comprehensive cho multiclass classification
    """
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Định nghĩa các algorithms và parameters
    algorithms = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'model__multi_class': ['ovr', 'multinomial'],
                'model__C': [0.1, 1.0, 10.0]
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'model__n_estimators': [50, 100],
                'model__max_depth': [3, 5, None]
            }
        },
        'SVM': {
            'model': SVC(random_state=42),
            'params': {
                'model__kernel': ['linear', 'rbf'],
                'model__C': [0.1, 1.0, 10.0]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'model__n_neighbors': [3, 5, 7, 9],
                'model__weights': ['uniform', 'distance']
            }
        }
    }
    
    # Kết quả cho từng algorithm
    results = {}
    best_models = {}
    
    for alg_name, alg_config in algorithms.items():
        print(f"\n=== ĐÁNH GIÁ {alg_name} ===")
        
        # Tạo pipeline
        if alg_name in ['SVM', 'KNN', 'LogisticRegression']:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', alg_config['model'])
            ])
        else:
            pipeline = Pipeline([
                ('model', alg_config['model'])
            ])
        
        # Grid search
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            pipeline, alg_config['params'], 
            cv=cv, scoring='f1_macro', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model predictions
        y_pred = grid_search.predict(X_test)
        
        # Tính metrics
        metrics = detailed_multiclass_report(y_test, y_pred, class_names)
        
        results[alg_name] = metrics
        best_models[alg_name] = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return results, best_models

# Chạy đánh giá comprehensive
print("=== ĐÁNH GIÁ TỔNG HỢP CÁC THUẬT TOÁN ===")
results, best_models = comprehensive_multiclass_evaluation(X, y)
```

### 5.2. Visualization và so sánh kết quả

```python
# Tạo bảng so sánh
metrics_comparison = []
for alg_name, metrics in results.items():
    metrics_comparison.append({
        'Algorithm': alg_name,
        'Accuracy': metrics['accuracy'],
        'F1_Macro': metrics['macro']['f1'],
        'F1_Micro': metrics['micro']['f1'],
        'F1_Weighted': metrics['weighted']['f1']
    })

comparison_df = pd.DataFrame(metrics_comparison)
print("\n=== BẢNG SO SÁNH TỔNG HỢP ===")
print(comparison_df)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy comparison
axes[0,0].bar(comparison_df['Algorithm'], comparison_df['Accuracy'])
axes[0,0].set_title('Accuracy Comparison')
axes[0,0].tick_params(axis='x', rotation=45)

# F1 Macro comparison  
axes[0,1].bar(comparison_df['Algorithm'], comparison_df['F1_Macro'])
axes[0,1].set_title('F1 Macro Comparison')
axes[0,1].tick_params(axis='x', rotation=45)

# F1 Micro comparison
axes[1,0].bar(comparison_df['Algorithm'], comparison_df['F1_Micro'])
axes[1,0].set_title('F1 Micro Comparison')
axes[1,0].tick_params(axis='x', rotation=45)

# F1 Weighted comparison
axes[1,1].bar(comparison_df['Algorithm'], comparison_df['F1_Weighted'])
axes[1,1].set_title('F1 Weighted Comparison') 
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Tìm best algorithm
best_algorithm = comparison_df.loc[comparison_df['F1_Macro'].idxmax(), 'Algorithm']
print(f"\n=== THUẬT TOÁN TỐT NHẤT (theo F1 Macro): {best_algorithm} ===")
```

---

## BÀI TẬP THỰC HÀNH

### Bài tập 1: Customer Segmentation cho E-commerce

**Mô tả:** Bạn là Data Analyst cho một công ty e-commerce. Công ty muốn phân loại khách hàng thành 4 nhóm dựa trên hành vi mua hàng để thiết kế chiến lược marketing phù hợp.

**Dữ liệu:** 
```python
# Tạo dữ liệu thực hành
np.random.seed(123)
n_customers = 2000

# Features
data = {
    'age': np.random.normal(35, 12, n_customers),
    'annual_income': np.random.normal(50000, 15000, n_customers),
    'spending_score': np.random.normal(50, 20, n_customers),
    'online_hours_per_week': np.random.normal(10, 5, n_customers),
    'num_purchases_last_year': np.random.poisson(15, n_customers),
    'avg_order_value': np.random.normal(75, 25, n_customers)
}

# Tạo target variable (4 segments)
# Logic: kết hợp thu nhập và spending score
segments = []
for i in range(n_customers):
    income_norm = (data['annual_income'][i] - 35000) / 15000
    spending_norm = (data['spending_score'][i] - 30) / 20
    
    score = income_norm + spending_norm
    if score < -0.5:
        segments.append(0)  # Low Value
    elif score < 0:
        segments.append(1)  # Medium Value
    elif score < 0.5:
        segments.append(2)  # High Value
    else:
        segments.append(3)  # Premium

df_exercise = pd.DataFrame(data)
df_exercise['customer_segment'] = segments

# Tạo một số nhiễu để làm dữ liệu thực tế hơn
noise_indices = np.random.choice(n_customers, size=int(0.1 * n_customers), replace=False)
df_exercise.loc[noise_indices, 'customer_segment'] = np.random.choice(4, len(noise_indices))

print("Dataset Customer Segmentation:")
print(df_exercise.head())
print(f"\nShape: {df_exercise.shape}")
print(f"Phân phối segments: \n{df_exercise['customer_segment'].value_counts().sort_index()}")
```

**Yêu cầu:**
1. **Khám phá dữ liệu (EDA):**
   - Visualize phân phối các features và target
   - Tìm correlation giữa các features
   - Phát hiện outliers

```python
# Code template cho học viên
import matplotlib.pyplot as plt
import seaborn as sns

def explore_customer_data(df):
    """
    TODO: Viết hàm khám phá dữ liệu
    - Plot histogram cho từng feature
    - Tạo correlation heatmap
    - Scatter plot matrix colored by segment
    - Box plots để phát hiện outliers
    """
    pass

# Gọi hàm
explore_customer_data(df_exercise)
```

2. **Preprocessing và Feature Engineering:**
```python
def preprocess_data(df):
    """
    TODO: Tiền xử lý dữ liệu
    - Xử lý outliers (IQR method)
    - Tạo feature mới: spending_ratio = avg_order_value / annual_income
    - Standardize features
    - Split train/test với stratification
    """
    pass

X_processed, y_processed = preprocess_data(df_exercise)
```

3. **Model Development và Comparison:**
```python
def compare_algorithms(X, y):
    """
    TODO: So sánh ít nhất 4 thuật toán:
    - Logistic Regression (OvR và Multinomial)
    - Random Forest
    - SVM (Linear và RBF)
    - Gradient Boosting
    
    Sử dung GridSearchCV để tìm hyperparameters tốt nhất
    Đánh giá bằng 5-fold cross-validation
    """
    pass

best_models = compare_algorithms(X_processed, y_processed)
```

4. **Imbalanced Data Handling:**
```python
# Tạo phiên bản imbalanced của dataset
df_imbalanced = create_imbalanced_version(df_exercise)

def handle_imbalanced_data(X, y):
    """
    TODO: Áp dụng và so sánh các kỹ thuật:
    - SMOTE
    - ADASYN  
    - Random Over/Under Sampling
    - SMOTE + Tomek Links
    
    Đánh giá hiệu quả bằng macro F1-score
    """
    pass
```

### Bài tập 2: Marketing Campaign Response Prediction

**Scenario:** Công ty muốn dự đoán phản hồi của khách hàng đối với các loại campaign marketing khác nhau.

```python
# Tạo dữ liệu marketing campaign
np.random.seed(456)
n_samples = 3000

marketing_data = {
    'customer_age': np.random.normal(40, 15, n_samples),
    'income': np.random.normal(55000, 20000, n_samples),
    'education_years': np.random.normal(14, 3, n_samples),
    'family_size': np.random.poisson(3, n_samples),
    'days_since_last_purchase': np.random.exponential(30, n_samples),
    'total_spent_last_year': np.random.gamma(2, 500, n_samples),
    'num_web_visits_last_month': np.random.poisson(8, n_samples),
    'complaint_last_year': np.random.binomial(1, 0.1, n_samples)
}

# Target: 5 loại phản hồi
# 0: No Response, 1: Email Interest, 2: Phone Inquiry, 
# 3: Store Visit, 4: Purchase
campaign_responses = np.random.choice(5, n_samples, 
                                    p=[0.4, 0.25, 0.15, 0.12, 0.08])

df_marketing = pd.DataFrame(marketing_data)
df_marketing['campaign_response'] = campaign_responses

print("Marketing Campaign Dataset:")
print(df_marketing['campaign_response'].value_counts().sort_index())
```

**Nhiệm vụ:**

1. **Advanced EDA với Focus on Business:**
```python
def marketing_eda(df):
    """
    TODO: Phân tích chuyên sâu cho marketing:
    - Customer journey analysis (từ No Response đến Purchase)
    - ROI analysis cho từng response type
    - Feature importance analysis
    - Cohort analysis nếu có thể
    """
    pass
```

2. **Feature Engineering for Marketing:**
```python
def create_marketing_features(df):
    """
    TODO: Tạo các features marketing-specific:
    - customer_lifetime_value = total_spent / days_since_first_purchase
    - engagement_score = web_visits * (1 - complaint_rate) 
    - purchase_frequency = total_spent / avg_order_value
    - recency_score = 1 / (1 + days_since_last_purchase)
    """
    pass
```

3. **Cost-Sensitive Learning:**
```python
# Define business costs
response_costs = {
    0: 0,    # No Response - no cost
    1: -2,   # Email Interest - small cost
    2: -5,   # Phone Inquiry - medium cost
    3: -10,  # Store Visit - high cost
    4: 50    # Purchase - high value
}

def cost_sensitive_evaluation(y_true, y_pred, costs):
    """
    TODO: Implement cost-sensitive evaluation
    Tính toán business value thay vì chỉ accuracy
    """
    pass
```

### Bài tập 3: Advanced Challenge - Multi-Channel Attribution

**Bối cảnh:** Phân tích attribution cho marketing multi-channel để hiểu customer journey.

```python
# Tạo dữ liệu phức tạp hơn
def generate_attribution_data():
    """
    Tạo dữ liệu mô phỏng customer journey qua nhiều touchpoint
    """
    np.random.seed(789)
    n_customers = 5000
    
    # Simulate customer journey
    data = []
    
    for customer_id in range(n_customers):
        # Customer characteristics
        age = np.random.normal(35, 12)
        income = np.random.normal(50000, 15000)
        
        # Journey simulation
        touchpoints = []
        channels = ['social', 'email', 'search', 'display', 'direct']
        
        # Number of touchpoints (1-10)
        n_touchpoints = np.random.poisson(3) + 1
        
        for _ in range(min(n_touchpoints, 10)):
            channel = np.random.choice(channels)
            touchpoints.append(channel)
        
        # Final outcome (5 categories)
        # 0: No conversion, 1: Newsletter signup, 2: Trial, 3: Purchase, 4: Premium
        outcome_prob = min(len(touchpoints) * 0.1 + income/100000, 0.9)
        if np.random.random() < outcome_prob:
            outcome = np.random.choice([1,2,3,4], p=[0.4, 0.3, 0.2, 0.1])
        else:
            outcome = 0
            
        data.append({
            'customer_id': customer_id,
            'age': age,
            'income': income,
            'touchpoints': '|'.join(touchpoints),
            'journey_length': len(touchpoints),
            'outcome': outcome
        })
    
    return pd.DataFrame(data)

df_attribution = generate_attribution_data()
print("Attribution Dataset Sample:")
print(df_attribution.head())
```

**Challenges:**
1. **Sequence Feature Engineering:** Trích xuất features từ customer journey sequence
2. **Imbalanced Multi-class:** Xử lý severe class imbalance
3. **Business Metric Optimization:** Optimize cho business KPIs thay vì accuracy

---

## HƯỚNG DẪN GIẢI BÀI TẬP

### Bài tập 1 - Solution Outline:

```python
# EDA Solution
def explore_customer_data(df):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Histograms for numerical features
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:-1]
    for i, col in enumerate(numeric_cols[:6]):
        ax = axes[i//3, i%3]
        df[col].hist(bins=30, ax=ax)
        ax.set_title(f'Distribution of {col}')
    
    # Correlation heatmap
    axes[2,0].clear()
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, ax=axes[2,0], cmap='coolwarm')
    
    # Segment distribution
    axes[2,1].clear()
    df['customer_segment'].value_counts().plot(kind='bar', ax=axes[2,1])
    axes[2,1].set_title('Customer Segment Distribution')
    
    plt.tight_layout()
    plt.show()

# Preprocessing Solution
def preprocess_data(df):
    # Handle outliers using IQR
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Feature engineering
    df['spending_ratio'] = df['avg_order_value'] / df['annual_income']
    df['purchase_per_hour'] = df['num_purchases_last_year'] / df['online_hours_per_week']
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col != 'customer_segment']
    X = df[feature_cols].values
    y = df['customer_segment'].values
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
```

---

## ĐÁNH GIÁ VÀ RUBRIC

### Tiêu chí đánh giá (100 điểm):

1. **Hiểu biết lý thuyết (25 điểm):**
   - Giải thích được sự khác biệt OvR vs OvO vs Multinomial
   - Phân biệt được micro vs macro metrics
   - Hiểu nguyên lý các sampling techniques

2. **Kỹ năng implementation (30 điểm):**
   - Code chạy được và cho kết quả đúng
   - Sử dụng scikit-learn API một cách thành thạo
   - Xử lý dữ liệu và preprocessing hợp lý

3. **Phân tích và diễn giải (25 điểm):**
   - Giải thích kết quả một cách có ý nghĩa business
   - So sánh algorithms dựa trên context cụ thể
   - Đưa ra recommendations hợp lý

4. **Creativity và advanced techniques (20 điểm):**
   - Feature engineering sáng tạo và có ý nghĩa
   - Áp dụng advanced sampling techniques
   - Cost-sensitive learning hoặc business-oriented metrics

---

## TÀI LIỆU THAM KHẢO

1. **Scikit-learn Documentation:** 
   - [Multiclass Classification](https://scikit-learn.org/stable/modules/multiclass.html)
   - [Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

2. **Imbalanced-learn Documentation:**
   - [Sampling Techniques](https://imbalanced-learn.org/stable/user_guide.html)

3. **Books:**
   - "Hands-On Machine Learning" - Aurélien Géron (Chapter 3)
   - "Pattern Recognition and Machine Learning" - Christopher Bishop

4. **Papers:**
   - "SMOTE: Synthetic Minority Oversampling Technique" - Chawla et al.
   - "ADASYN: Adaptive Synthetic Sampling Approach" - He et al.

---

## KẾT LUẬN

Bài học này đã cung cấp một framework hoàn chỉnh để giải quyết bài toán multiclass classification trong marketing analytics. Các kiến thức chính bao gồm:

- **Nhận dạng và tiếp cận** bài toán multiclass với các chiến lược OvR, OvO
- **Triển khai** các thuật toán chính sử dụng scikit-learn
- **Diễn giải** performance metrics phù hợp với từng context business
- **Xử lý** dữ liệu imbalanced bằng các kỹ thuật sampling tiên tiến
- **Áp dụng** framework đánh giá tổng hợp để chọn solution tối ưu

Thành công trong multiclass classification đòi hỏi không chỉ hiểu biết technical mà còn phải kết hợp với domain knowledge để đưa ra quyết định phù hợp với business context.
