# Chapter 06 More Tools and Techniques for Evaluating Regression Models

## Mục tiêu học tập
Sau khi hoàn thành bài học này, học viên sẽ có khả năng:
- Hiểu và áp dụng các thước đo đánh giá độ chính xác của mô hình hồi quy
- Tính toán và giải thích MAE và RMSE
- Sử dụng Recursive Feature Elimination (RFE) để lựa chọn đặc trưng
- So sánh và đánh giá các mô hình hồi quy dựa trên cây quyết định
- Lựa chọn mô hình phù hợp nhất cho bài toán cụ thể

---

## Phần 1: Tổng quan về Regression và các thước đo đánh giá

### 1.1 Regression - Phương pháp học có giám sát

**Regression** là một kỹ thuật học máy có giám sát được sử dụng để dự đoán các giá trị liên tục (continuous outcomes). Khác với classification dự đoán các lớp rời rạc, regression dự đoán các giá trị số thực.

**Ví dụ về bài toán regression:**
- Dự đoán giá nhà dựa trên diện tích, vị trí, số phòng
- Dự đoán doanh thu công ty dựa trên chi phí marketing
- Dự đoán thói quen chi tiêu của khách hàng dựa trên tuổi tác

### 1.2 Tầm quan trọng của việc đánh giá mô hình

Việc đánh giá độ chính xác của mô hình regression là cực kỳ quan trọng vì:
- Giúp lựa chọn mô hình tốt nhất
- Xác định mô hình có khả năng tổng quát hóa tốt
- Phát hiện overfitting hoặc underfitting
- Đưa ra quyết định kinh doanh dựa trên độ tin cậy của mô hình

---

## Phần 2: Các thước đo đánh giá độ chính xác - MAE và RMSE

### 2.1 Mean Absolute Error (MAE)

**Định nghĩa:** MAE là trung bình của giá trị tuyệt đối của các sai số giữa giá trị dự đoán và giá trị thực tế.

**Công thức:**
```
MAE = (1/n) × Σ|yi - ŷi|
```

Trong đó:
- n: số lượng mẫu
- yi: giá trị thực tế thứ i
- ŷi: giá trị dự đoán thứ i

**Ưu điểm:**
- Dễ hiểu và giải thích
- Không bị ảnh hưởng quá mức bởi outliers
- Có cùng đơn vị với biến phụ thuộc

**Nhược điểm:**
- Không phân biệt được sai số lớn và nhỏ
- Không khả vi tại điểm 0

### 2.2 Root Mean Squared Error (RMSE)

**Định nghĩa:** RMSE là căn bậc hai của trung bình bình phương các sai số.

**Công thức:**
```
RMSE = √[(1/n) × Σ(yi - ŷi)²]
```

**Ưu điểm:**
- Phạt nặng các sai số lớn
- Khả vi và có thể sử dụng trong gradient descent
- Có cùng đơn vị với biến phụ thuộc

**Nhược điểm:**
- Nhạy cảm với outliers
- Khó giải thích hơn MAE

### 2.3 So sánh MAE và RMSE

| Tiêu chí | MAE | RMSE |
|----------|-----|------|
| Nhạy cảm với outliers | Thấp | Cao |
| Độ khó giải thích | Dễ | Trung bình |
| Sử dụng trong tối ưu hóa | Hạn chế | Tốt |
| Giá trị | Luôn ≤ RMSE | Luôn ≥ MAE |

### 2.4 Code Python tính MAE và RMSE

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Ví dụ dữ liệu
y_true = np.array([100, 120, 90, 110, 95, 105, 115, 85])
y_pred = np.array([98, 125, 87, 108, 92, 107, 118, 88])

# Tính MAE
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.2f}")

# Tính RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse:.2f}")

# Tính thủ công để hiểu rõ hơn
mae_manual = np.mean(np.abs(y_true - y_pred))
rmse_manual = np.sqrt(np.mean((y_true - y_pred)**2))

print(f"MAE (manual): {mae_manual:.2f}")
print(f"RMSE (manual): {rmse_manual:.2f}")

# Visualize errors
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_true)), y_true, label='True values', alpha=0.7)
plt.scatter(range(len(y_pred)), y_pred, label='Predicted values', alpha=0.7)
plt.legend()
plt.title('True vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.show()
```

---

## Phần 3: Recursive Feature Elimination (RFE) - Lựa chọn đặc trưng

### 3.1 Khái niệm về Feature Selection

**Feature Selection** là quá trình lựa chọn tập con các đặc trưng quan trọng nhất từ tập đặc trưng ban đầu. Mục đích:
- Giảm độ phức tạp của mô hình
- Tránh overfitting
- Cải thiện hiệu suất dự đoán
- Giảm thời gian training

### 3.2 Recursive Feature Elimination (RFE)

**RFE** là một phương pháp wrapper feature selection hoạt động theo cơ chế:

1. **Bước 1:** Training mô hình với tất cả đặc trưng
2. **Bước 2:** Xếp hạng đặc trưng dựa trên tầm quan trọng
3. **Bước 3:** Loại bỏ đặc trưng ít quan trọng nhất
4. **Bước 4:** Lặp lại cho đến khi đạt số lượng đặc trưng mong muốn

### 3.3 Ưu và nhược điểm của RFE

**Ưu điểm:**
- Xem xét tương tác giữa các đặc trưng
- Hoạt động tốt với linear models
- Có thể chỉ định số lượng đặc trưng cuối cùng

**Nhược điểm:**
- Tốn thời gian tính toán
- Có thể không tối ưu cho tất cả loại mô hình
- Kết quả phụ thuộc vào mô hình base

### 3.4 Code Python sử dụng RFE

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Tạo dữ liệu mẫu về thói quen chi tiêu và tuổi
np.random.seed(42)
n_samples = 1000

# Tạo dữ liệu synthetic
age = np.random.uniform(18, 80, n_samples)
income = np.random.uniform(30000, 150000, n_samples)
education_years = np.random.uniform(10, 20, n_samples)
family_size = np.random.randint(1, 6, n_samples)
city_size = np.random.choice([1, 2, 3], n_samples)  # 1: nhỏ, 2: trung bình, 3: lớn

# Tạo biến phụ thuộc: spending habits (có tương quan với các biến trên)
spending = (age * 50 + income * 0.1 + education_years * 200 + 
           family_size * 1000 + city_size * 500 + 
           np.random.normal(0, 2000, n_samples))

# Thêm một số đặc trưng nhiễu (không liên quan)
noise_features = np.random.randn(n_samples, 3)

# Tạo DataFrame
data = pd.DataFrame({
    'age': age,
    'income': income,
    'education_years': education_years,
    'family_size': family_size,
    'city_size': city_size,
    'noise1': noise_features[:, 0],
    'noise2': noise_features[:, 1],
    'noise3': noise_features[:, 2],
    'spending': spending
})

print("Dữ liệu mẫu:")
print(data.head())
print("\nThông tin dataset:")
print(data.info())

# Tách features và target
X = data.drop('spending', axis=1)
y = data['spending']

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Áp dụng RFE
estimator = LinearRegression()
rfe = RFE(estimator=estimator, n_features_to_select=5, step=1)
rfe.fit(X_train_scaled, y_train)

# Xem kết quả RFE
feature_ranking = pd.DataFrame({
    'Feature': X.columns,
    'Selected': rfe.support_,
    'Ranking': rfe.ranking_
}).sort_values('Ranking')

print("\nKết quả RFE:")
print(feature_ranking)

# Lấy các đặc trưng được chọn
selected_features = X.columns[rfe.support_]
print(f"\nCác đặc trưng được chọn: {list(selected_features)}")

# Training mô hình với đặc trưng được chọn
X_train_selected = rfe.transform(X_train_scaled)
X_test_selected = rfe.transform(X_test_scaled)

model_rfe = LinearRegression()
model_rfe.fit(X_train_selected, y_train)

# Đánh giá mô hình
y_pred_rfe = model_rfe.predict(X_test_selected)

mae_rfe = mean_absolute_error(y_test, y_pred_rfe)
rmse_rfe = np.sqrt(mean_squared_error(y_test, y_pred_rfe))

print(f"\nĐánh giá mô hình RFE:")
print(f"MAE: {mae_rfe:.2f}")
print(f"RMSE: {rmse_rfe:.2f}")

# So sánh với mô hình sử dụng tất cả đặc trưng
model_all = LinearRegression()
model_all.fit(X_train_scaled, y_train)
y_pred_all = model_all.predict(X_test_scaled)

mae_all = mean_absolute_error(y_test, y_pred_all)
rmse_all = np.sqrt(mean_squared_error(y_test, y_pred_all))

print(f"\nĐánh giá mô hình sử dụng tất cả đặc trưng:")
print(f"MAE: {mae_all:.2f}")
print(f"RMSE: {rmse_all:.2f}")
```

---

## Phần 4: Tree-based Regression Models

### 4.1 Decision Tree Regression

**Decision Tree Regression** chia không gian đầu vào thành các vùng và dự đoán giá trị trung bình của target trong mỗi vùng.

**Ưu điểm:**
- Dễ hiểu và giải thích
- Không cần chuẩn hóa dữ liệu
- Xử lý được cả numerical và categorical features
- Tự động feature selection

**Nhược điểm:**
- Dễ overfitting
- Không ổn định (nhạy cảm với thay đổi nhỏ trong dữ liệu)
- Bias với categorical features có nhiều categories

### 4.2 Random Forest Regression

**Random Forest** là ensemble method kết hợp nhiều decision trees.

**Cơ chế hoạt động:**
1. Tạo nhiều decision trees từ các bootstrap samples
2. Mỗi tree chỉ sử dụng subset ngẫu nhiên của features
3. Dự đoán cuối cùng = trung bình dự đoán của các trees

**Ưu điểm:**
- Giảm overfitting so với single tree
- Ổn định hơn decision tree
- Có thể đánh giá feature importance
- Hoạt động tốt với dữ liệu lớn

**Nhược điểm:**
- Khó giải thích hơn single tree
- Có thể overfit với noisy data
- Tốn nhiều memory và computation

### 4.3 Code Python so sánh các mô hình Tree-based

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Sử dụng dữ liệu từ phần trước
# Không cần scaling cho tree-based models

# 1. Decision Tree Regression
print("=== DECISION TREE REGRESSION ===")

# Hyperparameter tuning cho Decision Tree
dt_params = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

dt_grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    dt_params,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_

print(f"Best parameters: {dt_grid.best_params_}")

# Đánh giá Decision Tree
y_pred_dt = best_dt.predict(X_test)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))

print(f"Decision Tree - MAE: {mae_dt:.2f}, RMSE: {rmse_dt:.2f}")

# 2. Random Forest Regression
print("\n=== RANDOM FOREST REGRESSION ===")

# Hyperparameter tuning cho Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

print(f"Best parameters: {rf_grid.best_params_}")

# Đánh giá Random Forest
y_pred_rf = best_rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"Random Forest - MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")

# 3. So sánh Feature Importance
plt.figure(figsize=(15, 5))

# Decision Tree Feature Importance
plt.subplot(1, 2, 1)
dt_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_dt.feature_importances_
}).sort_values('Importance', ascending=True)

plt.barh(dt_importance['Feature'], dt_importance['Importance'])
plt.title('Decision Tree - Feature Importance')
plt.xlabel('Importance')

# Random Forest Feature Importance
plt.subplot(1, 2, 2)
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=True)

plt.barh(rf_importance['Feature'], rf_importance['Importance'])
plt.title('Random Forest - Feature Importance')
plt.xlabel('Importance')

plt.tight_layout()
plt.show()

# 4. So sánh tất cả mô hình
models_comparison = pd.DataFrame({
    'Model': ['Linear Regression (All Features)', 'Linear Regression (RFE)', 
              'Decision Tree', 'Random Forest'],
    'MAE': [mae_all, mae_rfe, mae_dt, mae_rf],
    'RMSE': [rmse_all, rmse_rfe, rmse_dt, rmse_rf]
})

print("\n=== SO SÁNH TẤT CẢ MÔ HÌNH ===")
print(models_comparison)

# Visualization so sánh
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# MAE comparison
ax1.bar(models_comparison['Model'], models_comparison['MAE'])
ax1.set_title('Mean Absolute Error Comparison')
ax1.set_ylabel('MAE')
ax1.tick_params(axis='x', rotation=45)

# RMSE comparison
ax2.bar(models_comparison['Model'], models_comparison['RMSE'])
ax2.set_title('Root Mean Squared Error Comparison')
ax2.set_ylabel('RMSE')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 5. Residual Analysis
def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 4))
    
    # Residuals vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{model_name} - Residuals vs Predicted')
    
    # Residuals histogram
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'{model_name} - Residuals Distribution')
    
    plt.tight_layout()
    plt.show()

# Phân tích residuals cho mô hình tốt nhất
best_model_idx = models_comparison['RMSE'].idxmin()
best_model_name = models_comparison.loc[best_model_idx, 'Model']

print(f"\nMô hình tốt nhất: {best_model_name}")

if 'Random Forest' in best_model_name:
    plot_residuals(y_test, y_pred_rf, 'Random Forest')
elif 'Decision Tree' in best_model_name:
    plot_residuals(y_test, y_pred_dt, 'Decision Tree')
```

---

## Phần 5: Lựa chọn mô hình tối ưu

### 5.1 Tiêu chí lựa chọn mô hình

Khi lựa chọn mô hình regression, cần xem xét:

1. **Độ chính xác:** MAE, RMSE, R²
2. **Khả năng tổng quát hóa:** Cross-validation scores
3. **Tính giải thích được:** Feature importance, coefficients
4. **Thời gian training và prediction**
5. **Độ phức tạp của mô hình**
6. **Yêu cầu cụ thể của business**

### 5.2 Cross-validation để đánh giá tính ổn định

```python
from sklearn.model_selection import cross_val_score

# Cross-validation cho tất cả mô hình
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': best_dt,
    'Random Forest': best_rf
}

cv_results = {}

for name, model in models.items():
    # Sử dụng dữ liệu phù hợp cho từng mô hình
    if name == 'Linear Regression':
        X_cv = X_train_scaled
    else:
        X_cv = X_train
    
    cv_scores = cross_val_score(
        model, X_cv, y_train, 
        cv=5, scoring='neg_mean_squared_error'
    )
    
    cv_results[name] = {
        'mean_rmse': np.sqrt(-cv_scores.mean()),
        'std_rmse': np.sqrt(cv_scores.std())
    }

print("=== CROSS-VALIDATION RESULTS ===")
cv_df = pd.DataFrame(cv_results).T
print(cv_df)
```

### 5.3 Business considerations

```python
# Tạo báo cáo tổng hợp
def create_model_report():
    report = """
    === BÁO CÁO LỰA CHỌN MÔ HÌNH ===
    
    1. MÔ HÌNH TUYẾN TÍNH (Linear Regression):
       - Ưu điểm: Đơn giản, dễ giải thích, training nhanh
       - Nhược điểm: Giả định tuyến tính, có thể underfitting
       - Phù hợp: Khi cần giải thích mối quan hệ tuyến tính
    
    2. CÂY QUYẾT ĐỊNH (Decision Tree):
       - Ưu điểm: Dễ hiểu, không cần scaling, xử lý non-linear
       - Nhược điểm: Dễ overfitting, không ổn định
       - Phù hợp: Khi cần mô hình có thể giải thích từng bước quyết định
    
    3. RANDOM FOREST:
       - Ưu điểm: Ổn định, xử lý tốt overfitting, feature importance
       - Nhược điểm: Khó giải thích, tốn resources
       - Phù hợp: Khi ưu tiên độ chính xác cao và có đủ computational resources
    """
    return report

print(create_model_report())
```

---

## Bài tập thực hành

### Bài tập 1: Dự đoán giá nhà
```python
"""
Bài tập 1: Sử dụng Boston Housing dataset
1. Load và khám phá dữ liệu
2. Tính MAE và RMSE cho Linear Regression
3. Áp dụng RFE để lựa chọn 5 đặc trưng tốt nhất
4. So sánh Decision Tree và Random Forest
5. Chọn mô hình tốt nhất và giải thích lý do
"""

from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings('ignore')

# Load dữ liệu
boston = load_boston()
X_boston = pd.DataFrame(boston.data, columns=boston.feature_names)
y_boston = boston.target

print("Bài tập 1: Hoàn thành các bước sau")
print("1. Khám phá dữ liệu Boston Housing")
print("2. Chia train/test (80/20)")
print("3. Áp dụng các kỹ thuật đã học")
print("4. So sánh và lựa chọn mô hình tối ưu")
```

### Bài tập 2: Feature Engineering và Model Comparison
```python
"""
Bài tập 2: Tạo dữ liệu tùy chỉnh
1. Tạo dataset với 1000 mẫu, 10 features (5 relevant, 5 noise)
2. Sử dụng RFE với các mô hình khác nhau làm estimator
3. So sánh hiệu quả của RFE với Linear Regression và Random Forest
4. Phân tích feature importance và ranking
"""

print("Bài tập 2: Feature Engineering Challenge")
print("Tạo một pipeline hoàn chỉnh từ raw data đến model selection")
```

### Bài tập 3: Real-world Application
```python
"""
Bài tập 3: Ứng dụng thực tế
Scenario: Bạn là data scientist tại một công ty e-commerce
Nhiệm vụ: Dự đoán giá trị đơn hàng của khách hàng

Features có sẵn:
- Tuổi khách hàng
- Số lượng sản phẩm đã xem
- Thời gian trên website
- Lịch sử mua hàng
- Loại membership

Yêu cầu:
1. Thiết kế pipeline đánh giá mô hình
2. Sử dụng tất cả kỹ thuật đã học
3. Đưa ra khuyến nghị business
4. Tạo visualization cho stakeholders
"""

print("Bài tập 3: E-commerce Order Value Prediction")
print("Áp dụng tất cả kiến thức vào bài toán thực tế")
```

---

## Tóm tắt và Checkpoints

### Kiến thức cốt lõi cần nắm vững:

✅ **MAE vs RMSE:**
- MAE: Dễ giải thích, ít bị ảnh hưởng bởi outliers
- RMSE: Phạt nặng sai số lớn, phù hợp cho optimization

✅ **RFE (Recursive Feature Elimination):**
- Phương pháp wrapper feature selection
- Hoạt động tốt với linear models
- Cần cân nhắc computational cost

✅ **Tree-based Models:**
- Decision Tree: Dễ giải thích nhưng dễ overfit
- Random Forest: Ổn định hơn, tốt cho production

✅ **Model Selection Process:**
1. Define evaluation metrics
2. Cross-validation for robustness
3. Compare multiple models
4. Consider business requirements
5. Monitor model performance

### Câu hỏi ôn tập:

1. Khi nào nên sử dụng MAE thay vì RMSE?
2. RFE hoạt động như thế nào và khi nào nên sử dụng?
3. So sánh ưu nhược điểm của Decision Tree và Random Forest
4. Làm thế nào để chọn mô hình tốt nhất cho một bài toán cụ thể?
5. Tại sao cần cross-validation trong model evaluation?

---

## Tài liệu tham khảo và đọc thêm

1. **Scikit-learn Documentation:** Feature Selection và Model Evaluation
2. **"Hands-On Machine Learning" by Aurélien Géron:** Chapters on Regression
3. **"The Elements of Statistical Learning":** Mathematical foundations
4. **Kaggle Learn:** Practical exercises và competitions
5. **Towards Data Science:** Advanced techniques và case studies

---

*Lưu ý cho giáo viên: Bài học này được thiết kế để học viên thực hành song song với lý thuyết. Khuyến khích học viên chạy code và thử nghiệm với parameters khác nhau để hiểu sâu hơn về từng kỹ thuật.*
