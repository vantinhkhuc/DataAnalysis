# Chapter 03: Unsupervised Learning and Customer Segmentation

## Mục tiêu học tập
Sau khi hoàn thành bài học này, học viên sẽ có thể:
- Hiểu rõ nhu cầu và tầm quan trọng của customer segmentation
- Nắm vững thuật toán K-means và ứng dụng trong phân khúc khách hàng
- Thực hiện phân tích thống kê mô tả và tổng hợp dữ liệu
- Sử dụng các công cụ Python để thực hiện segmentation
- Phân tích và diễn giải kết quả phân khúc khách hàng
- Áp dụng các kỹ thuật nâng cao trong customer segmentation

---

## Phần 1: Understanding the Need for Customer Segmentation

### 1.1 Customer Segmentation là gì?

Customer Segmentation (Phân khúc khách hàng) là quá trình chia nhóm khách hàng thành các phân khúc nhỏ hơn dựa trên các đặc điểm chung như:
- **Hành vi mua sắm** (frequency, monetary, recency)
- **Đặc điểm nhân khẩu học** (tuổi, giới tính, thu nhập)
- **Địa lý** (khu vực, thành phố)
- **Tâm lý** (sở thích, giá trị, thái độ)

### 1.2 Tại sao Customer Segmentation quan trọng?

**Lợi ích cho doanh nghiệp:**
1. **Cá nhân hóa marketing**: Tạo chiến dịch phù hợp với từng nhóm khách hàng
2. **Tối ưu hóa sản phẩm**: Phát triển sản phẩm đáp ứng nhu cầu cụ thể
3. **Tăng ROI**: Tập trung nguồn lực vào các khách hàng có giá trị cao
4. **Giảm churn rate**: Xác định khách hàng có nguy cơ rời bỏ
5. **Pricing strategy**: Định giá phù hợp cho từng phân khúc

**Ví dụ thực tế:**
- **Netflix**: Phân khúc người dùng dựa trên thể loại phim yêu thích
- **Amazon**: Gợi ý sản phẩm dựa trên lịch sử mua hàng
- **Spotify**: Tạo playlist cá nhân hóa

### 1.3 Các phương pháp Segmentation truyền thống vs Machine Learning

**Phương pháp truyền thống:**
- Dựa trên kinh nghiệm và trực giác
- Phân chia theo các tiêu chí đơn giản (tuổi, giới tính)
- Hạn chế về khả năng xử lý dữ liệu lớn

**Phương pháp Machine Learning:**
- Tự động phát hiện pattern trong dữ liệu
- Xử lý được nhiều chiều dữ liệu
- Khách quan và dựa trên dữ liệu
- Có thể cập nhật tự động khi có dữ liệu mới

---

## Phần 2: Machine Learning Approach to Segmentation

### 2.1 Unsupervised Learning trong Customer Segmentation

**Định nghĩa**: Unsupervised Learning là phương pháp machine learning không cần nhãn (label) để huấn luyện mô hình.

**Đặc điểm:**
- Không có "đáp án đúng" trước
- Mục tiêu: Tìm ra cấu trúc ẩn trong dữ liệu
- Các thuật toán chính: Clustering, Association Rules, Dimensionality Reduction

### 2.2 Clustering và ứng dụng

**Clustering** là quá trình nhóm các điểm dữ liệu tương tự nhau vào cùng một cụm (cluster).

**Các thuật toán Clustering phổ biến:**
1. **K-Means**: Phân chia dữ liệu thành K cụm
2. **Hierarchical Clustering**: Tạo cây phân cấp các cụm
3. **DBSCAN**: Phát hiện cụm có mật độ cao
4. **Gaussian Mixture Model**: Mô hình hỗn hợp Gaussian

### 2.3 Tại sao chọn K-Means cho Customer Segmentation?

**Ưu điểm:**
- Đơn giản, dễ hiểu và implement
- Hiệu quả với dữ liệu lớn
- Kết quả ổn định và có thể tái lập
- Phù hợp với dữ liệu khách hàng thường có dạng spherical clusters

**Nhược điểm:**
- Cần xác định trước số cụm K
- Nhạy cảm với outliers
- Chỉ tìm được cụm hình tròn

---

## Phần 3: K-Means Clustering Algorithm

### 3.1 Thuật toán K-Means

**Các bước thực hiện:**
1. **Khởi tạo**: Chọn K centroids ngẫu nhiên
2. **Assignment**: Gán mỗi điểm dữ liệu vào cụm có centroid gần nhất
3. **Update**: Cập nhật centroid = trung bình của các điểm trong cụm
4. **Repeat**: Lặp lại bước 2-3 cho đến khi hội tụ

### 3.2 Code Implementation cơ bản

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Thiết lập style cho plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CustomerSegmentation:
    def __init__(self, data):
        """
        Khởi tạo class CustomerSegmentation
        
        Parameters:
        data: DataFrame chứa dữ liệu khách hàng
        """
        self.data = data
        self.scaled_data = None
        self.kmeans = None
        self.labels = None
        
    def preprocess_data(self, features):
        """
        Tiền xử lý dữ liệu: standardization
        
        Parameters:
        features: list tên các cột để sử dụng cho clustering
        """
        # Chọn features
        self.features = features
        self.feature_data = self.data[features].copy()
        
        # Standardization
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(self.feature_data)
        
        print("Dữ liệu đã được chuẩn hóa!")
        print(f"Shape: {self.scaled_data.shape}")
        
    def find_optimal_k(self, k_range=range(2, 11)):
        """
        Tìm số cụm tối ưu bằng Elbow Method và Silhouette Score
        """
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_data)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_data, kmeans.labels_))
        
        # Vẽ biểu đồ
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow Method
        ax1.plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
        ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Số cụm (K)')
        ax1.set_ylabel('Inertia')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette Score
        ax2.plot(k_range, silhouette_scores, marker='s', color='orange', linewidth=2, markersize=8)
        ax2.set_title('Silhouette Score', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Số cụm (K)')
        ax2.set_ylabel('Silhouette Score')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Tìm K tối ưu
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        print(f"K tối ưu theo Silhouette Score: {optimal_k_silhouette}")
        
        return inertias, silhouette_scores
    
    def perform_clustering(self, n_clusters):
        """
        Thực hiện K-means clustering
        """
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(self.scaled_data)
        
        # Thêm labels vào data gốc
        self.data['Cluster'] = self.labels
        
        print(f"Đã phân chia thành {n_clusters} cụm!")
        print(f"Silhouette Score: {silhouette_score(self.scaled_data, self.labels):.3f}")
```

### 3.3 Xác định số cụm tối ưu

**Phương pháp 1: Elbow Method**
```python
def elbow_method_explanation():
    """
    Giải thích Elbow Method
    """
    print("""
    ELBOW METHOD:
    
    1. Tính WCSS (Within-Cluster Sum of Squares) cho các giá trị K khác nhau
    2. Vẽ biểu đồ WCSS theo K
    3. Tìm điểm "gãy" (elbow) - nơi WCSS giảm chậm lại
    4. Điểm elbow chính là K tối ưu
    
    Công thức WCSS: Σ(distance từ điểm đến centroid)²
    """)
```

**Phương pháp 2: Silhouette Score**
```python
def silhouette_explanation():
    """
    Giải thích Silhouette Score
    """
    print("""
    SILHOUETTE SCORE:
    
    1. Đo lường độ tương tự của một điểm với cụm của nó
    2. So sánh với các cụm khác
    3. Giá trị từ -1 đến 1:
       - Gần 1: Phân cụm tốt
       - Gần 0: Điểm nằm trên biên giới
       - Âm: Có thể bị phân sai cụm
       
    Công thức: s = (b - a) / max(a, b)
    Trong đó:
    - a: Khoảng cách trung bình đến các điểm trong cùng cụm
    - b: Khoảng cách trung bình đến cụm gần nhất
    """)
```

---

## Phần 4: Summary and Descriptive Statistics

### 4.1 Các hàm thống kê cơ bản

```python
def explore_data(df):
    """
    Khám phá dữ liệu cơ bản
    """
    print("=" * 50)
    print("THÔNG TIN CƠ BẢN VỀ DỮ LIỆU")
    print("=" * 50)
    
    print(f"1. Kích thước dữ liệu: {df.shape}")
    print(f"2. Số lượng missing values:")
    print(df.isnull().sum())
    
    print(f"\n3. Kiểu dữ liệu:")
    print(df.dtypes)
    
    print(f"\n4. Thống kê mô tả:")
    print(df.describe())

def analyze_clusters(df, features):
    """
    Phân tích chi tiết các cụm
    """
    print("=" * 50)
    print("PHÂN TÍCH CÁC CỤM KHÁCH HÀNG")
    print("=" * 50)
    
    # 1. Kích thước các cụm
    cluster_sizes = df['Cluster'].value_counts().sort_index()
    print("1. Kích thước các cụm:")
    for cluster, size in cluster_sizes.items():
        percentage = (size / len(df)) * 100
        print(f"   Cụm {cluster}: {size} khách hàng ({percentage:.1f}%)")
    
    # 2. Thống kê theo cụm
    print("\n2. Thống kê trung bình theo cụm:")
    cluster_stats = df.groupby('Cluster')[features].mean()
    print(cluster_stats.round(2))
    
    # 3. Đặc điểm nổi bật của từng cụm
    print("\n3. Đặc điểm nổi bật:")
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster][features]
        print(f"\n   Cụm {cluster}:")
        for feature in features:
            mean_val = cluster_data[feature].mean()
            overall_mean = df[feature].mean()
            if mean_val > overall_mean * 1.2:
                print(f"     - {feature}: CAO ({mean_val:.2f} vs {overall_mean:.2f})")
            elif mean_val < overall_mean * 0.8:
                print(f"     - {feature}: THẤP ({mean_val:.2f} vs {overall_mean:.2f})")
            else:
                print(f"     - {feature}: TRUNG BÌNH ({mean_val:.2f})")

def advanced_statistics(df, features):
    """
    Thống kê nâng cao
    """
    print("=" * 50)
    print("THỐNG KÊ NÂNG CAO")
    print("=" * 50)
    
    # Correlation matrix
    print("1. Ma trận tương quan:")
    corr_matrix = df[features].corr()
    print(corr_matrix.round(3))
    
    # Phân phối dữ liệu
    print("\n2. Thông tin phân phối:")
    for feature in features:
        skewness = df[feature].skew()
        kurtosis = df[feature].kurtosis()
        print(f"   {feature}:")
        print(f"     - Skewness: {skewness:.3f}")
        print(f"     - Kurtosis: {kurtosis:.3f}")
```

### 4.2 Visualization Functions

```python
def plot_cluster_analysis(df, features):
    """
    Vẽ biểu đồ phân tích cụm
    """
    n_features = len(features)
    n_clusters = df['Cluster'].nunique()
    
    # 1. Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(features[:4]):  # Chỉ vẽ 4 features đầu
        for cluster in range(n_clusters):
            cluster_data = df[df['Cluster'] == cluster][feature]
            axes[i].hist(cluster_data, alpha=0.6, label=f'Cluster {cluster}', bins=20)
        
        axes[i].set_title(f'Phân phối {feature} theo cụm')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Tần suất')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Box plots
    fig, axes = plt.subplots(1, len(features), figsize=(5*len(features), 6))
    if len(features) == 1:
        axes = [axes]
    
    for i, feature in enumerate(features):
        df.boxplot(column=feature, by='Cluster', ax=axes[i])
        axes[i].set_title(f'Box Plot: {feature}')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel(feature)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Scatter plots (nếu có ít nhất 2 features)
    if len(features) >= 2:
        plt.figure(figsize=(12, 8))
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for cluster in range(n_clusters):
            cluster_data = df[df['Cluster'] == cluster]
            plt.scatter(cluster_data[features[0]], cluster_data[features[1]], 
                       c=colors[cluster % len(colors)], label=f'Cluster {cluster}', alpha=0.7)
        
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title(f'Scatter Plot: {features[0]} vs {features[1]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

def create_cluster_profiles(df, features):
    """
    Tạo hồ sơ chi tiết cho từng cụm
    """
    cluster_profiles = {}
    
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        profile = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df) * 100,
            'characteristics': {}
        }
        
        for feature in features:
            profile['characteristics'][feature] = {
                'mean': cluster_data[feature].mean(),
                'median': cluster_data[feature].median(),
                'std': cluster_data[feature].std(),
                'min': cluster_data[feature].min(),
                'max': cluster_data[feature].max()
            }
        
        cluster_profiles[f'Cluster_{cluster}'] = profile
    
    return cluster_profiles
```

---

## Phần 5: Thực hành với dữ liệu thực tế

### 5.1 Tạo dữ liệu mẫu

```python
def create_sample_data():
    """
    Tạo dữ liệu khách hàng mẫu
    """
    np.random.seed(42)
    n_customers = 1000
    
    # Tạo các nhóm khách hàng khác nhau
    data = []
    
    # Nhóm 1: Khách hàng VIP (20%)
    n_vip = int(n_customers * 0.2)
    vip_data = {
        'CustomerID': range(1, n_vip + 1),
        'Annual_Income': np.random.normal(80000, 15000, n_vip),
        'Spending_Score': np.random.normal(80, 10, n_vip),
        'Age': np.random.normal(40, 8, n_vip),
        'Purchase_Frequency': np.random.normal(15, 3, n_vip)
    }
    
    # Nhóm 2: Khách hàng trung bình (50%)
    n_avg = int(n_customers * 0.5)
    avg_data = {
        'CustomerID': range(n_vip + 1, n_vip + n_avg + 1),
        'Annual_Income': np.random.normal(50000, 12000, n_avg),
        'Spending_Score': np.random.normal(50, 15, n_avg),
        'Age': np.random.normal(35, 10, n_avg),
        'Purchase_Frequency': np.random.normal(8, 2, n_avg)
    }
    
    # Nhóm 3: Khách hàng ít mua sắm (30%)
    n_low = n_customers - n_vip - n_avg
    low_data = {
        'CustomerID': range(n_vip + n_avg + 1, n_customers + 1),
        'Annual_Income': np.random.normal(30000, 8000, n_low),
        'Spending_Score': np.random.normal(25, 8, n_low),
        'Age': np.random.normal(45, 12, n_low),
        'Purchase_Frequency': np.random.normal(3, 1, n_low)
    }
    
    # Kết hợp dữ liệu
    df = pd.DataFrame({
        'CustomerID': list(vip_data['CustomerID']) + list(avg_data['CustomerID']) + list(low_data['CustomerID']),
        'Annual_Income': np.concatenate([vip_data['Annual_Income'], avg_data['Annual_Income'], low_data['Annual_Income']]),
        'Spending_Score': np.concatenate([vip_data['Spending_Score'], avg_data['Spending_Score'], low_data['Spending_Score']]),
        'Age': np.concatenate([vip_data['Age'], avg_data['Age'], low_data['Age']]),
        'Purchase_Frequency': np.concatenate([vip_data['Purchase_Frequency'], avg_data['Purchase_Frequency'], low_data['Purchase_Frequency']])
    })
    
    # Đảm bảo giá trị dương
    df['Annual_Income'] = np.abs(df['Annual_Income'])
    df['Spending_Score'] = np.clip(df['Spending_Score'], 1, 100)
    df['Age'] = np.clip(df['Age'], 18, 80)
    df['Purchase_Frequency'] = np.abs(df['Purchase_Frequency'])
    
    return df

# Tạo và khám phá dữ liệu
df = create_sample_data()
print("Dữ liệu mẫu đã được tạo!")
explore_data(df)
```

### 5.2 Quy trình hoàn chỉnh

```python
def complete_segmentation_workflow():
    """
    Quy trình hoàn chỉnh cho Customer Segmentation
    """
    # Bước 1: Tạo dữ liệu
    print("BƯỚC 1: TẠO DỮ LIỆU")
    df = create_sample_data()
    
    # Bước 2: Khám phá dữ liệu
    print("\nBƯỚC 2: KHÁM PHÁ DỮ LIỆU")
    explore_data(df)
    
    # Bước 3: Chuẩn bị features cho clustering
    print("\nBƯỚC 3: CHUẨN BỊ FEATURES")
    features = ['Annual_Income', 'Spending_Score', 'Age', 'Purchase_Frequency']
    
    # Bước 4: Khởi tạo và tiền xử lý
    print("\nBƯỚC 4: TIỀN XỬ LÝ")
    segmentation = CustomerSegmentation(df)
    segmentation.preprocess_data(features)
    
    # Bước 5: Tìm số cụm tối ưu
    print("\nBƯỚC 5: TÌM SỐ CỤM TỐI ƯU")
    segmentation.find_optimal_k()
    
    # Bước 6: Thực hiện clustering
    print("\nBƯỚC 6: THỰC HIỆN CLUSTERING")
    optimal_k = 3  # Dựa trên kết quả từ bước 5
    segmentation.perform_clustering(optimal_k)
    
    # Bước 7: Phân tích kết quả
    print("\nBƯỚC 7: PHÂN TÍCH KẾT QUẢ")
    analyze_clusters(segmentation.data, features)
    
    # Bước 8: Visualization
    print("\nBƯỚC 8: VISUALIZATION")
    plot_cluster_analysis(segmentation.data, features)
    
    # Bước 9: Tạo cluster profiles
    print("\nBƯỚC 9: TẠO CLUSTER PROFILES")
    profiles = create_cluster_profiles(segmentation.data, features)
    
    return segmentation, profiles

# Chạy quy trình hoàn chỉnh
segmentation_result, cluster_profiles = complete_segmentation_workflow()
```

---

## Phần 6: Các vấn đề nâng cao

### 6.1 Feature Engineering cho Customer Segmentation

```python
def create_rfm_features(df):
    """
    Tạo RFM features (Recency, Frequency, Monetary)
    """
    # Giả sử có dữ liệu transaction
    print("Tạo RFM Features:")
    print("- Recency: Số ngày từ lần mua cuối")
    print("- Frequency: Tần suất mua hàng")  
    print("- Monetary: Giá trị mua hàng")
    
    # Ví dụ code tạo RFM
    rfm_code = """
    # Tính Recency
    df['Recency'] = (datetime.now() - df['Last_Purchase_Date']).dt.days
    
    # Tính Frequency
    df['Frequency'] = df.groupby('CustomerID')['TransactionID'].transform('count')
    
    # Tính Monetary
    df['Monetary'] = df.groupby('CustomerID')['Amount'].transform('sum')
    """
    print(rfm_code)

def handle_categorical_features():
    """
    Xử lý categorical features
    """
    categorical_code = """
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=['Gender', 'City'], prefix=['Gender', 'City'])
    
    # Label encoding
    le = LabelEncoder()
    df['Category_Encoded'] = le.fit_transform(df['Category'])
    """
    print("Xử lý Categorical Features:")
    print(categorical_code)

def dimensionality_reduction():
    """
    Giảm chiều dữ liệu với PCA
    """
    pca_code = """
    from sklearn.decomposition import PCA
    
    # Áp dụng PCA
    pca = PCA(n_components=0.95)  # Giữ 95% variance
    reduced_data = pca.fit_transform(scaled_data)
    
    # Xem explained variance ratio
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    """
    print("Dimensionality Reduction với PCA:")
    print(pca_code)
```

### 6.2 Xử lý Outliers

```python
def detect_and_handle_outliers(df, features):
    """
    Phát hiện và xử lý outliers
    """
    print("PHÁT HIỆN VÀ XỬ LÝ OUTLIERS")
    print("=" * 40)
    
    outlier_indices = []
    
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        feature_outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)].index
        outlier_indices.extend(feature_outliers)
        
        print(f"{feature}:")
        print(f"  - Lower bound: {lower_bound:.2f}")
        print(f"  - Upper bound: {upper_bound:.2f}")
        print(f"  - Số outliers: {len(feature_outliers)}")
    
    # Unique outliers
    unique_outliers = list(set(outlier_indices))
    print(f"\nTổng số outliers duy nhất: {len(unique_outliers)}")
    
    # Các phương pháp xử lý
    print("\nCác phương pháp xử lý outliers:")
    print("1. Loại bỏ outliers")
    print("2. Cap outliers (winsorizing)")
    print("3. Log transformation")
    print("4. Robust scaling")
    
    return unique_outliers

def robust_scaling_example():
    """
    Ví dụ về Robust Scaling
    """
    robust_code = """
    from sklearn.preprocessing import RobustScaler
    
    # Robust Scaler ít bị ảnh hưởng bởi outliers
    robust_scaler = RobustScaler()
    robust_scaled_data = robust_scaler.fit_transform(df[features])
    
    # So sánh với StandardScaler
    from sklearn.preprocessing import StandardScaler
    standard_scaler = StandardScaler()
    standard_scaled_data = standard_scaler.fit_transform(df[features])
    """
    print("Robust Scaling Example:")
    print(robust_code)

### 6.3 Model Validation và Stability

```python
def validate_clustering_stability(df, features, n_runs=10):
    """
    Kiểm tra tính ổn định của clustering
    """
    from sklearn.metrics import adjusted_rand_score
    
    print("KIỂM TRA TÍNH ỔN ĐỊNH CỦA CLUSTERING")
    print("=" * 45)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # Chạy K-means nhiều lần với random seeds khác nhau
    results = []
    base_labels = None
    
    for i in range(n_runs):
        kmeans = KMeans(n_clusters=3, random_state=i, n_init=10)
        labels = kmeans.fit_predict(scaled_data)
        
        if base_labels is None:
            base_labels = labels
        else:
            # Tính ARI với kết quả đầu tiên
            ari_score = adjusted_rand_score(base_labels, labels)
            results.append(ari_score)
    
    avg_ari = np.mean(results)
    std_ari = np.std(results)
    
    print(f"Adjusted Rand Index trung bình: {avg_ari:.3f}")
    print(f"Độ lệch chuẩn: {std_ari:.3f}")
    
    if avg_ari > 0.8:
        print("✓ Clustering rất ổn định")
    elif avg_ari > 0.6:
        print("⚠ Clustering khá ổn định")
    else:
        print("✗ Clustering không ổn định - cần điều chỉnh")
    
    return results
```

---

## Bài Tập Ứng Dụng

### Bài Tập 1: Phân khúc khách hàng E-commerce

**Mô tả**: Bạn là Data Analyst của một công ty thương mại điện tử. Công ty cung cấp cho bạn dữ liệu về hành vi mua sắm của khách hàng và yêu cầu phân khúc để tối ưu hóa chiến lược marketing.

**Dữ liệu**: 
- CustomerID: ID khách hàng
- Age: Tuổi
- Annual_Income: Thu nhập hàng năm (USD)
- Spending_Score: Điểm chi tiêu (1-100)
- Purchase_Amount: Tổng giá trị đơn hàng
- Days_Since_Last_Purchase: Số ngày từ lần mua cuối
- Number_of_Purchases: Tổng số lần mua hàng

```python
def create_ecommerce_dataset():
    """
    Tạo dataset mô phỏng dữ liệu e-commerce
    """
    np.random.seed(123)
    n_customers = 2000
    
    # Tạo 4 nhóm khách hàng khác biệt
    data = []
    
    # Nhóm 1: High Value Customers (15%)
    n_hvc = int(n_customers * 0.15)
    hvc_data = {
        'Age': np.random.normal(40, 8, n_hvc),
        'Annual_Income': np.random.normal(90000, 20000, n_hvc),
        'Spending_Score': np.random.normal(85, 10, n_hvc),
        'Purchase_Amount': np.random.normal(5000, 1000, n_hvc),
        'Days_Since_Last_Purchase': np.random.normal(15, 5, n_hvc),
        'Number_of_Purchases': np.random.normal(25, 5, n_hvc)
    }
    
    # Nhóm 2: Regular Customers (40%)
    n_rc = int(n_customers * 0.4)
    rc_data = {
        'Age': np.random.normal(35, 10, n_rc),
        'Annual_Income': np.random.normal(55000, 15000, n_rc),
        'Spending_Score': np.random.normal(55, 15, n_rc),
        'Purchase_Amount': np.random.normal(2500, 800, n_rc),
        'Days_Since_Last_Purchase': np.random.normal(30, 10, n_rc),
        'Number_of_Purchases': np.random.normal(12, 4, n_rc)
    }
    
    # Nhóm 3: Price Sensitive Customers (30%)
    n_psc = int(n_customers * 0.3)
    psc_data = {
        'Age': np.random.normal(45, 12, n_psc),
        'Annual_Income': np.random.normal(35000, 10000, n_psc),
        'Spending_Score': np.random.normal(30, 12, n_psc),
        'Purchase_Amount': np.random.normal(800, 300, n_psc),
        'Days_Since_Last_Purchase': np.random.normal(60, 20, n_psc),
        'Number_of_Purchases': np.random.normal(6, 2, n_psc)
    }
    
    # Nhóm 4: Inactive Customers (15%)
    n_ic = n_customers - n_hvc - n_rc - n_psc
    ic_data = {
        'Age': np.random.normal(50, 15, n_ic),
        'Annual_Income': np.random.normal(45000, 12000, n_ic),
        'Spending_Score': np.random.normal(20, 8, n_ic),
        'Purchase_Amount': np.random.normal(500, 200, n_ic),
        'Days_Since_Last_Purchase': np.random.normal(120, 30, n_ic),
        'Number_of_Purchases': np.random.normal(2, 1, n_ic)
    }
    
    # Kết hợp tất cả dữ liệu
    all_data = {}
    for key in hvc_data.keys():
        all_data[key] = np.concatenate([
            hvc_data[key], rc_data[key], psc_data[key], ic_data[key]
        ])
    
    df = pd.DataFrame(all_data)
    df['CustomerID'] = range(1, len(df) + 1)
    
    # Đảm bảo giá trị hợp lệ
    df['Age'] = np.clip(df['Age'], 18, 80)
    df['Annual_Income'] = np.abs(df['Annual_Income'])
    df['Spending_Score'] = np.clip(df['Spending_Score'], 1, 100)
    df['Purchase_Amount'] = np.abs(df['Purchase_Amount'])
    df['Days_Since_Last_Purchase'] = np.abs(df['Days_Since_Last_Purchase'])
    df['Number_of_Purchases'] = np.abs(df['Number_of_Purchases'])
    
    return df

# Yêu cầu bài tập
def exercise_1_requirements():
    """
    Yêu cầu chi tiết cho bài tập 1
    """
    print("""
    BÀI TẬP 1: PHÂN KHÚC KHÁCH HÀNG E-COMMERCE
    ==========================================
    
    NHIỆM VỤ:
    1. Tải và khám phá dữ liệu
    2. Tiền xử lý dữ liệu (xử lý outliers, standardization)
    3. Thực hiện EDA (Exploratory Data Analysis)
    4. Tìm số cụm tối ưu bằng cả Elbow Method và Silhouette Score
    5. Áp dụng K-means clustering
    6. Phân tích và diễn giải kết quả
    7. Tạo customer personas cho từng segment
    8. Đề xuất chiến lược marketing cho từng nhóm
    
    KẾT QUẢ MONG MUỐN:
    - Report chi tiết về đặc điểm từng segment
    - Visualization hấp dẫn
    - Actionable insights cho business
    - Code có comment đầy đủ
    
    TIÊU CHÍ ĐÁNH GIÁ:
    - Chất lượng phân tích dữ liệu (25%)
    - Tính đúng đắn của clustering (25%)
    - Chất lượng visualization (20%)
    - Business insights (20%)
    - Code quality (10%)
    """)

# Template code cho học viên
exercise_1_template = '''
# BÀI TẬP 1: E-COMMERCE CUSTOMER SEGMENTATION
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Bước 1: Tạo dữ liệu
df_ecommerce = create_ecommerce_dataset()

# Bước 2: EDA - Khám phá dữ liệu
# TODO: Thực hiện EDA chi tiết
print("=== KHÁM PHÁ DỮ LIỆU ===")
# Your code here

# Bước 3: Tiền xử lý
# TODO: Xử lý outliers và chuẩn hóa dữ liệu
features = ['Age', 'Annual_Income', 'Spending_Score', 'Purchase_Amount', 
           'Days_Since_Last_Purchase', 'Number_of_Purchases']
# Your code here

# Bước 4: Tìm K tối ưu
# TODO: Implement Elbow Method và Silhouette Analysis
# Your code here

# Bước 5: Clustering
# TODO: Áp dụng K-means với K tối ưu
# Your code here

# Bước 6: Phân tích kết quả
# TODO: Tạo cluster profiles và visualization
# Your code here

# Bước 7: Business Insights
# TODO: Đưa ra insights và recommendations
# Your code here
'''
```

### Bài Tập 2: Phân khúc khách hàng ngân hàng với RFM Analysis

**Mô tả**: Bạn làm việc cho một ngân hàng và được yêu cầu phân tích hành vi giao dịch của khách hàng để phát triển các sản phẩm tài chính phù hợp.

```python
def create_banking_dataset():
    """
    Tạo dataset mô phỏng dữ liệu ngân hàng
    """
    np.random.seed(456)
    n_customers = 1500
    
    # Tạo transaction data
    customers = []
    
    for customer_id in range(1, n_customers + 1):
        # Random customer characteristics
        age = np.random.randint(25, 70)
        account_balance = np.random.lognormal(8, 1.5)  # Log-normal distribution
        
        # Generate transaction patterns based on customer type
        if np.random.random() < 0.2:  # 20% high-value customers
            n_transactions = np.random.randint(50, 200)
            avg_transaction = np.random.normal(5000, 2000)
            recency = np.random.randint(1, 7)
        elif np.random.random() < 0.5:  # 30% regular customers  
            n_transactions = np.random.randint(20, 80)
            avg_transaction = np.random.normal(1500, 800)
            recency = np.random.randint(3, 15)
        else:  # 50% low-activity customers
            n_transactions = np.random.randint(5, 30)
            avg_transaction = np.random.normal(500, 300)
            recency = np.random.randint(10, 60)
        
        # Ensure positive values
        avg_transaction = max(100, avg_transaction)
        total_transaction_value = n_transactions * avg_transaction
        
        customers.append({
            'CustomerID': customer_id,
            'Age': age,
            'Account_Balance': max(0, account_balance),
            'Total_Transaction_Value': total_transaction_value,
            'Transaction_Frequency': n_transactions,
            'Avg_Transaction_Amount': avg_transaction,
            'Days_Since_Last_Transaction': recency,
            'Credit_Score': np.random.randint(300, 850),
            'Number_of_Products': np.random.randint(1, 8)
        })
    
    return pd.DataFrame(customers)

def create_rfm_features_banking(df):
    """
    Tạo RFM features cho dữ liệu ngân hàng
    """
    df_rfm = df.copy()
    
    # R - Recency (Days since last transaction)
    df_rfm['Recency'] = df_rfm['Days_Since_Last_Transaction']
    
    # F - Frequency (Number of transactions)
    df_rfm['Frequency'] = df_rfm['Transaction_Frequency']
    
    # M - Monetary (Total transaction value)
    df_rfm['Monetary'] = df_rfm['Total_Transaction_Value']
    
    # Additional banking-specific features
    df_rfm['CLV'] = df_rfm['Monetary'] / df_rfm['Age']  # Customer Lifetime Value proxy
    df_rfm['Engagement_Score'] = (df_rfm['Frequency'] * df_rfm['Number_of_Products']) / df_rfm['Recency']
    
    return df_rfm

def exercise_2_requirements():
    """
    Yêu cầu chi tiết cho bài tập 2
    """
    print("""
    BÀI TẬP 2: PHÂN KHÚC KHÁCH HÀNG NGÂN HÀNG VỚI RFM
    ===============================================
    
    BỐI CẢNH:
    Bạn là Data Analyst tại một ngân hàng. Ngân hàng muốn:
    - Xác định khách hàng VIP để tập trung chăm sóc
    - Phát hiện khách hàng có nguy cơ churn
    - Phát triển sản phẩm tài chính phù hợp từng segment
    
    NHIỆM VỤ:
    1. Tạo RFM features từ dữ liệu giao dịch
    2. Thực hiện RFM analysis truyền thống (quintile scoring)
    3. Áp dụng K-means clustering trên RFM data
    4. So sánh kết quả giữa 2 phương pháp
    5. Tạo customer journey mapping
    6. Đề xuất chiến lược retention và cross-selling
    7. Tính toán potential revenue impact
    
    FEATURES CHÍNH:
    - Recency: Số ngày từ giao dịch cuối
    - Frequency: Tần suất giao dịch
    - Monetary: Tổng giá trị giao dịch
    - Account_Balance: Số dư tài khoản
    - Credit_Score: Điểm tín dụng
    - Number_of_Products: Số sản phẩm đang sử dụng
    
    DELIVERABLES:
    - RFM segments với business interpretation
    - Customer personas chi tiết
    - Action plan cho từng segment
    - ROI estimation cho các chiến lược đề xuất
    """)

# Template code
exercise_2_template = '''
# BÀI TẬP 2: BANKING CUSTOMER SEGMENTATION WITH RFM
# ==============================================

# Bước 1: Tạo và khám phá dữ liệu banking
df_banking = create_banking_dataset()

# Bước 2: Tạo RFM features
df_rfm = create_rfm_features_banking(df_banking)

# Bước 3: Traditional RFM Analysis
def traditional_rfm_analysis(df):
    """
    TODO: Implement traditional RFM scoring
    - Chia mỗi RFM thành 5 quintiles (1-5)
    - Tạo RFM score (ví dụ: 555 = top customer)
    - Phân loại customers thành segments
    """
    # Your code here
    pass

# Bước 4: K-means RFM Clustering
def kmeans_rfm_clustering(df):
    """
    TODO: Áp dụng K-means trên RFM features
    - Standardize RFM features
    - Find optimal K
    - Perform clustering
    """
    # Your code here
    pass

# Bước 5: So sánh 2 phương pháp
def compare_methods(df):
    """
    TODO: So sánh traditional RFM vs K-means
    - Visualization comparison
    - Silhouette analysis
    - Business interpretation
    """
    # Your code here
    pass

# Bước 6: Business Strategy
def develop_strategies(df):
    """
    TODO: Phát triển chiến lược cho từng segment
    - Customer personas
    - Product recommendations
    - Marketing strategies
    - Churn prevention plans
    """
    # Your code here
    pass
'''
```

### Bài Tập 3: Multi-dimensional Customer Segmentation cho Retail Chain

**Mô tả**: Bạn là Head of Analytics cho một chuỗi bán lẻ lớn với nhiều cửa hàng trên toàn quốc. Công ty muốn thực hiện segmentation phức tạp kết hợp nhiều chiều dữ liệu.

```python
def create_retail_chain_dataset():
    """
    Tạo dataset phức tạp cho retail chain
    """
    np.random.seed(789)
    n_customers = 3000
    
    # Define store locations và seasonal patterns
    store_locations = ['North', 'South', 'East', 'West', 'Central']
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    product_categories = ['Electronics', 'Clothing', 'Home', 'Sports', 'Beauty']
    
    customers = []
    
    for customer_id in range(1, n_customers + 1):
        # Demographic info
        age = np.random.randint(18, 80)
        gender = np.random.choice(['M', 'F'])
        location = np.random.choice(store_locations)
        
        # Shopping behavior varies by demographics và location
        if age < 30:  # Young customers
            digital_engagement = np.random.normal(0.8, 0.1)
            price_sensitivity = np.random.normal(0.7, 0.1)
            brand_loyalty = np.random.normal(0.4, 0.1)
        elif age < 50:  # Middle-aged customers
            digital_engagement = np.random.normal(0.6, 0.15)
            price_sensitivity = np.random.normal(0.5, 0.15)
            brand_loyalty = np.random.normal(0.6, 0.1)
        else:  # Older customers
            digital_engagement = np.random.normal(0.3, 0.1)
            price_sensitivity = np.random.normal(0.4, 0.1)
            brand_loyalty = np.random.normal(0.8, 0.1)
        
        # Seasonal shopping patterns
        seasonal_spending = {}
        base_spending = np.random.lognormal(6, 1)
        
        for season in seasons:
            if season == 'Winter':  # Holiday season
                seasonal_multiplier = np.random.normal(1.5, 0.3)
            elif season == 'Summer':  # Vacation season
                seasonal_multiplier = np.random.normal(1.2, 0.2)
            else:
                seasonal_multiplier = np.random.normal(1.0, 0.1)
            
            seasonal_spending[season] = max(0, base_spending * seasonal_multiplier)
        
        # Product preferences
        preferred_categories = np.random.choice(product_categories, 
                                              size=np.random.randint(1, 4), 
                                              replace=False)
        
        # Transaction patterns
        avg_basket_size = np.random.normal(3, 1)
        visit_frequency = np.random.normal(15, 5)  # visits per month
        online_vs_offline_ratio = digital_engagement + np.random.normal(0, 0.1)
        
        # Engagement metrics
        email_open_rate = digital_engagement * np.random.normal(1, 0.2)
        social_media_engagement = digital_engagement * np.random.normal(1, 0.3)
        loyalty_program_usage = brand_loyalty * np.random.normal(1, 0.2)
        
        # Ensure values are in valid ranges
        for var in [digital_engagement, price_sensitivity, brand_loyalty, 
                   online_vs_offline_ratio, email_open_rate, 
                   social_media_engagement, loyalty_program_usage]:
            var = max(0, min(1, var))
        
        customers.append({
            'CustomerID': customer_id,
            'Age': age,
            'Gender': gender,
            'Location': location,
            'Annual_Spending': sum(seasonal_spending.values()),
            'Spring_Spending': seasonal_spending['Spring'],
            'Summer_Spending': seasonal_spending['Summer'],
            'Fall_Spending': seasonal_spending['Fall'],
            'Winter_Spending': seasonal_spending['Winter'],
            'Avg_Basket_Size': max(1, avg_basket_size),
            'Visit_Frequency': max(1, visit_frequency),
            'Online_Offline_Ratio': max(0, min(1, online_vs_offline_ratio)),
            'Email_Open_Rate': max(0, min(1, email_open_rate)),
            'Social_Media_Engagement': max(0, min(1, social_media_engagement)),
            'Loyalty_Program_Usage': max(0, min(1, loyalty_program_usage)),
            'Price_Sensitivity': max(0, min(1, price_sensitivity)),
            'Brand_Loyalty': max(0, min(1, brand_loyalty)),
            'Preferred_Categories': ','.join(preferred_categories),
            'Digital_Engagement': max(0, min(1, digital_engagement))
        })
    
    return pd.DataFrame(customers)

def exercise_3_requirements():
    """
    Yêu cầu chi tiết cho bài tập 3
    """
    print("""
    BÀI TẬP 3: MULTI-DIMENSIONAL CUSTOMER SEGMENTATION
    ===============================================
    
    BỐI CẢNH:
    Chuỗi bán lẻ QuickMart có 50 cửa hàng trên toàn quốc với:
    - 3 triệu khách hàng active
    - 5 categories sản phẩm chính  
    - Cả online và offline channels
    - Loyalty program với 60% participation rate
    
    THÁCH THỨC KINH DOANH:
    - Competition từ e-commerce giants
    - Changing consumer behavior post-COVID
    - Need for personalized experience
    - Inventory optimization across locations
    
    NHIỆM VỤ NÂNG CAO:
    1. Feature Engineering:
       - Tạo seasonal indices
       - Channel preference scores
       - Category affinity metrics
       - Geographic clustering
    
    2. Advanced Clustering:
       - Thử multiple algorithms (K-means, Hierarchical, DBSCAN)
       - Ensemble clustering methods
       - Stability analysis across time periods
    
    3. Multi-level Segmentation:
       - Macro segments (behavioral)
       - Micro segments (demographic + geographic)
       - Temporal segments (seasonal patterns)
    
    4. Validation và Business Impact:
       - A/B test design for segment validation
       - Revenue impact modeling
       - Customer satisfaction prediction
    
    5. Advanced Analytics:
       - Customer journey mapping
       - Next-best-action recommendations
       - Churn prediction by segment
       - LTV calculation
    
    EXPECTED OUTPUTS:
    - Comprehensive segmentation strategy
    - Interactive dashboard design
    - Implementation roadmap
    - Success metrics framework
    """)

# Template code cho bài tập phức tạp
exercise_3_template = '''
# BÀI TẬP 3: MULTI-DIMENSIONAL CUSTOMER SEGMENTATION
# ===============================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df_retail = create_retail_chain_dataset()

# PHASE 1: ADVANCED FEATURE ENGINEERING
# =====================================

def create_advanced_features(df):
    """
    TODO: Tạo features nâng cao
    
    1. Seasonal Features:
       - Seasonal spending variation
       - Peak season indicator
       - Spending consistency index
    
    2. Behavioral Features:
       - Channel preference score
       - Engagement composite score
       - Purchase pattern regularity
    
    3. Value Features:
       - Customer lifetime value estimate
       - Profitability score
       - Growth potential index
    """
    # Your advanced feature engineering here
    pass

def geographic_analysis(df):
    """
    TODO: Phân tích địa lý
    - Regional spending patterns
    - Location-based clustering
    - Market penetration analysis
    """
    # Your code here
    pass

# PHASE 2: ENSEMBLE CLUSTERING APPROACH
# ==================================

def ensemble_clustering(df, features):
    """
    TODO: Implement ensemble clustering
    - Multiple algorithms comparison
    - Consensus clustering
    - Stability analysis
    """
    # Your code here
    pass

def hierarchical_clustering_analysis(df, features):
    """
    TODO: Hierarchical clustering với dendrogram
    - Determine optimal number of clusters
    - Analyze cluster hierarchy
    - Compare with K-means results
    """
    # Your code here
    pass

# PHASE 3: MULTI-LEVEL SEGMENTATION
# ==============================

def macro_segmentation(df):
    """
    TODO: Macro-level behavioral segments
    - High-level customer archetypes
    - Cross-segment analysis
    - Segment migration patterns
    """
    # Your code here
    pass

def micro_segmentation(df):
    """
    TODO: Micro-level segments within macro segments
    - Demographic refinement
    - Geographic sub-segments
    - Product preference clusters
    """
    # Your code here
    pass

# PHASE 4: BUSINESS IMPACT ANALYSIS
# ==============================

def calculate_segment_value(df):
    """
    TODO: Tính toán giá trị của từng segment
    - Revenue contribution
    - Profitability analysis
    - Growth potential
    """
    # Your code here
    pass

def develop_action_plans(df):
    """
    TODO: Phát triển action plans
    - Personalization strategies
    - Channel optimization
    - Product recommendations
    - Retention programs
    """
    # Your code here
    pass

# PHASE 5: ADVANCED VISUALIZATION
# ============================

def create_interactive_dashboard():
    """
    TODO: Tạo interactive dashboard concept
    - Segment overview
    - Drill-down capabilities  
    - Real-time updates
    """
    # Your dashboard design here
    pass

def segment_journey_mapping(df):
    """
    TODO: Customer journey mapping by segment
    - Touchpoint analysis
    - Conversion funnels
    - Experience optimization opportunities
    """
    # Your code here
    pass

# RUN COMPLETE ANALYSIS
# ===================
if __name__ == "__main__":
    print("Starting Multi-Dimensional Customer Segmentation Analysis...")
    
    # Execute all phases
    # Phase 1: Feature Engineering
    # Phase 2: Ensemble Clustering  
    # Phase 3: Multi-level Segmentation
    # Phase 4: Business Impact Analysis
    # Phase 5: Visualization and Reporting
'''
```

---

## Hướng dẫn chấm điểm và đánh giá

### Rubric cho các bài tập:

**Bài Tập 1 (Cơ bản)**:
- Code functionality (40%)
- Data analysis quality (30%) 
- Business insights (20%)
- Presentation (10%)

**Bài Tập 2 (Trung bình)**:
- Technical implementation (35%)
- RFM analysis depth (25%)
- Method comparison (20%) 
- Strategic recommendations (20%)

**Bài Tập 3 (Nâng cao)**:
- Advanced techniques usage (30%)
- Feature engineering creativity (25%)
- Business impact analysis (25%)
- Innovation and insights (20%)

### Tiêu chí đánh giá chung:
- **Xuất sắc (90-100%)**: Vượt expectation, có insights độc đáo
- **Tốt (80-89%)**: Hoàn thành tốt tất cả requirements  
- **Khá (70-79%)**: Hoàn thành cơ bản với một số thiếu sót
- **Trung bình (60-69%)**: Hoàn thành một phần, thiếu insights
- **Yếu (<60%)**: Không hoàn thành được yêu cầu cơ bản

---

## Phần 7: Case Studies và Ứng dụng thực tế

### 7.1 Case Study: Netflix Customer Segmentation

```python
def netflix_case_study():
    """
    Case study: Netflix sử dụng clustering như thế nào
    """
    print("""
    NETFLIX CUSTOMER SEGMENTATION STRATEGY
    ====================================
    
    1. DATA SOURCES:
       - Viewing history và duration
       - Rating patterns
       - Search behavior
       - Device usage
       - Time-of-day preferences
       - Genre preferences
       
    2. SEGMENTATION APPROACH:
       - Collaborative filtering + K-means
       - Content-based clustering
       - Temporal pattern analysis
       - Demographic overlays
       
    3. BUSINESS APPLICATIONS:
       - Personalized recommendations
       - Content acquisition decisions
       - UI/UX personalization
       - Marketing campaign targeting
       
    4. RESULTS:
       - 80% of content watched comes from recommendations
       - Increased user engagement by 20%
       - Reduced churn rate by 15%
    """)

def implement_netflix_approach(df):
    """
    Mô phỏng approach của Netflix
    """
    # Tạo viewing patterns
    viewing_features = [
        'total_viewing_hours',
        'avg_session_duration', 
        'genre_diversity_score',
        'binge_watching_tendency',
        'prime_time_ratio',
        'weekend_viewing_ratio'
    ]
    
    clustering_code = """
    # Netflix-style clustering
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Feature engineering for viewing patterns
    df['binge_score'] = df['avg_session_duration'] / df['content_length']
    df['diversity_score'] = df['unique_genres'] / df['total_content']
    df['loyalty_score'] = df['completion_rate'] * df['rating_frequency']
    
    # Multi-layer clustering
    # Layer 1: Behavioral segments
    behavioral_features = ['viewing_frequency', 'session_duration', 'content_diversity']
    behavioral_clusters = KMeans(n_clusters=5).fit_predict(df[behavioral_features])
    
    # Layer 2: Content preference segments  
    content_features = ['action_pref', 'comedy_pref', 'drama_pref', 'documentary_pref']
    content_clusters = KMeans(n_clusters=8).fit_predict(df[content_features])
    
    # Combine segments
    df['behavioral_segment'] = behavioral_clusters
    df['content_segment'] = content_clusters
    df['combined_segment'] = df['behavioral_segment'].astype(str) + '_' + df['content_segment'].astype(str)
    """
    
    print("Netflix-style Implementation:")
    print(clustering_code)
```

### 7.2 Case Study: Amazon Customer Segmentation

```python
def amazon_case_study():
    """
    Case study: Amazon's segmentation for personalization
    """
    print("""
    AMAZON CUSTOMER SEGMENTATION STRATEGY
    ===================================
    
    1. MULTI-DIMENSIONAL APPROACH:
       - Purchase history clustering
       - Browsing behavior analysis
       - Price sensitivity segments
       - Category affinity groups
       - Geographic preferences
       - Seasonal shopping patterns
    
    2. REAL-TIME SEGMENTATION:
       - Session-based clustering
       - Dynamic segment updates
       - Contextual recommendations
       - Cross-device tracking
    
    3. BUSINESS IMPACT:
       - "People who bought X also bought Y"
       - Dynamic pricing strategies
       - Inventory optimization
       - Supply chain efficiency
       
    4. ADVANCED TECHNIQUES:
       - Deep learning embeddings
       - Graph-based clustering
       - Multi-armed bandit algorithms
       - Reinforcement learning
    """)

def amazon_recommendation_system():
    """
    Mô phỏng Amazon recommendation approach
    """
    recommendation_code = """
    # Amazon-style recommendation clustering
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    
    def create_user_item_matrix(df):
        '''Tạo user-item matrix từ purchase history'''
        user_item_matrix = df.pivot_table(
            index='CustomerID', 
            columns='ProductID', 
            values='Rating',
            fill_value=0
        )
        return user_item_matrix
    
    def collaborative_filtering_clusters(user_item_matrix, n_clusters=10):
        '''Clustering users dựa trên collaborative filtering'''
        # Tính user similarity
        user_similarity = cosine_similarity(user_item_matrix)
        
        # K-means trên similarity matrix
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        user_clusters = kmeans.fit_predict(user_similarity)
        
        return user_clusters
    
    def content_based_clusters(df, n_clusters=15):
        '''Clustering sản phẩm dựa trên content'''
        product_features = ['category', 'price_range', 'brand', 'rating']
        # Feature encoding và clustering
        encoded_features = pd.get_dummies(df[product_features])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        product_clusters = kmeans.fit_predict(encoded_features)
        
        return product_clusters
    """
    print("Amazon Recommendation System:")
    print(recommendation_code)
```

### 7.3 Evaluation Metrics nâng cao

```python
def advanced_evaluation_metrics():
    """
    Các metrics nâng cao để đánh giá clustering
    """
    evaluation_code = """
    from sklearn.metrics import (
        silhouette_score, 
        calinski_harabasz_score,
        davies_bouldin_score,
        adjusted_rand_score,
        normalized_mutual_info_score
    )
    
    def comprehensive_evaluation(X, labels, true_labels=None):
        '''Đánh giá toàn diện clustering results'''
        
        results = {}
        
        # Internal validation metrics
        results['silhouette_score'] = silhouette_score(X, labels)
        results['calinski_harabasz'] = calinski_harabasz_score(X, labels)  
        results['davies_bouldin'] = davies_bouldin_score(X, labels)
        
        # External validation (nếu có true labels)
        if true_labels is not None:
            results['adjusted_rand_score'] = adjusted_rand_score(true_labels, labels)
            results['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, labels)
        
        # Business metrics
        results['cluster_sizes'] = np.bincount(labels)
        results['cluster_balance'] = np.std(results['cluster_sizes']) / np.mean(results['cluster_sizes'])
        
        return results
    
    def stability_analysis(X, n_runs=20, n_clusters=5):
        '''Phân tích tính ổn định của clustering'''
        stability_scores = []
        base_labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
        
        for i in range(1, n_runs):
            labels = KMeans(n_clusters=n_clusters, random_state=i).fit_predict(X)
            stability = adjusted_rand_score(base_labels, labels)
            stability_scores.append(stability)
        
        return {
            'mean_stability': np.mean(stability_scores),
            'std_stability': np.std(stability_scores),
            'min_stability': np.min(stability_scores),
            'max_stability': np.max(stability_scores)
        }
    """
    print("Advanced Evaluation Metrics:")
    print(evaluation_code)
```

---

## Phần 8: Tools và Technologies

### 8.1 Production-Ready Implementation

```python
def production_clustering_pipeline():
    """
    Pipeline clustering cho production environment
    """
    pipeline_code = """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.model_selection import GridSearchCV
    import joblib
    import logging
    
    class ProductionClusteringPipeline:
        def __init__(self, config):
            self.config = config
            self.pipeline = None
            self.logger = self._setup_logging()
        
        def _setup_logging(self):
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger(__name__)
        
        def create_pipeline(self):
            '''Tạo sklearn pipeline'''
            self.pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clusterer', KMeans(random_state=42))
            ])
            return self.pipeline
        
        def hyperparameter_tuning(self, X, param_grid):
            '''Hyperparameter tuning với cross-validation'''
            grid_search = GridSearchCV(
                self.pipeline, 
                param_grid,
                cv=5,
                scoring='silhouette',
                n_jobs=-1
            )
            grid_search.fit(X)
            
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            self.logger.info(f"Best score: {grid_search.best_score_}")
            
            return grid_search.best_estimator_
        
        def fit_predict(self, X):
            '''Fit model và predict clusters'''
            try:
                labels = self.pipeline.fit_predict(X)
                self.logger.info(f"Clustering completed. Found {len(np.unique(labels))} clusters")
                return labels
            except Exception as e:
                self.logger.error(f"Clustering failed: {str(e)}")
                raise
        
        def save_model(self, filepath):
            '''Lưu model'''
            joblib.dump(self.pipeline, filepath)
            self.logger.info(f"Model saved to {filepath}")
        
        def load_model(self, filepath):
            '''Load model'''
            self.pipeline = joblib.load(filepath)
            self.logger.info(f"Model loaded from {filepath}")
            
        def predict_new_data(self, X_new):
            '''Predict clusters cho dữ liệu mới'''
            return self.pipeline.predict(X_new)
    
    # Usage example
    config = {'n_clusters': 5, 'random_state': 42}
    clustering_pipeline = ProductionClusteringPipeline(config)
    pipeline = clustering_pipeline.create_pipeline()
    
    param_grid = {
        'clusterer__n_clusters': [3, 4, 5, 6, 7],
        'clusterer__init': ['k-means++', 'random'],
        'clusterer__max_iter': [100, 300, 500]
    }
    """
    print("Production-Ready Clustering Pipeline:")
    print(pipeline_code)

def monitoring_and_alerting():
    """
    Monitoring system cho clustering models
    """
    monitoring_code = """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import smtplib
    from email.mime.text import MIMEText
    
    class ClusteringMonitor:
        def __init__(self, model, baseline_metrics):
            self.model = model
            self.baseline_metrics = baseline_metrics
            self.alert_thresholds = {
                'silhouette_degradation': 0.1,
                'cluster_size_imbalance': 0.5,
                'data_drift': 0.2
            }
        
        def check_model_performance(self, X_new):
            '''Kiểm tra performance của model trên dữ liệu mới'''
            labels = self.model.predict(X_new)
            current_silhouette = silhouette_score(X_new, labels)
            
            performance_degradation = (
                self.baseline_metrics['silhouette_score'] - current_silhouette
            )
            
            if performance_degradation > self.alert_thresholds['silhouette_degradation']:
                self._send_alert(
                    f"Model performance degraded by {performance_degradation:.3f}"
                )
            
            return {
                'current_silhouette': current_silhouette,
                'performance_degradation': performance_degradation,
                'status': 'degraded' if performance_degradation > 0.1 else 'stable'
            }
        
        def detect_data_drift(self, X_baseline, X_new):
            '''Phát hiện data drift'''
            # Statistical tests for each feature
            from scipy.stats import ks_2samp
            
            drift_scores = []
            for i in range(X_baseline.shape[1]):
                statistic, p_value = ks_2samp(X_baseline[:, i], X_new[:, i])
                drift_scores.append(p_value)
            
            avg_drift = np.mean(drift_scores)
            
            if avg_drift < self.alert_thresholds['data_drift']:
                self._send_alert(f"Data drift detected. Average p-value: {avg_drift:.3f}")
            
            return {
                'drift_scores': drift_scores,
                'average_drift': avg_drift,
                'drift_detected': avg_drift < 0.2
            }
        
        def _send_alert(self, message):
            '''Gửi alert notification'''
            print(f"ALERT: {message}")
            # Implement email/Slack notification logic here
    """
    print("Monitoring and Alerting System:")
    print(monitoring_code)
```

### 8.2 Big Data Implementation

```python
def spark_clustering_example():
    """
    Clustering với Apache Spark cho big data
    """
    spark_code = """
    from pyspark.sql import SparkSession
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import ClusteringEvaluator
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("CustomerSegmentation") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    def spark_clustering_pipeline(df, feature_cols, k_range=[2, 3, 4, 5, 6]):
        '''Spark clustering pipeline'''
        
        # Feature preparation
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features_raw"
        )
        
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        
        # Find optimal K
        evaluator = ClusteringEvaluator()
        best_k = 3
        best_score = -1
        
        for k in k_range:
            kmeans = KMeans(featuresCol="features", k=k, seed=42)
            
            pipeline = Pipeline(stages=[assembler, scaler, kmeans])
            model = pipeline.fit(df)
            
            predictions = model.transform(df)
            score = evaluator.evaluate(predictions)
            
            print(f"K={k}, Silhouette Score={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        # Final model with best K
        final_kmeans = KMeans(featuresCol="features", k=best_k, seed=42)
        final_pipeline = Pipeline(stages=[assembler, scaler, final_kmeans])
        final_model = final_pipeline.fit(df)
        
        return final_model, best_k, best_score
    
    # Usage
    # df_spark = spark.read.csv("customer_data.csv", header=True, inferSchema=True)
    # feature_columns = ["Annual_Income", "Spending_Score", "Age"]
    # model, best_k, score = spark_clustering_pipeline(df_spark, feature_columns)
    """
    print("Apache Spark Implementation:")
    print(spark_code)

def dask_clustering_example():
    """
    Clustering với Dask cho out-of-core processing
    """
    dask_code = """
    import dask.dataframe as dd
    from dask_ml.cluster import KMeans
    from dask_ml.preprocessing import StandardScaler
    from dask_ml.model_selection import train_test_split
    
    def dask_clustering_pipeline(filepath):
        '''Dask clustering cho large datasets'''
        
        # Load large dataset
        df = dd.read_csv(filepath)
        
        # Feature selection
        feature_cols = ['Annual_Income', 'Spending_Score', 'Age', 'Purchase_Amount']
        X = df[feature_cols]
        
        # Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        
        # Add labels to dataframe
        df['Cluster'] = labels
        
        # Compute results (lazy evaluation until now)
        results = df.compute()
        
        return results, kmeans, scaler
    
    # Memory-efficient processing
    def incremental_clustering(filepath, chunk_size=10000):
        '''Incremental clustering cho very large datasets'''
        
        from sklearn.cluster import MiniBatchKMeans
        import pandas as pd
        
        # Initialize incremental model
        kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=1000)
        scaler = StandardScaler()
        
        # Process in chunks
        chunk_iter = pd.read_csv(filepath, chunksize=chunk_size)
        
        for i, chunk in enumerate(chunk_iter):
            print(f"Processing chunk {i+1}")
            
            # Preprocess chunk
            X_chunk = chunk[feature_cols].fillna(0)
            X_scaled = scaler.partial_fit(X_chunk).transform(X_chunk)
            
            # Incremental fit
            kmeans.partial_fit(X_scaled)
        
        return kmeans, scaler
    """
    print("Dask Implementation:")
    print(dask_code)
```

---

## Phần 9: Best Practices và Common Pitfalls

### 9.1 Best Practices

```python
def clustering_best_practices():
    """
    Best practices cho customer segmentation
    """
    practices = """
    CLUSTERING BEST PRACTICES
    ========================
    
    1. DATA PREPARATION:
       ✓ Handle missing values appropriately
       ✓ Remove or treat outliers carefully
       ✓ Scale features (StandardScaler vs RobustScaler)
       ✓ Feature selection based on business relevance
       ✓ Consider feature engineering (ratios, interactions)
    
    2. MODEL SELECTION:
       ✓ Try multiple algorithms (K-means, Hierarchical, DBSCAN)
       ✓ Use multiple validation methods (Elbow, Silhouette, Gap Statistic)
       ✓ Consider domain knowledge in choosing K
       ✓ Test stability across different random seeds
       ✓ Validate results with business stakeholders
    
    3. FEATURE ENGINEERING:
       ✓ Create meaningful derived features
       ✓ Consider temporal patterns
       ✓ Include interaction terms when relevant
       ✓ Use domain expertise for feature creation
       ✓ Balance between complexity and interpretability
    
    4. VALIDATION:
       ✓ Use holdout data for validation
       ✓ Test on different time periods
       ✓ A/B test segment-based strategies
       ✓ Monitor performance over time
       ✓ Get feedback from business users
    
    5. INTERPRETATION:
       ✓ Create clear, actionable segment profiles
       ✓ Use visualization effectively
       ✓ Translate technical results to business language
       ✓ Provide specific recommendations
       ✓ Document assumptions and limitations
    
    6. DEPLOYMENT:
       ✓ Build automated pipelines
       ✓ Implement monitoring and alerting
       ✓ Plan for model updates and retraining
       ✓ Ensure scalability for production use
       ✓ Maintain model documentation
    """
    print(practices)

def common_pitfalls():
    """
    Common pitfalls và cách tránh
    """
    pitfalls = """
    COMMON PITFALLS TO AVOID
    =======================
    
    1. DATA ISSUES:
       ❌ Using raw data without preprocessing
       ❌ Not handling missing values properly
       ❌ Ignoring outliers completely
       ❌ Using irrelevant or noisy features
       ❌ Not checking for data leakage
    
    2. ALGORITHM ISSUES:
       ❌ Choosing K arbitrarily without validation
       ❌ Using only one clustering algorithm
       ❌ Not checking for local optima
       ❌ Ignoring algorithm assumptions
       ❌ Over-fitting to current data
    
    3. BUSINESS ALIGNMENT ISSUES:
       ❌ Creating segments that aren't actionable
       ❌ Too many or too few segments
       ❌ Not involving business stakeholders
       ❌ Ignoring operational constraints
       ❌ Creating segments that change too frequently
    
    4. VALIDATION ISSUES:
       ❌ Only using internal validation metrics
       ❌ Not testing stability over time
       ❌ Cherry-picking results
       ❌ Not validating with business outcomes
       ❌ Ignoring segment interpretability
    
    5. DEPLOYMENT ISSUES:
       ❌ No plan for model maintenance
       ❌ Not monitoring model performance
       ❌ Hard-coding parameters
       ❌ No version control for models
       ❌ Not planning for data drift
    """
    print(pitfalls)
```

### 9.2 Troubleshooting Guide

```python
def troubleshooting_guide():
    """
    Troubleshooting guide cho clustering issues
    """
    guide = """
    TROUBLESHOOTING CLUSTERING ISSUES
    ================================
    
    PROBLEM: Poor Silhouette Scores
    CAUSES & SOLUTIONS:
    • Wrong number of clusters → Try different K values
    • Poor feature scaling → Apply StandardScaler or RobustScaler  
    • Irrelevant features → Perform feature selection
    • Natural clusters don't exist → Consider different approach
    • Wrong algorithm for data shape → Try different algorithms
    
    PROBLEM: Unstable Clusters
    CAUSES & SOLUTIONS:
    • K-means sensitivity → Use K-means++ initialization
    • Random seed variation → Set random_state parameter
    • Insufficient iterations → Increase max_iter
    • Poor convergence → Check convergence criteria
    • Outliers affecting centroids → Use robust clustering methods
    
    PROBLEM: Uninterpretable Results
    CAUSES & SOLUTIONS:
    • Too many features → Apply dimensionality reduction
    • Complex interactions → Simplify feature set
    • No clear business logic → Involve domain experts
    • Poor visualization → Improve plotting methods
    • Mixed data types → Handle categorical variables properly
    
    PROBLEM: Segments Not Actionable
    CAUSES & SOLUTIONS:
    • Too many segments → Reduce number of clusters
    • Overlapping segments → Improve feature engineering
    • No clear differentiation → Add discriminating features
    • Business constraints ignored → Involve stakeholders
    • Technical focus only → Add business perspective
    """
    print(guide)

def debugging_tools():
    """
    Tools và techniques để debug clustering
    """
    debug_code = """
    def debug_clustering_results(X, labels, feature_names):
        '''Comprehensive debugging của clustering results'''
        
        print("=== CLUSTERING DEBUG REPORT ===")
        
        # 1. Basic statistics
        n_clusters = len(np.unique(labels))
        n_samples = len(labels)
        
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of samples: {n_samples}")
        print(f"Cluster sizes: {np.bincount(labels)}")
        
        # 2. Silhouette analysis per cluster
        from sklearn.metrics import silhouette_samples
        silhouette_scores = silhouette_samples(X, labels)
        
        print("\\nSilhouette scores per cluster:")
        for i in range(n_clusters):
            cluster_scores = silhouette_scores[labels == i]
            print(f"Cluster {i}: mean={cluster_scores.mean():.3f}, "
                  f"std={cluster_scores.std():.3f}, "
                  f"min={cluster_scores.min():.3f}")
        
        # 3. Feature importance analysis
        print("\\nFeature importance by cluster:")
        df_debug = pd.DataFrame(X, columns=feature_names)
        df_debug['Cluster'] = labels
        
        cluster_means = df_debug.groupby('Cluster').mean()
        overall_means = df_debug[feature_names].mean()
        
        for cluster in range(n_clusters):
            print(f"\\nCluster {cluster} distinctive features:")
            cluster_mean = cluster_means.loc[cluster]
            
            # Find most distinctive features
            deviations = (cluster_mean - overall_means) / overall_means
            top_features = deviations.abs().nlargest(3)
            
            for feature in top_features.index:
                direction = "higher" if deviations[feature] > 0 else "lower"
                print(f"  {feature}: {deviations[feature]:.2%} {direction} than average")
        
        # 4. Outlier analysis
        print("\\nOutlier analysis:")
        for cluster in range(n_clusters):
            cluster_data = X[labels == cluster]
            cluster_center = cluster_data.mean(axis=0)
            
            # Find points far from cluster center
            distances = np.linalg.norm(cluster_data - cluster_center, axis=1)
            outlier_threshold = np.percentile(distances, 95)
            n_outliers = np.sum(distances > outlier_threshold)
            
            print(f"Cluster {cluster}: {n_outliers} potential outliers "
                  f"({n_outliers/len(cluster_data)*100:.1f}%)")
        
        return {
            'silhouette_scores': silhouette_scores,
            'cluster_means': cluster_means,
            'feature_deviations': deviations
        }
    
    def visualize_debug_info(X, labels, debug_info):
        '''Visualization để debug clustering'''
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Silhouette plot
        from sklearn.metrics import silhouette_score
        silhouette_scores = debug_info['silhouette_scores']
        
        y_lower = 10
        for i in range(len(np.unique(labels))):
            cluster_silhouette_scores = silhouette_scores[labels == i]
            cluster_silhouette_scores.sort()
            
            size_cluster_i = cluster_silhouette_scores.shape[0]
            y_upper = y_lower + size_cluster_i
            
            axes[0,0].fill_betweenx(range(y_lower, y_upper),
                                   0, cluster_silhouette_scores,
                                   alpha=0.7, label=f'Cluster {i}')
            y_lower = y_upper + 10
        
        axes[0,0].axvline(x=silhouette_score(X, labels), color="red", linestyle="--")
        axes[0,0].set_title('Silhouette Plot')
        axes[0,0].set_xlabel('Silhouette Score')
        axes[0,0].set_ylabel('Cluster')
        
        # 2. Cluster size distribution
        cluster_sizes = np.bincount(labels)
        axes[0,1].bar(range(len(cluster_sizes)), cluster_sizes)
        axes[0,1].set_title('Cluster Size Distribution')
        axes[0,1].set_xlabel('Cluster')
        axes[0,1].set_ylabel('Number of Points')
        
        # 3. Feature heatmap (if 2D projection available)
        if X.shape[1] >= 2:
            scatter = axes[1,0].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
            axes[1,0].set_title('Cluster Visualization (First 2 Features)')
            plt.colorbar(scatter, ax=axes[1,0])
        
        # 4. Intra-cluster distances
        intra_distances = []
        for cluster in range(len(np.unique(labels))):
            cluster_data = X[labels == cluster]
            if len(cluster_data) > 1:
                center = cluster_data.mean(axis=0)
                distances = np.linalg.norm(cluster_data - center, axis=1)
                intra_distances.extend(distances)
        
        axes[1,1].hist(intra_distances, bins=30, alpha=0.7)
        axes[1,1].set_title('Intra-cluster Distance Distribution')
        axes[1,1].set_xlabel('Distance to Cluster Center')
        axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    """
    print("Debugging Tools:")
    print(debug_code)
```

---

## Tài liệu tham khảo và học tập thêm

### Recommended Reading:
1. **Sách**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 9
2. **Paper**: "K-means++: The Advantages of Careful Seeding" by Arthur & Vassilvitskii
3. **Online Course**: Andrew Ng's Machine Learning Course (Coursera)
4. **Documentation**: Scikit-learn Clustering Documentation

### Datasets để thực hành:
1. **Mall Customer Segmentation** (Kaggle)
2. **Online Retail Dataset** (UCI ML Repository)
3. **Credit Card Customer Data** (Kaggle)
4. **Telco Customer Churn** (Kaggle)

### Tools và Libraries:
- **Python**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **R**: cluster, factoextra, NbClust
- **Big Data**: Apache Spark MLlib, Dask-ML
- **Visualization**: Plotly, Bokeh, Tableau

---

## Tóm tắt bài học

Qua bài học này, học viên đã được trang bị:

1. **Kiến thức nền tảng** về Customer Segmentation và Unsupervised Learning
2. **Kỹ năng kỹ thuật** implement K-means và các thuật toán clustering khác  
3. **Khả năng phân tích** dữ liệu và diễn giải kết quả clustering
4. **Tư duy kinh doanh** để chuyển đổi insights thành strategies
5. **Kinh nghiệm thực tế** qua các
