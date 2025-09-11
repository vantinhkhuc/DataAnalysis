# Lab 01 : PYTHON CƠ BẢN CHO PHÂN TÍCH DỮ LIỆU

**Thời gian:** 150 phút (2.5 giờ)  
**Đối tượng:** Học viên có kiến thức lập trình C# cơ bản  
**Phân bổ thời gian:** Lý thuyết 50 phút + Thực hành 100 phút

---

## MỤC TIÊU BÀI HỌC

### Kiến thức:
- Hiểu được syntax cơ bản của Python và sự khác biệt với C#
- Nắm vững các kiểu dữ liệu và cấu trúc dữ liệu trong Python
- Hiểu về thư viện NumPy và Pandas cho phân tích dữ liệu
- Biết cách sử dụng Matplotlib cho visualization cơ bản

### Kỹ năng:
- Viết được code Python cơ bản cho xử lý dữ liệu
- Sử dụng được Pandas để đọc, xử lý và phân tích dữ liệu
- Tạo được biểu đồ đơn giản với Matplotlib
- Thực hiện được các phép toán thống kê cơ bản

---

## PHẦN I: LỶ THUYẾT (50 PHÚT)

### 1. So sánh Python và C# (10 phút)

#### Điểm khác biệt chính:

**C# (Quen thuộc)**
```csharp
// Khai báo biến có kiểu
int number = 10;
string name = "John";

// Method với access modifier
public static void Main(string[] args)
{
    Console.WriteLine("Hello World");
}

// Strongly typed
List<int> numbers = new List<int>();
```

**Python (Mới)**
```python
# Khai báo biến không cần kiểu
number = 10
name = "John"

# Function đơn giản hơn
def main():
    print("Hello World")

# Dynamic typing
numbers = []  # có thể chứa bất kỳ kiểu nào
```

#### Ưu điểm của Python cho Data Analysis:
- Syntax đơn giản, dễ đọc
- Thư viện phong phú cho data science
- Interactive development (Jupyter Notebook)
- Cộng đồng lớn và tài liệu phong phú

### 2. Syntax cơ bản Python (15 phút)

#### 2.1 Biến và Kiểu dữ liệu
```python
# Kiểu số
age = 25          # int
price = 19.99     # float
is_student = True # bool

# Kiểu chuỗi
name = "Nguyen Van A"
description = '''Đây là chuỗi
nhiều dòng'''

# Kiểu None (tương đương null trong C#)
data = None
```

#### 2.2 Collections
```python
# List (tương đương List<object> trong C#)
fruits = ["apple", "banana", "orange"]
mixed_list = [1, "hello", True, 3.14]

# Dictionary (tương đương Dictionary<string, object>)
person = {
    "name": "John",
    "age": 30,
    "city": "Hanoi"
}

# Tuple (immutable)
coordinates = (10, 20)
```

#### 2.3 Control Flow
```python
# If statement (không cần dấu ngoặc và dấu chấm phẩy)
if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")

# For loop
for fruit in fruits:
    print(fruit)

# While loop
count = 0
while count < 5:
    print(count)
    count += 1
```

### 3. Thư viện cho Data Analysis (15 phút)

#### 3.1 NumPy - Xử lý mảng số học
```python
import numpy as np

# Tạo mảng
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])

# Các phép toán
print(arr.mean())  # Trung bình
print(arr.sum())   # Tổng
print(arr.std())   # Độ lệch chuẩn
```

#### 3.2 Pandas - Xử lý dữ liệu có cấu trúc
```python
import pandas as pd

# DataFrame (giống DataTable trong C#)
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['Hanoi', 'HCMC', 'Danang']
})

# Đọc dữ liệu từ file
df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')
```

#### 3.3 Matplotlib - Visualization
```python
import matplotlib.pyplot as plt

# Biểu đồ đường
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Sample Plot')
plt.show()

# Biểu đồ cột
plt.bar(['A', 'B', 'C'], [1, 3, 2])
plt.show()
```

### 4. Jupyter Notebook (10 phút)

#### Ưu điểm:
- Chạy code từng cell
- Kết hợp code, text và visualization
- Dễ dàng thử nghiệm và debug

#### Các phím tắt quan trọng:
- `Shift + Enter`: Chạy cell hiện tại
- `Ctrl + Enter`: Chạy cell không di chuyển
- `A`: Thêm cell ở trên
- `B`: Thêm cell ở dưới
- `M`: Chuyển sang Markdown
- `Y`: Chuyển sang Code

---

## PHẦN II: THỰC HÀNH (100 PHÚT)

### Bài tập 1: Làm quen với Python syntax (20 phút)

#### Yêu cầu:
Viết chương trình Python thực hiện các nhiệm vụ sau:

1. **Tạo danh sách sinh viên (5 phút)**
```python
# TODO: Tạo danh sách sinh viên với thông tin: tên, tuổi, điểm
students = [
    {"name": "Nguyen Van A", "age": 20, "score": 8.5},
    {"name": "Tran Thi B", "age": 19, "score": 9.0},
    {"name": "Le Van C", "age": 21, "score": 7.5}
]
```

2. **Xử lý dữ liệu cơ bản (10 phút)**
```python
# TODO: Viết function tính điểm trung bình
def calculate_average_score(students):
    # Gợi ý: sử dụng sum() và len()
    pass

# TODO: Tìm sinh viên có điểm cao nhất
def find_top_student(students):
    # Gợi ý: sử dụng max() với key parameter
    pass

# TODO: Lọc sinh viên trên 18 tuổi
def filter_adult_students(students):
    # Gợi ý: sử dụng list comprehension
    pass
```

3. **In kết quả (5 phút)**
```python
# TODO: In thông tin theo format đẹp
print(f"Điểm trung bình lớp: {calculate_average_score(students):.2f}")
print(f"Sinh viên xuất sắc nhất: {find_top_student(students)['name']}")
print("Danh sách sinh viên trưởng thành:")
for student in filter_adult_students(students):
    print(f"- {student['name']} ({student['age']} tuổi)")
```

#### Đáp án:
```python
students = [
    {"name": "Nguyen Van A", "age": 20, "score": 8.5},
    {"name": "Tran Thi B", "age": 19, "score": 9.0},
    {"name": "Le Van C", "age": 21, "score": 7.5}
]

def calculate_average_score(students):
    total_score = sum(student["score"] for student in students)
    return total_score / len(students)

def find_top_student(students):
    return max(students, key=lambda x: x["score"])

def filter_adult_students(students):
    return [student for student in students if student["age"] >= 18]

# In kết quả
print(f"Điểm trung bình lớp: {calculate_average_score(students):.2f}")
print(f"Sinh viên xuất sắc nhất: {find_top_student(students)['name']}")
print("Danh sách sinh viên trưởng thành:")
for student in filter_adult_students(students):
    print(f"- {student['name']} ({student['age']} tuổi)")
```

### Bài tập 2: Làm việc với NumPy (20 phút)

#### Yêu cầu:
Phân tích dữ liệu bán hàng của một cửa hàng

```python
import numpy as np

# Dữ liệu bán hàng 30 ngày (triệu đồng)
sales_data = np.array([
    12.5, 15.3, 18.2, 14.7, 16.8, 13.9, 17.4, 19.1, 15.6, 16.2,
    14.3, 18.7, 16.9, 15.8, 17.2, 13.4, 16.5, 18.9, 15.1, 17.6,
    14.8, 16.3, 18.1, 15.9, 17.8, 16.4, 18.5, 15.2, 17.1, 16.7
])

# TODO 1: Tính các thống kê cơ bản
print("=== THỐNG KÊ BÁN HÀNG ===")
# Doanh thu trung bình
# Doanh thu cao nhất và thấp nhất
# Độ lệch chuẩn
# Tổng doanh thu

# TODO 2: Phân tích theo tuần (mỗi tuần 7 ngày)
# Reshape dữ liệu thành ma trận 4x7 (4 tuần, 7 ngày)
# Tính doanh thu trung bình mỗi tuần
# Tìm tuần có doanh thu cao nhất

# TODO 3: Phân tích ngày trong tuần
# Tính doanh thu trung bình theo ngày trong tuần
# Tìm ngày bán chạy nhất trong tuần
```

#### Đáp án:
```python
import numpy as np

sales_data = np.array([
    12.5, 15.3, 18.2, 14.7, 16.8, 13.9, 17.4, 19.1, 15.6, 16.2,
    14.3, 18.7, 16.9, 15.8, 17.2, 13.4, 16.5, 18.9, 15.1, 17.6,
    14.8, 16.3, 18.1, 15.9, 17.8, 16.4, 18.5, 15.2, 17.1, 16.7
])

# Thống kê cơ bản
print("=== THỐNG KÊ BÁN HÀNG ===")
print(f"Doanh thu trung bình: {sales_data.mean():.2f} triệu")
print(f"Doanh thu cao nhất: {sales_data.max():.2f} triệu")
print(f"Doanh thu thấp nhất: {sales_data.min():.2f} triệu")
print(f"Độ lệch chuẩn: {sales_data.std():.2f} triệu")
print(f"Tổng doanh thu: {sales_data.sum():.2f} triệu")

# Phân tích theo tuần
weekly_data = sales_data[:28].reshape(4, 7)  # Lấy 28 ngày đầu
weekly_avg = weekly_data.mean(axis=1)
print(f"\nDoanh thu trung bình theo tuần:")
for i, avg in enumerate(weekly_avg):
    print(f"Tuần {i+1}: {avg:.2f} triệu")
print(f"Tuần có doanh thu cao nhất: Tuần {weekly_avg.argmax() + 1}")

# Phân tích ngày trong tuần
daily_avg = weekly_data.mean(axis=0)
days = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'CN']
print(f"\nDoanh thu trung bình theo ngày trong tuần:")
for day, avg in zip(days, daily_avg):
    print(f"{day}: {avg:.2f} triệu")
print(f"Ngày bán chạy nhất: {days[daily_avg.argmax()]}")
```

### Bài tập 3: Pandas DataFrame (30 phút)

#### Yêu cầu:
Phân tích dữ liệu khách hàng của một cửa hàng online

```python
import pandas as pd
import numpy as np

# Tạo dữ liệu mẫu
np.random.seed(42)
customers_data = {
    'CustomerID': range(1001, 1101),
    'Name': [f'Customer_{i}' for i in range(1, 101)],
    'Age': np.random.randint(18, 65, 100),
    'City': np.random.choice(['Hanoi', 'HCMC', 'Danang', 'Haiphong', 'Cantho'], 100),
    'TotalSpent': np.random.uniform(100, 5000, 100),
    'OrderCount': np.random.randint(1, 20, 100),
    'LastOrderDate': pd.date_range('2023-01-01', periods=100, freq='D')
}

df = pd.DataFrame(customers_data)

# TODO 1: Khám phá dữ liệu cơ bản (5 phút)
print("=== KHÁM PHÁ DỮ LIỆU ===")
# In 5 dòng đầu
# Thông tin về DataFrame (shape, columns, dtypes)
# Thống kê mô tả

# TODO 2: Phân tích theo thành phố (10 phút)
print("\n=== PHÂN TÍCH THEO THÀNH PHỐ ===")
# Số lượng khách hàng mỗi thành phố
# Tổng chi tiêu theo thành phố
# Khách hàng chi tiêu cao nhất mỗi thành phố

# TODO 3: Phân tích theo độ tuổi (10 phút)
print("\n=== PHÂN TÍCH THEO ĐỘ TUỔI ===")
# Tạo nhóm tuổi: 18-25, 26-35, 36-45, 46+
# Chi tiêu trung bình theo nhóm tuổi
# Số đơn hàng trung bình theo nhóm tuổi

# TODO 4: Tìm top khách hàng (5 phút)
print("\n=== TOP KHÁCH HÀNG ===")
# Top 10 khách hàng chi tiêu nhiều nhất
# Top 10 khách hàng có nhiều đơn hàng nhất
# Khách hàng có giá trị đơn hàng trung bình cao nhất
```

#### Đáp án:
```python
import pandas as pd
import numpy as np

# Tạo dữ liệu
np.random.seed(42)
customers_data = {
    'CustomerID': range(1001, 1101),
    'Name': [f'Customer_{i}' for i in range(1, 101)],
    'Age': np.random.randint(18, 65, 100),
    'City': np.random.choice(['Hanoi', 'HCMC', 'Danang', 'Haiphong', 'Cantho'], 100),
    'TotalSpent': np.random.uniform(100, 5000, 100).round(2),
    'OrderCount': np.random.randint(1, 20, 100),
    'LastOrderDate': pd.date_range('2023-01-01', periods=100, freq='D')
}

df = pd.DataFrame(customers_data)

# 1. Khám phá dữ liệu cơ bản
print("=== KHÁM PHÁ DỮ LIỆU ===")
print("5 dòng đầu:")
print(df.head())
print(f"\nKích thước: {df.shape}")
print(f"Cột: {list(df.columns)}")
print("\nThống kê mô tả:")
print(df.describe())

# 2. Phân tích theo thành phố
print("\n=== PHÂN TÍCH THEO THÀNH PHỐ ===")
city_analysis = df.groupby('City').agg({
    'CustomerID': 'count',
    'TotalSpent': ['sum', 'mean'],
    'OrderCount': 'mean'
}).round(2)

print("Số lượng khách hàng theo thành phố:")
print(df['City'].value_counts())
print("\nTổng chi tiêu theo thành phố:")
print(df.groupby('City')['TotalSpent'].sum().sort_values(ascending=False))

# Top spender mỗi thành phố
print("\nKhách hàng chi tiêu cao nhất mỗi thành phố:")
top_by_city = df.loc[df.groupby('City')['TotalSpent'].idxmax()]
print(top_by_city[['City', 'Name', 'TotalSpent']])

# 3. Phân tích theo độ tuổi
print("\n=== PHÂN TÍCH THEO ĐỘ TUỔI ===")
df['AgeGroup'] = pd.cut(df['Age'], 
                       bins=[17, 25, 35, 45, 100], 
                       labels=['18-25', '26-35', '36-45', '46+'])

age_analysis = df.groupby('AgeGroup').agg({
    'TotalSpent': 'mean',
    'OrderCount': 'mean'
}).round(2)
print("Chi tiêu và đơn hàng trung bình theo nhóm tuổi:")
print(age_analysis)

# 4. Top khách hàng
print("\n=== TOP KHÁCH HÀNG ===")
print("Top 10 khách hàng chi tiêu nhiều nhất:")
top_spenders = df.nlargest(10, 'TotalSpent')[['Name', 'City', 'TotalSpent']]
print(top_spenders)

print("\nTop 10 khách hàng có nhiều đơn hàng nhất:")
top_buyers = df.nlargest(10, 'OrderCount')[['Name', 'City', 'OrderCount']]
print(top_buyers)

# Giá trị đơn hàng trung bình
df['AvgOrderValue'] = (df['TotalSpent'] / df['OrderCount']).round(2)
print("\nTop 10 khách hàng có giá trị đơn hàng trung bình cao nhất:")
top_aov = df.nlargest(10, 'AvgOrderValue')[['Name', 'City', 'AvgOrderValue']]
print(top_aov)
```

### Bài tập 4: Data Visualization (30 phút)

#### Yêu cầu:
Tạo các biểu đồ trực quan hóa dữ liệu khách hàng

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sử dụng dữ liệu từ bài tập 3
# (Code tạo dữ liệu như trên)

# TODO 1: Biểu đồ cột - Số khách hàng theo thành phố (8 phút)
plt.figure(figsize=(10, 6))
# Tạo biểu đồ cột
# Thêm title, labels
# Xoay labels nếu cần
# Thêm giá trị trên mỗi cột

# TODO 2: Biểu đồ histogram - Phân bố độ tuổi (8 phút)
plt.figure(figsize=(10, 6))
# Tạo histogram với 15 bins
# Thêm đường trung bình
# Thêm title và labels

# TODO 3: Biểu đồ scatter - Mối quan hệ Age vs TotalSpent (8 phút)
plt.figure(figsize=(10, 6))
# Tạo scatter plot
# Color theo City
# Thêm trendline
# Legend và labels

# TODO 4: Subplots - Kết hợp nhiều biểu đồ (6 phút)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
# Biểu đồ 1: Bar chart - TotalSpent theo City
# Biểu đồ 2: Pie chart - Phân bố City  
# Biểu đồ 3: Box plot - TotalSpent theo AgeGroup
# Biểu đồ 4: Line plot - Xu hướng đặt hàng theo thời gian
```

#### Đáp án:
```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sử dụng dữ liệu từ bài tập 3
np.random.seed(42)
customers_data = {
    'CustomerID': range(1001, 1101),
    'Name': [f'Customer_{i}' for i in range(1, 101)],
    'Age': np.random.randint(18, 65, 100),
    'City': np.random.choice(['Hanoi', 'HCMC', 'Danang', 'Haiphong', 'Cantho'], 100),
    'TotalSpent': np.random.uniform(100, 5000, 100).round(2),
    'OrderCount': np.random.randint(1, 20, 100),
    'LastOrderDate': pd.date_range('2023-01-01', periods=100, freq='D')
}
df = pd.DataFrame(customers_data)

# 1. Biểu đồ cột - Số khách hàng theo thành phố
plt.figure(figsize=(10, 6))
city_counts = df['City'].value_counts()
bars = plt.bar(city_counts.index, city_counts.values, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
plt.title('Số lượng khách hàng theo thành phố', fontsize=16, fontweight='bold')
plt.xlabel('Thành phố')
plt.ylabel('Số lượng khách hàng')
plt.xticks(rotation=45)

# Thêm giá trị trên mỗi cột
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 2. Biểu đồ histogram - Phân bố độ tuổi
plt.figure(figsize=(10, 6))
plt.hist(df['Age'], bins=15, color='skyblue', alpha=0.7, edgecolor='black')
plt.axvline(df['Age'].mean(), color='red', linestyle='--', 
            label=f'Trung bình: {df["Age"].mean():.1f} tuổi')
plt.title('Phân bố độ tuổi khách hàng', fontsize=16, fontweight='bold')
plt.xlabel('Độ tuổi')
plt.ylabel('Số lượng khách hàng')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Biểu đồ scatter - Age vs TotalSpent
plt.figure(figsize=(12, 8))
colors = {'Hanoi': 'red', 'HCMC': 'blue', 'Danang': 'green', 
          'Haiphong': 'orange', 'Cantho': 'purple'}

for city in df['City'].unique():
    city_data = df[df['City'] == city]
    plt.scatter(city_data['Age'], city_data['TotalSpent'], 
               c=colors[city], label=city, alpha=0.6, s=60)

# Trendline
z = np.polyfit(df['Age'], df['TotalSpent'], 1)
p = np.poly1d(z)
plt.plot(df['Age'], p(df['Age']), "r--", alpha=0.8, linewidth=2)

plt.title('Mối quan hệ giữa độ tuổi và tổng chi tiêu', fontsize=16, fontweight='bold')
plt.xlabel('Độ tuổi')
plt.ylabel('Tổng chi tiêu (VNĐ)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Subplots - Kết hợp nhiều biểu đồ
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Biểu đồ 1: Bar chart - TotalSpent theo City
city_spending = df.groupby('City')['TotalSpent'].sum()
axes[0, 0].bar(city_spending.index, city_spending.values, color='lightcoral')
axes[0, 0].set_title('Tổng chi tiêu theo thành phố')
axes[0, 0].tick_params(axis='x', rotation=45)

# Biểu đồ 2: Pie chart - Phân bố City
city_counts = df['City'].value_counts()
axes[0, 1].pie(city_counts.values, labels=city_counts.index, autopct='%1.1f%%')
axes[0, 1].set_title('Phân bố khách hàng theo thành phố')

# Biểu đồ 3: Box plot - TotalSpent theo AgeGroup
df['AgeGroup'] = pd.cut(df['Age'], bins=[17, 25, 35, 45, 100], 
                       labels=['18-25', '26-35', '36-45', '46+'])
age_groups = [df[df['AgeGroup'] == group]['TotalSpent'].values 
              for group in df['AgeGroup'].cat.categories]
axes[1, 0].boxplot(age_groups, labels=df['AgeGroup'].cat.categories)
axes[1, 0].set_title('Phân bố chi tiêu theo nhóm tuổi')
axes[1, 0].tick_params(axis='x', rotation=45)

# Biểu đồ 4: Line plot - Xu hướng theo thời gian
monthly_orders = df.groupby(df['LastOrderDate'].dt.month)['OrderCount'].sum()
axes[1, 1].plot(monthly_orders.index, monthly_orders.values, 
                marker='o', linewidth=2, markersize=6)
axes[1, 1].set_title('Xu hướng đơn hàng theo tháng')
axes[1, 1].set_xlabel('Tháng')
axes[1, 1].set_ylabel('Tổng số đơn hàng')

plt.tight_layout()
plt.show()
```

---

## ĐÁNH GIÁ VÀ TỔNG KẾT

### Tiêu chí đánh giá:
- **Kiến thức lý thuyết (30%)**: Hiểu syntax Python, khác biệt với C#
- **Thực hành cơ bản (25%)**: Bài tập 1 - Python syntax
- **NumPy (20%)**: Bài tập 2 - Xử lý mảng và tính toán
- **Pandas (15%)**: Bài tập 3 - DataFrame operations
- **Visualization (10%)**: Bài tập 4 - Matplotlib

### Câu hỏi ôn tập:
1. So sánh cách khai báo biến trong Python và C#?
2. Pandas DataFrame khác gì với DataTable trong C#?
3. Khi nào nên dùng NumPy, khi nào nên dùng Pandas?
4. Làm thế nào để đọc file CSV vào Python?
5. Cách tạo biểu đồ cơ bản với Matplotlib?

### Bài tập về nhà:
1. Tìm hiểu thêm về thư viện Seaborn cho visualization nâng cao
2. Thực hành với dataset thực tế (có thể download từ Kaggle)
3. Tạo một báo cáo phân tích dữ liệu hoàn chỉnh với Jupyter Notebook

### Tài liệu tham khảo:
- **Python Official Documentation**: https://docs.python.org/3/
- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **NumPy Documentation**: https://numpy.org/doc/
- **Matplotlib Documentation**: https://matplotlib.org/stable/
- **Jupyter Notebook**: https://jupyter.org/documentation

---

## PHỤ LỤC: CODE TEMPLATES VÀ CHEAT SHEET

### A. Template cho Data Analysis Project

```python
# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATA
def load_data(file_path):
    """Load data from various formats"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)

# 2. EXPLORE DATA  
def explore_data(df):
    """Basic data exploration"""
    print("=== DATA INFO ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())
    print("\n=== DATA TYPES ===")
    print(df.dtypes)
    print("\n=== STATISTICAL SUMMARY ===")
    print(df.describe())

# 3. CLEAN DATA
def clean_data(df):
    """Basic data cleaning"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    # Option 1: Drop rows with missing values
    # df = df.dropna()
    
    # Option 2: Fill missing values
    # df = df.fillna(method='forward')
    
    return df

# 4. VISUALIZE DATA
def create_visualizations(df):
    """Create basic visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram for numeric column
    numeric_col = df.select_dtypes(include=[np.number]).columns[0]
    axes[0, 0].hist(df[numeric_col].dropna(), bins=20)
    axes[0, 0].set_title(f'Distribution of {numeric_col}')
    
    # Count plot for categorical column
    cat_col = df.select_dtypes(include=['object']).columns[0]
    df[cat_col].value_counts().plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title(f'Count of {cat_col}')
    
    plt.tight_layout()
    plt.show()

# Main analysis function
def main_analysis(file_path):
    """Main analysis pipeline"""
    # Load data
    df = load_data(file_path)
    
    # Explore data
    explore_data(df)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Visualize data
    create_visualizations(df_clean)
    
    return df_clean

# Usage
# df = main_analysis('your_data.csv')
```

### B. Python for C# Developers Cheat Sheet

| **Concept** | **C#** | **Python** |
|-------------|--------|------------|
| **Variables** | `int age = 25;` | `age = 25` |
| **String formatting** | `$"Hello {name}"` | `f"Hello {name}"` |
| **Arrays/Lists** | `int[] arr = {1,2,3};` | `arr = [1, 2, 3]` |
| **Dictionary** | `Dictionary<string, int>` | `dict = {'key': value}` |
| **For loop** | `for(int i=0; i<n; i++)` | `for i in range(n):` |
| **Functions** | `public int Add(int a, int b)` | `def add(a, b):` |
| **Classes** | `public class Person { }` | `class Person:` |
| **Null check** | `if (obj != null)` | `if obj is not None:` |
| **Exception handling** | `try { } catch { }` | `try: except:` |

### C. Pandas Quick Reference

```python
# READING DATA
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
df = pd.read_json('file.json')

# BASIC INFO
df.head()          # First 5 rows
df.tail()          # Last 5 rows  
df.info()          # Data types and null counts
df.describe()      # Statistical summary
df.shape           # (rows, columns)
df.columns         # Column names

# SELECTING DATA
df['column']       # Select single column
df[['col1', 'col2']]  # Select multiple columns
df.iloc[0]         # Select by position
df.loc[0]          # Select by index
df[df['age'] > 25] # Filter rows

# GROUPING AND AGGREGATION
df.groupby('column').sum()
df.groupby('column').agg({'col1': 'mean', 'col2': 'sum'})
df.pivot_table(values='value', index='row', columns='col')

# MISSING DATA
df.isnull().sum()  # Count missing values
df.dropna()        # Remove rows with missing values
df.fillna(value)   # Fill missing values

# SORTING
df.sort_values('column')
df.sort_values(['col1', 'col2'], ascending=[True, False])
```

### D. Common Matplotlib Patterns

```python
import matplotlib.pyplot as plt

# BASIC SETUP
plt.figure(figsize=(10, 6))
plt.style.use('seaborn')  # Use seaborn style

# LINE PLOT
plt.plot(x, y, marker='o', linestyle='-', color='blue', label='Data')
plt.title('Title')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.legend()
plt.grid(True)

# BAR PLOT
plt.bar(categories, values, color='skyblue')
plt.xticks(rotation=45)

# HISTOGRAM  
plt.hist(data, bins=20, alpha=0.7, color='green')

# SCATTER PLOT
plt.scatter(x, y, c=colors, s=sizes, alpha=0.6)
plt.colorbar()

# SUBPLOTS
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(x, y)
axes[0, 1].bar(x, y)
plt.tight_layout()

# SAVE FIGURE
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

### E. Error Handling Best Practices

```python
# FILE OPERATIONS
try:
    df = pd.read_csv('data.csv')
    print("Data loaded successfully")
except FileNotFoundError:
    print("File not found. Please check the file path.")
except pd.errors.EmptyDataError:
    print("The file is empty.")
except Exception as e:
    print(f"An error occurred: {e}")

# DATA OPERATIONS
def safe_division(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
    except TypeError:
        print("Invalid input types")
        return None

# VALIDATION
def validate_dataframe(df, required_columns):
    """Validate if DataFrame has required columns"""
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    return True
```

---

## HƯỚNG DẪN CHO GIẢNG VIÊN

### Chuẩn bị trước buổi học:
1. **Môi trường**: Đảm bảo tất cả máy đã cài đặt Python, Jupyter Notebook
2. **Dữ liệu**: Chuẩn bị file dữ liệu mẫu (CSV, Excel) cho thực hành
3. **Code**: Kiểm tra tất cả code examples hoạt động chính xác

### Gợi ý dẫn dắt:
1. **Mở đầu**: So sánh với C# để học viên dễ tiếp cận
2. **Demo**: Thực hiện live coding để học viên theo dõi
3. **Tương tác**: Đặt câu hỏi trong quá trình giảng
4. **Thực hành**: Đi kiểm tra và hỗ trợ từng nhóm

### Xử lý tình huống:
- **Học viên chậm**: Cung cấp code template
- **Lỗi technical**: Có backup code và dữ liệu
- **Câu hỏi khó**: Ghi chú để trả lời sau hoặc tìm hiểu thêm

### Mở rộng nâng cao:
- **Seaborn** cho visualization đẹp hơn
- **Plotly** cho interactive charts  
- **Scikit-learn** cho machine learning cơ bản
- **Streamlit** cho tạo web app đơn giản

---

**LƯU Ý**: Bài học này được thiết kế linh hoạt. Giảng viên có thể điều chỉnh thời gian và độ khó tùy theo trình độ lớp học.

### Bài tập 2B: Ma trận điểm số và phân tích học tập (25 phút)

#### Yêu cầu:
Phân tích ma trận điểm của 50 sinh viên qua 6 môn học

```python
import numpy as np

# Tạo dữ liệu giả lập
np.random.seed(123)
n_students = 50
n_subjects = 6
subject_names = ['Toán', 'Lý', 'Hóa', 'Sinh', 'Văn', 'Anh']

# Ma trận điểm 50x6 (50 sinh viên, 6 môn)
scores_matrix = np.random.normal(7.5, 1.5, (n_students, n_subjects))
# Đảm bảo điểm trong khoảng 0-10
scores_matrix = np.clip(scores_matrix, 0, 10)

print(f"Ma trận điểm shape: {scores_matrix.shape}")
print(f"5 sinh viên đầu tiên:\n{scores_matrix[:5]}")

# TODO 1: Phân tích tổng quan (8 phút)
print("\n=== PHÂN TÍCH TỔNG QUAN ===")
# Điểm trung bình mỗi môn
# Môn học có điểm trung bình cao nhất/thấp nhất
# Độ lệch chuẩn mỗi môn (môn nào có độ phân tán cao nhất)
# Ma trận correlation giữa các môn

# TODO 2: Phân tích sinh viên (8 phút)  
print("\n=== PHÂN TÍCH SINH VIÊN ===")
# Điểm trung bình mỗi sinh viên
# Top 10 sinh viên có điểm cao nhất
# Sinh viên cần học bổ sung (điểm trung bình < 5.0)
# Phân phối điểm trung bình của lớp

# TODO 3: Phân tích nâng cao (9 phút)
print("\n=== PHÂN TÍCH NÂNG CAO ===")
# Tìm sinh viên "cân bằng" (độ lệch chuẩn điểm thấp)
# Tìm sinh viên "thiên lệch" (giỏi 1-2 môn, yếu các môn khác)
# Ma trận chuẩn hóa Z-score
# Ranking sinh viên theo từng môn
```

#### Đáp án:
```python
import numpy as np

# Tạo dữ liệu
np.random.seed(123)
n_students = 50
n_subjects = 6
subject_names = ['Toán', 'Lý', 'Hóa', 'Sinh', 'Văn', 'Anh']

scores_matrix = np.random.normal(7.5, 1.5, (n_students, n_subjects))
scores_matrix = np.clip(scores_matrix, 0, 10)

print(f"Ma trận điểm shape: {scores_matrix.shape}")

# 1. Phân tích tổng quan
print("\n=== PHÂN TÍCH TỔNG QUAN ===")
subject_means = scores_matrix.mean(axis=0)
subject_stds = scores_matrix.std(axis=0)

print("Điểm trung bình theo môn:")
for i, (subject, mean) in enumerate(zip(subject_names, subject_means)):
    print(f"  {subject}: {mean:.2f} (±{subject_stds[i]:.2f})")

best_subject = subject_names[np.argmax(subject_means)]
worst_subject = subject_names[np.argmin(subject_means)]
print(f"\nMôn có điểm cao nhất: {best_subject} ({subject_means.max():.2f})")
print(f"Môn có điểm thấp nhất: {worst_subject} ({subject_means.min():.2f})")

most_varied = subject_names[np.argmax(subject_stds)]
print(f"Môn có độ phân tán cao nhất: {most_varied} (std={subject_stds.max():.2f})")

# Ma trận correlation
correlation_matrix = np.corrcoef(scores_matrix.T)
print(f"\nMa trận tương quan:\n{correlation_matrix.round(2)}")

# 2. Phân tích sinh viên
print("\n=== PHÂN TÍCH SINH VIÊN ===")
student_means = scores_matrix.mean(axis=1)
student_stds = scores_matrix.std(axis=1)

print(f"Điểm trung bình lớp: {student_means.mean():.2f}")
print(f"Độ lệch chuẩn lớp: {student_means.std():.2f}")

# Top 10 sinh viên
top_students = np.argsort(student_means)[-10:][::-1]
print("\nTop 10 sinh viên xuất sắc:")
for rank, student_idx in enumerate(top_students, 1):
    print(f"  {rank}. Sinh viên {student_idx+1}: {student_means[student_idx]:.2f}")

# Sinh viên cần học bổ sung
weak_students = np.where(student_means < 5.0)[0]
print(f"\nSố sinh viên cần học bổ sung: {len(weak_students)}")
if len(weak_students) > 0:
    print("Danh sách:")
    for student_idx in weak_students:
        print(f"  Sinh viên {student_idx+1}: {student_means[student_idx]:.2f}")

# Phân phối điểm
bins = [0, 5, 6.5, 8, 9, 10]
hist, _ = np.histogram(student_means, bins=bins)
labels = ['Yếu (<5)', 'Trung bình (5-6.5)', 'Khá (6.5-8)', 'Giỏi (8-9)', 'Xuất sắc (9-10)']
print("\nPhân phối xếp loại:")
for label, count in zip(labels, hist):
    print(f"  {label}: {count} sinh viên ({count/n_students*100:.1f}%)")

# 3. Phân tích nâng cao
print("\n=== PHÂN TÍCH NÂNG CAO ===")

# Sinh viên cân bằng (độ lệch chuẩn thấp)
balanced_threshold = np.percentile(student_stds, 25)  # 25% thấp nhất
balanced_students = np.where(student_stds <= balanced_threshold)[0]
print(f"Top 5 sinh viên có điểm cân bằng nhất:")
balanced_top5 = balanced_students[np.argsort(student_stds[balanced_students])[:5]]
for student_idx in balanced_top5:
    print(f"  Sinh viên {student_idx+1}: ĐTB={student_means[student_idx]:.2f}, STD={student_stds[student_idx]:.2f}")

# Sinh viên thiên lệch (độ lệch chuẩn cao)
unbalanced_threshold = np.percentile(student_stds, 75)  # 25% cao nhất
unbalanced_students = np.where(student_stds >= unbalanced_threshold)[0]
print(f"\nTop 5 sinh viên có điểm thiên lệch nhất:")
unbalanced_top5 = unbalanced_students[np.argsort(student_stds[unbalanced_students])[-5:]]
for student_idx in unbalanced_top5:
    scores_str = ", ".join([f"{score:.1f}" for score in scores_matrix[student_idx]])
    print(f"  Sinh viên {student_idx+1}: STD={student_stds[student_idx]:.2f} ({scores_str})")

# Z-score normalization
z_scores = (scores_matrix - scores_matrix.mean(axis=0)) / scores_matrix.std(axis=0)
print(f"\nMa trận Z-score shape: {z_scores.shape}")
print("Z-score trung bình mỗi sinh viên (5 sinh viên đầu):")
for i in range(5):
    print(f"  Sinh viên {i+1}: {z_scores[i].mean():.2f}")

# Ranking theo từng môn
print("\nTop 3 mỗi môn:")
for subject_idx, subject in enumerate(subject_names):
    top3_idx = np.argsort(scores_matrix[:, subject_idx])[-3:][::-1]
    print(f"  {subject}:")
    for rank, student_idx in enumerate(top3_idx, 1):
        print(f"    {rank}. SV{student_idx+1}: {scores_matrix[student_idx, subject_idx]:.2f}")
```

### Bài tập 2C: Phân tích dữ liệu tài chính với NumPy (25 phút)

#### Yêu cầu:
Phân tích dữ liệu giá cổ phiếu và tính toán các chỉ số tài chính

```python
import numpy as np
import matplotlib.pyplot as plt

# Tạo dữ liệu giá cổ phiếu trong 252 ngày giao dịch (1 năm)
np.random.seed(456)
n_days = 252
initial_price = 100

# Tạo dữ liệu random walk cho giá cổ phiếu
daily_returns = np.random.normal(0.0008, 0.02, n_days)  # Trung bình 0.08% mỗi ngày
prices = np.zeros(n_days)
prices[0] = initial_price

for i in range(1, n_days):
    prices[i] = prices[i-1] * (1 + daily_returns[i])

# Tạo volume giao dịch
volumes = np.random.lognormal(mean=10, sigma=0.5, size=n_days).astype(int) * 1000

print(f"Dữ liệu {n_days} ngày giao dịch")
print(f"Giá khởi đầu: ${initial_price}")
print(f"Giá cuối năm: ${prices[-1]:.2f}")

# TODO 1: Phân tích cơ bản (8 phút)
print("\n=== PHÂN TÍCH CƠ BẢN ===")
# Giá cao nhất, thấp nhất trong năm
# Tỷ suất sinh lời tổng cộng (Total Return)
# Tỷ suất sinh lời trung bình hằng ngày
# Volatility (độ biến động) = std của daily returns
# Sharpe Ratio (giả sử risk-free rate = 2%/năm)

# TODO 2: Phân tích kỹ thuật (10 phút)
print("\n=== PHÂN TÍCH KỸ THUẬT ===")
# Tính Moving Average 20 ngày và 50 ngày
# Bollinger Bands (MA ± 2*STD)  
# RSI đơn giản (Relative Strength Index)
# Tìm ngày có volume giao dịch bất thường (> 2*std)

# TODO 3: Phân tích rủi ro (7 phút)
print("\n=== PHÂN TÍCH RỦI RO ===")
# Value at Risk (VaR) 95% và 99%
# Maximum Drawdown (sụt giảm lớn nhất từ đỉnh)
# Tính toán rolling volatility 30 ngày
# Phân tích phân phối returns (skewness, kurtosis)
```

#### Đáp án:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Tạo dữ liệu
np.random.seed(456)
n_days = 252
initial_price = 100

daily_returns = np.random.normal(0.0008, 0.02, n_days)
prices = np.zeros(n_days)
prices[0] = initial_price

for i in range(1, n_days):
    prices[i] = prices[i-1] * (1 + daily_returns[i])

volumes = np.random.lognormal(mean=10, sigma=0.5, size=n_days).astype(int) * 1000

print(f"Dữ liệu {n_days} ngày giao dịch")
print(f"Giá đầu năm: ${prices[0]:.2f}")
print(f"Giá cuối năm: ${prices[-1]:.2f}")

# 1. Phân tích cơ bản
print("\n=== PHÂN TÍCH CƠ BẢN ===")
max_price = np.max(prices)
min_price = np.min(prices)
max_day = np.argmax(prices) + 1
min_day = np.argmin(prices) + 1

print(f"Giá cao nhất: ${max_price:.2f} (ngày {max_day})")
print(f"Giá thấp nhất: ${min_price:.2f} (ngày {min_day})")

total_return = (prices[-1] - prices[0]) / prices[0] * 100
print(f"Tổng tỷ suất sinh lời: {total_return:.2f}%")

avg_daily_return = np.mean(daily_returns) * 100
print(f"Tỷ suất sinh lời trung bình hàng ngày: {avg_daily_return:.4f}%")

volatility = np.std(daily_returns) * np.sqrt# BÀI HỌC: PYTHON CƠ BẢN CHO PHÂN TÍCH DỮ LIỆU

**Thời gian:** 150 phút (2.5 giờ)  
**Đối tượng:** Học viên có kiến thức lập trình C# cơ bản  
**Phân bổ thời gian:** Lý thuyết 50 phút + Thực hành 100 phút

---

## MỤC TIÊU BÀI HỌC

### Kiến thức:
- Hiểu được syntax cơ bản của Python và sự khác biệt với C#
- Nắm vững các kiểu dữ liệu và cấu trúc dữ liệu trong Python
- Hiểu về thư viện NumPy và Pandas cho phân tích dữ liệu
- Biết cách sử dụng Matplotlib cho visualization cơ bản

### Kỹ năng:
- Viết được code Python cơ bản cho xử lý dữ liệu
- Sử dụng được Pandas để đọc, xử lý và phân tích dữ liệu
- Tạo được biểu đồ đơn giản với Matplotlib
- Thực hiện được các phép toán thống kê cơ bản

---

## PHẦN I: LỶ THUYẾT (50 PHÚT)

### 1. So sánh Python và C# (10 phút)

#### Điểm khác biệt chính:

**C# (Quen thuộc)**
```csharp
// Khai báo biến có kiểu
int number = 10;
string name = "John";

// Method với access modifier
public static void Main(string[] args)
{
    Console.WriteLine("Hello World");
}

// Strongly typed
List<int> numbers = new List<int>();
```

**Python (Mới)**
```python
# Khai báo biến không cần kiểu
number = 10
name = "John"

# Function đơn giản hơn
def main():
    print("Hello World")

# Dynamic typing
numbers = []  # có thể chứa bất kỳ kiểu nào
```

#### Ưu điểm của Python cho Data Analysis:
- Syntax đơn giản, dễ đọc
- Thư viện phong phú cho data science
- Interactive development (Jupyter Notebook)
- Cộng đồng lớn và tài liệu phong phú

### 2. Syntax cơ bản Python (15 phút)

#### 2.1 Biến và Kiểu dữ liệu
```python
# Kiểu số
age = 25          # int
price = 19.99     # float
is_student = True # bool

# Kiểu chuỗi
name = "Nguyen Van A"
description = '''Đây là chuỗi
nhiều dòng'''

# Kiểu None (tương đương null trong C#)
data = None
```

#### 2.2 Collections
```python
# List (tương đương List<object> trong C#)
fruits = ["apple", "banana", "orange"]
mixed_list = [1, "hello", True, 3.14]

# Dictionary (tương đương Dictionary<string, object>)
person = {
    "name": "John",
    "age": 30,
    "city": "Hanoi"
}

# Tuple (immutable)
coordinates = (10, 20)
```

#### 2.3 Control Flow
```python
# If statement (không cần dấu ngoặc và dấu chấm phẩy)
if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")

# For loop
for fruit in fruits:
    print(fruit)

# While loop
count = 0
while count < 5:
    print(count)
    count += 1
```

### 3. Thư viện cho Data Analysis (15 phút)

#### 3.1 NumPy - Xử lý mảng số học
```python
import numpy as np

# Tạo mảng
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])

# Các phép toán
print(arr.mean())  # Trung bình
print(arr.sum())   # Tổng
print(arr.std())   # Độ lệch chuẩn
```

#### 3.2 Pandas - Xử lý dữ liệu có cấu trúc
```python
import pandas as pd

# DataFrame (giống DataTable trong C#)
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['Hanoi', 'HCMC', 'Danang']
})

# Đọc dữ liệu từ file
df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')
```

#### 3.3 Matplotlib - Visualization
```python
import matplotlib.pyplot as plt

# Biểu đồ đường
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Sample Plot')
plt.show()

# Biểu đồ cột
plt.bar(['A', 'B', 'C'], [1, 3, 2])
plt.show()
```

### 4. Jupyter Notebook (10 phút)

#### Ưu điểm:
- Chạy code từng cell
- Kết hợp code, text và visualization
- Dễ dàng thử nghiệm và debug

#### Các phím tắt quan trọng:
- `Shift + Enter`: Chạy cell hiện tại
- `Ctrl + Enter`: Chạy cell không di chuyển
- `A`: Thêm cell ở trên
- `B`: Thêm cell ở dưới
- `M`: Chuyển sang Markdown
- `Y`: Chuyển sang Code

---

## PHẦN II: THỰC HÀNH (100 PHÚT)

### Bài tập 1: Làm quen với Python syntax (20 phút)

#### Yêu cầu:
Viết chương trình Python thực hiện các nhiệm vụ sau:

1. **Tạo danh sách sinh viên (5 phút)**
```python
# TODO: Tạo danh sách sinh viên với thông tin: tên, tuổi, điểm
students = [
    {"name": "Nguyen Van A", "age": 20, "score": 8.5},
    {"name": "Tran Thi B", "age": 19, "score": 9.0},
    {"name": "Le Van C", "age": 21, "score": 7.5}
]
```

2. **Xử lý dữ liệu cơ bản (10 phút)**
```python
# TODO: Viết function tính điểm trung bình
def calculate_average_score(students):
    # Gợi ý: sử dụng sum() và len()
    pass

# TODO: Tìm sinh viên có điểm cao nhất
def find_top_student(students):
    # Gợi ý: sử dụng max() với key parameter
    pass

# TODO: Lọc sinh viên trên 18 tuổi
def filter_adult_students(students):
    # Gợi ý: sử dụng list comprehension
    pass
```

3. **In kết quả (5 phút)**
```python
# TODO: In thông tin theo format đẹp
print(f"Điểm trung bình lớp: {calculate_average_score(students):.2f}")
print(f"Sinh viên xuất sắc nhất: {find_top_student(students)['name']}")
print("Danh sách sinh viên trưởng thành:")
for student in filter_adult_students(students):
    print(f"- {student['name']} ({student['age']} tuổi)")
```

#### Đáp án:
```python
students = [
    {"name": "Nguyen Van A", "age": 20, "score": 8.5},
    {"name": "Tran Thi B", "age": 19, "score": 9.0},
    {"name": "Le Van C", "age": 21, "score": 7.5}
]

def calculate_average_score(students):
    total_score = sum(student["score"] for student in students)
    return total_score / len(students)

def find_top_student(students):
    return max(students, key=lambda x: x["score"])

def filter_adult_students(students):
    return [student for student in students if student["age"] >= 18]

# In kết quả
print(f"Điểm trung bình lớp: {calculate_average_score(students):.2f}")
print(f"Sinh viên xuất sắc nhất: {find_top_student(students)['name']}")
print("Danh sách sinh viên trưởng thành:")
for student in filter_adult_students(students):
    print(f"- {student['name']} ({student['age']} tuổi)")
```

### Bài tập 2: Làm việc với NumPy (20 phút)

#### Yêu cầu:
Phân tích dữ liệu bán hàng của một cửa hàng

```python
import numpy as np

# Dữ liệu bán hàng 30 ngày (triệu đồng)
sales_data = np.array([
    12.5, 15.3, 18.2, 14.7, 16.8, 13.9, 17.4, 19.1, 15.6, 16.2,
    14.3, 18.7, 16.9, 15.8, 17.2, 13.4, 16.5, 18.9, 15.1, 17.6,
    14.8, 16.3, 18.1, 15.9, 17.8, 16.4, 18.5, 15.2, 17.1, 16.7
])

# TODO 1: Tính các thống kê cơ bản
print("=== THỐNG KÊ BÁN HÀNG ===")
# Doanh thu trung bình
# Doanh thu cao nhất và thấp nhất
# Độ lệch chuẩn
# Tổng doanh thu

# TODO 2: Phân tích theo tuần (mỗi tuần 7 ngày)
# Reshape dữ liệu thành ma trận 4x7 (4 tuần, 7 ngày)
# Tính doanh thu trung bình mỗi tuần
# Tìm tuần có doanh thu cao nhất

# TODO 3: Phân tích ngày trong tuần
# Tính doanh thu trung bình theo ngày trong tuần
# Tìm ngày bán chạy nhất trong tuần
```

#### Đáp án:
```python
import numpy as np

sales_data = np.array([
    12.5, 15.3, 18.2, 14.7, 16.8, 13.9, 17.4, 19.1, 15.6, 16.2,
    14.3, 18.7, 16.9, 15.8, 17.2, 13.4, 16.5, 18.9, 15.1, 17.6,
    14.8, 16.3, 18.1, 15.9, 17.8, 16.4, 18.5, 15.2, 17.1, 16.7
])

# Thống kê cơ bản
print("=== THỐNG KÊ BÁN HÀNG ===")
print(f"Doanh thu trung bình: {sales_data.mean():.2f} triệu")
print(f"Doanh thu cao nhất: {sales_data.max():.2f} triệu")
print(f"Doanh thu thấp nhất: {sales_data.min():.2f} triệu")
print(f"Độ lệch chuẩn: {sales_data.std():.2f} triệu")
print(f"Tổng doanh thu: {sales_data.sum():.2f} triệu")

# Phân tích theo tuần
weekly_data = sales_data[:28].reshape(4, 7)  # Lấy 28 ngày đầu
weekly_avg = weekly_data.mean(axis=1)
print(f"\nDoanh thu trung bình theo tuần:")
for i, avg in enumerate(weekly_avg):
    print(f"Tuần {i+1}: {avg:.2f} triệu")
print(f"Tuần có doanh thu cao nhất: Tuần {weekly_avg.argmax() + 1}")

# Phân tích ngày trong tuần
daily_avg = weekly_data.mean(axis=0)
days = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'CN']
print(f"\nDoanh thu trung bình theo ngày trong tuần:")
for day, avg in zip(days, daily_avg):
    print(f"{day}: {avg:.2f} triệu")
print(f"Ngày bán chạy nhất: {days[daily_avg.argmax()]}")
```

### Bài tập 3: Pandas DataFrame (30 phút)

#### Yêu cầu:
Phân tích dữ liệu khách hàng của một cửa hàng online

```python
import pandas as pd
import numpy as np

# Tạo dữ liệu mẫu
np.random.seed(42)
customers_data = {
    'CustomerID': range(1001, 1101),
    'Name': [f'Customer_{i}' for i in range(1, 101)],
    'Age': np.random.randint(18, 65, 100),
    'City': np.random.choice(['Hanoi', 'HCMC', 'Danang', 'Haiphong', 'Cantho'], 100),
    'TotalSpent': np.random.uniform(100, 5000, 100),
    'OrderCount': np.random.randint(1, 20, 100),
    'LastOrderDate': pd.date_range('2023-01-01', periods=100, freq='D')
}

df = pd.DataFrame(customers_data)

# TODO 1: Khám phá dữ liệu cơ bản (5 phút)
print("=== KHÁM PHÁ DỮ LIỆU ===")
# In 5 dòng đầu
# Thông tin về DataFrame (shape, columns, dtypes)
# Thống kê mô tả

# TODO 2: Phân tích theo thành phố (10 phút)
print("\n=== PHÂN TÍCH THEO THÀNH PHỐ ===")
# Số lượng khách hàng mỗi thành phố
# Tổng chi tiêu theo thành phố
# Khách hàng chi tiêu cao nhất mỗi thành phố

# TODO 3: Phân tích theo độ tuổi (10 phút)
print("\n=== PHÂN TÍCH THEO ĐỘ TUỔI ===")
# Tạo nhóm tuổi: 18-25, 26-35, 36-45, 46+
# Chi tiêu trung bình theo nhóm tuổi
# Số đơn hàng trung bình theo nhóm tuổi

# TODO 4: Tìm top khách hàng (5 phút)
print("\n=== TOP KHÁCH HÀNG ===")
# Top 10 khách hàng chi tiêu nhiều nhất
# Top 10 khách hàng có nhiều đơn hàng nhất
# Khách hàng có giá trị đơn hàng trung bình cao nhất
```

#### Đáp án:
```python
import pandas as pd
import numpy as np

# Tạo dữ liệu
np.random.seed(42)
customers_data = {
    'CustomerID': range(1001, 1101),
    'Name': [f'Customer_{i}' for i in range(1, 101)],
    'Age': np.random.randint(18, 65, 100),
    'City': np.random.choice(['Hanoi', 'HCMC', 'Danang', 'Haiphong', 'Cantho'], 100),
    'TotalSpent': np.random.uniform(100, 5000, 100).round(2),
    'OrderCount': np.random.randint(1, 20, 100),
    'LastOrderDate': pd.date_range('2023-01-01', periods=100, freq='D')
}

df = pd.DataFrame(customers_data)

# 1. Khám phá dữ liệu cơ bản
print("=== KHÁM PHÁ DỮ LIỆU ===")
print("5 dòng đầu:")
print(df.head())
print(f"\nKích thước: {df.shape}")
print(f"Cột: {list(df.columns)}")
print("\nThống kê mô tả:")
print(df.describe())

# 2. Phân tích theo thành phố
print("\n=== PHÂN TÍCH THEO THÀNH PHỐ ===")
city_analysis = df.groupby('City').agg({
    'CustomerID': 'count',
    'TotalSpent': ['sum', 'mean'],
    'OrderCount': 'mean'
}).round(2)

print("Số lượng khách hàng theo thành phố:")
print(df['City'].value_counts())
print("\nTổng chi tiêu theo thành phố:")
print(df.groupby('City')['TotalSpent'].sum().sort_values(ascending=False))

# Top spender mỗi thành phố
print("\nKhách hàng chi tiêu cao nhất mỗi thành phố:")
top_by_city = df.loc[df.groupby('City')['TotalSpent'].idxmax()]
print(top_by_city[['City', 'Name', 'TotalSpent']])

# 3. Phân tích theo độ tuổi
print("\n=== PHÂN TÍCH THEO ĐỘ TUỔI ===")
df['AgeGroup'] = pd.cut(df['Age'], 
                       bins=[17, 25, 35, 45, 100], 
                       labels=['18-25', '26-35', '36-45', '46+'])

age_analysis = df.groupby('AgeGroup').agg({
    'TotalSpent': 'mean',
    'OrderCount': 'mean'
}).round(2)
print("Chi tiêu và đơn hàng trung bình theo nhóm tuổi:")
print(age_analysis)

# 4. Top khách hàng
print("\n=== TOP KHÁCH HÀNG ===")
print("Top 10 khách hàng chi tiêu nhiều nhất:")
top_spenders = df.nlargest(10, 'TotalSpent')[['Name', 'City', 'TotalSpent']]
print(top_spenders)

print("\nTop 10 khách hàng có nhiều đơn hàng nhất:")
top_buyers = df.nlargest(10, 'OrderCount')[['Name', 'City', 'OrderCount']]
print(top_buyers)

# Giá trị đơn hàng trung bình
df['AvgOrderValue'] = (df['TotalSpent'] / df['OrderCount']).round(2)
print("\nTop 10 khách hàng có giá trị đơn hàng trung bình cao nhất:")
top_aov = df.nlargest(10, 'AvgOrderValue')[['Name', 'City', 'AvgOrderValue']]
print(top_aov)
```

### Bài tập 4: Data Visualization (30 phút)

#### Yêu cầu:
Tạo các biểu đồ trực quan hóa dữ liệu khách hàng

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sử dụng dữ liệu từ bài tập 3
# (Code tạo dữ liệu như trên)

# TODO 1: Biểu đồ cột - Số khách hàng theo thành phố (8 phút)
plt.figure(figsize=(10, 6))
# Tạo biểu đồ cột
# Thêm title, labels
# Xoay labels nếu cần
# Thêm giá trị trên mỗi cột

# TODO 2: Biểu đồ histogram - Phân bố độ tuổi (8 phút)
plt.figure(figsize=(10, 6))
# Tạo histogram với 15 bins
# Thêm đường trung bình
# Thêm title và labels

# TODO 3: Biểu đồ scatter - Mối quan hệ Age vs TotalSpent (8 phút)
plt.figure(figsize=(10, 6))
# Tạo scatter plot
# Color theo City
# Thêm trendline
# Legend và labels

# TODO 4: Subplots - Kết hợp nhiều biểu đồ (6 phút)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
# Biểu đồ 1: Bar chart - TotalSpent theo City
# Biểu đồ 2: Pie chart - Phân bố City  
# Biểu đồ 3: Box plot - TotalSpent theo AgeGroup
# Biểu đồ 4: Line plot - Xu hướng đặt hàng theo thời gian
```

#### Đáp án:
```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sử dụng dữ liệu từ bài tập 3
np.random.seed(42)
customers_data = {
    'CustomerID': range(1001, 1101),
    'Name': [f'Customer_{i}' for i in range(1, 101)],
    'Age': np.random.randint(18, 65, 100),
    'City': np.random.choice(['Hanoi', 'HCMC', 'Danang', 'Haiphong', 'Cantho'], 100),
    'TotalSpent': np.random.uniform(100, 5000, 100).round(2),
    'OrderCount': np.random.randint(1, 20, 100),
    'LastOrderDate': pd.date_range('2023-01-01', periods=100, freq='D')
}
df = pd.DataFrame(customers_data)

# 1. Biểu đồ cột - Số khách hàng theo thành phố
plt.figure(figsize=(10, 6))
city_counts = df['City'].value_counts()
bars = plt.bar(city_counts.index, city_counts.values, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
plt.title('Số lượng khách hàng theo thành phố', fontsize=16, fontweight='bold')
plt.xlabel('Thành phố')
plt.ylabel('Số lượng khách hàng')
plt.xticks(rotation=45)

# Thêm giá trị trên mỗi cột
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 2. Biểu đồ histogram - Phân bố độ tuổi
plt.figure(figsize=(10, 6))
plt.hist(df['Age'], bins=15, color='skyblue', alpha=0.7, edgecolor='black')
plt.axvline(df['Age'].mean(), color='red', linestyle='--', 
            label=f'Trung bình: {df["Age"].mean():.1f} tuổi')
plt.title('Phân bố độ tuổi khách hàng', fontsize=16, fontweight='bold')
plt.xlabel('Độ tuổi')
plt.ylabel('Số lượng khách hàng')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Biểu đồ scatter - Age vs TotalSpent
plt.figure(figsize=(12, 8))
colors = {'Hanoi': 'red', 'HCMC': 'blue', 'Danang': 'green', 
          'Haiphong': 'orange', 'Cantho': 'purple'}

for city in df['City'].unique():
    city_data = df[df['City'] == city]
    plt.scatter(city_data['Age'], city_data['TotalSpent'], 
               c=colors[city], label=city, alpha=0.6, s=60)

# Trendline
z = np.polyfit(df['Age'], df['TotalSpent'], 1)
p = np.poly1d(z)
plt.plot(df['Age'], p(df['Age']), "r--", alpha=0.8, linewidth=2)

plt.title('Mối quan hệ giữa độ tuổi và tổng chi tiêu', fontsize=16, fontweight='bold')
plt.xlabel('Độ tuổi')
plt.ylabel('Tổng chi tiêu (VNĐ)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Subplots - Kết hợp nhiều biểu đồ
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Biểu đồ 1: Bar chart - TotalSpent theo City
city_spending = df.groupby('City')['TotalSpent'].sum()
axes[0, 0].bar(city_spending.index, city_spending.values, color='lightcoral')
axes[0, 0].set_title('Tổng chi tiêu theo thành phố')
axes[0, 0].tick_params(axis='x', rotation=45)

# Biểu đồ 2: Pie chart - Phân bố City
city_counts = df['City'].value_counts()
axes[0, 1].pie(city_counts.values, labels=city_counts.index, autopct='%1.1f%%')
axes[0, 1].set_title('Phân bố khách hàng theo thành phố')

# Biểu đồ 3: Box plot - TotalSpent theo AgeGroup
df['AgeGroup'] = pd.cut(df['Age'], bins=[17, 25, 35, 45, 100], 
                       labels=['18-25', '26-35', '36-45', '46+'])
age_groups = [df[df['AgeGroup'] == group]['TotalSpent'].values 
              for group in df['AgeGroup'].cat.categories]
axes[1, 0].boxplot(age_groups, labels=df['AgeGroup'].cat.categories)
axes[1, 0].set_title('Phân bố chi tiêu theo nhóm tuổi')
axes[1, 0].tick_params(axis='x', rotation=45)

# Biểu đồ 4: Line plot - Xu hướng theo thời gian
monthly_orders = df.groupby(df['LastOrderDate'].dt.month)['OrderCount'].sum()
axes[1, 1].plot(monthly_orders.index, monthly_orders.values, 
                marker='o', linewidth=2, markersize=6)
axes[1, 1].set_title('Xu hướng đơn hàng theo tháng')
axes[1, 1].set_xlabel('Tháng')
axes[1, 1].set_ylabel('Tổng số đơn hàng')

plt.tight_layout()
plt.show()
```

---

## ĐÁNH GIÁ VÀ TỔNG KẾT

### Tiêu chí đánh giá:
- **Kiến thức lý thuyết (30%)**: Hiểu syntax Python, khác biệt với C#
- **Thực hành cơ bản (25%)**: Bài tập 1 - Python syntax
- **NumPy (20%)**: Bài tập 2 - Xử lý mảng và tính toán
- **Pandas (15%)**: Bài tập 3 - DataFrame operations
- **Visualization (10%)**: Bài tập 4 - Matplotlib

### Câu hỏi ôn tập:
1. So sánh cách khai báo biến trong Python và C#?
2. Pandas DataFrame khác gì với DataTable trong C#?
3. Khi nào nên dùng NumPy, khi nào nên dùng Pandas?
4. Làm thế nào để đọc file CSV vào Python?
5. Cách tạo biểu đồ cơ bản với Matplotlib?

### Bài tập về nhà:

## BÀI TẬP PYTHON SYNTAX

### Bài tập về nhà 1: Hệ thống quản lý thư viện (45 phút)

**Mục tiêu:** Áp dụng OOP và collections để xây dựng hệ thống quản lý thư viện

**Yêu cầu:**
1. Tạo class `Book` với các thuộc tính: ISBN, title, author, year, available_copies
2. Tạo class `Member` với các thuộc tính: member_id, name, borrowed_books (list)
3. Tạo class `Library` để quản lý sách và thành viên:
   - Thêm/xóa sách
   - Đăng ký thành viên mới
   - Mượn/trả sách
   - Tìm kiếm sách theo tác giả, tựa đề
   - Thống kê sách được mượn nhiều nhất
   - Xuất báo cáo thành viên quá hạn

**Gợi ý cấu trúc:**
```python
class Book:
    def __init__(self, isbn, title, author, year, copies):
        pass
    
    def __str__(self):
        pass

class Member:
    def __init__(self, member_id, name):
        pass
    
    def borrow_book(self, book):
        pass
    
    def return_book(self, book):
        pass

class Library:
    def __init__(self):
        self.books = {}  # ISBN -> Book object
        self.members = {}  # member_id -> Member object
        self.borrowed_history = []  # Lịch sử mượn trả
    
    def add_book(self, book):
        pass
    
    def register_member(self, member):
        pass
    
    def borrow_book(self, member_id, isbn):
        pass
    
    def return_book(self, member_id, isbn):
        pass
    
    def search_books_by_author(self, author):
        pass
    
    def get_most_popular_books(self, top_n=5):
        pass
    
    def generate_member_report(self):
        pass

# Test case
if __name__ == "__main__":
    library = Library()
    
    # Thêm sách mẫu
    books = [
        Book("978-0134685991", "Effective Python", "Brett Slatkin", 2019, 3),
        Book("978-1491946008", "Fluent Python", "Luciano Ramalho", 2015, 2),
        Book("978-0596009267", "Python Cookbook", "David Beazley", 2013, 4)
    ]
    
    for book in books:
        library.add_book(book)
    
    # Test các chức năng...
```

### Bài tập về nhà 2: Máy tính toán học nâng cao (40 phút)

**Mục tiêu:** Sử dụng functions, decorators, và error handling

**Yêu cầu:**
1. Tạo các function tính toán cơ bản và nâng cao
2. Sử dụng decorators để log thời gian thực hiện và validate input
3. Xử lý exception cho các trường hợp lỗi
4. Tạo calculator với history và các chức năng khôi phục

**Gợi ý cấu trúc:**
```python
import functools
import time
import math
from typing import Union, List, Callable

def timer_decorator(func):
    """Decorator để đo thời gian thực hiện function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # TODO: Implement timer logic
        pass
    return wrapper

def validate_numbers(func):
    """Decorator để validate input là số"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # TODO: Implement validation logic
        pass
    return wrapper

class AdvancedCalculator:
    def __init__(self):
        self.history = []
        self.memory = 0
    
    @timer_decorator
    @validate_numbers
    def add(self, a: float, b: float) -> float:
        pass
    
    @timer_decorator
    @validate_numbers
    def factorial(self, n: int) -> int:
        pass
    
    @timer_decorator
    @validate_numbers
    def fibonacci(self, n: int) -> List[int]:
        pass
    
    @timer_decorator
    @validate_numbers
    def solve_quadratic(self, a: float, b: float, c: float) -> tuple:
        """Giải phương trình bậc 2: ax² + bx + c = 0"""
        pass
    
    @timer_decorator
    def prime_factors(self, n: int) -> List[int]:
        """Phân tích số nguyên thành thừa số nguyên tố"""
        pass
    
    def save_to_history(self, operation: str, result: Union[float, int, list]):
        pass
    
    def get_history(self, last_n: int = 10) -> List[dict]:
        pass
    
    def clear_history(self):
        pass
    
    def memory_store(self, value: float):
        pass
    
    def memory_recall(self) -> float:
        pass

# Test cases
calc = AdvancedCalculator()
print(calc.add(10, 5))
print(calc.factorial(5))
print(calc.fibonacci(10))
print(calc.solve_quadratic(1, -5, 6))  # x² - 5x + 6 = 0
print(calc.prime_factors(60))
print(calc.get_history())
```

### Bài tập về nhà 3: Web Scraper và Data Parser (50 phút)

**Mục tiêu:** Xử lý string, regex, file I/O, và data structures phức tạp

**Yêu cầu:**
1. Tạo parser để xử lý dữ liệu từ các file format khác nhau
2. Sử dụng regex để extract thông tin
3. Implement caching và error recovery
4. Tạo output ở nhiều format khác nhau

**Gợi ý cấu trúc:**
```python
import re
import json
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Any
import urllib.request
from pathlib import Path

class DataParser:
    def __init__(self):
        self.cache = {}
        self.parsed_data = []
    
    def parse_log_file(self, file_path: str) -> List[Dict]:
        """
        Parse Apache/Nginx log file
        Format: IP - - [timestamp] "method URL protocol" status size "referer" "user-agent"
        """
        log_pattern = r'(\d+\.\d+\.\d+\.\d+).*\[(.*?)\] "(.*?)" (\d+) (\d+)'
        # TODO: Implement log parsing
        pass
    
    def parse_csv_with_validation(self, file_path: str, required_columns: List[str]) -> List[Dict]:
        """Parse CSV với validation columns"""
        # TODO: Implement CSV parsing with validation
        pass
    
    def extract_emails_phones(self, text: str) -> Dict[str, List[str]]:
        """Extract email addresses và phone numbers từ text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+84|0)[3|5|7|8|9][0-9]{8}'
        # TODO: Implement extraction
        pass
    
    def parse_xml_config(self, file_path: str) -> Dict[str, Any]:
        """Parse XML configuration file"""
        # TODO: Implement XML parsing
        pass
    
    def clean_and_normalize_data(self, data: List[Dict]) -> List[Dict]:
        """
        Clean data: remove duplicates, normalize formats, handle missing values
        """
        # TODO: Implement data cleaning
        pass
    
    def export_to_format(self, data: List[Dict], format_type: str, output_path: str):
        """Export data to JSON, CSV, or XML format"""
        if format_type.lower() == 'json':
            # TODO: Export to JSON
            pass
        elif format_type.lower() == 'csv':
            # TODO: Export to CSV
            pass
        elif format_type.lower() == 'xml':
            # TODO: Export to XML
            pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about parsed data"""
        # TODO: Calculate various statistics
        pass

class WebDataCollector(DataParser):
    def __init__(self):
        super().__init__()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def fetch_and_parse_rss(self, rss_url: str) -> List[Dict]:
        """Fetch và parse RSS feed"""
        # TODO: Implement RSS parsing
        pass
    
    def parse_html_table(self, html_content: str, table_selector: str = None) -> List[Dict]:
        """Extract data from HTML table"""
        # TODO: Implement HTML table parsing (without BeautifulSoup)
        pass

# Test cases
if __name__ == "__main__":
    parser = DataParser()
    
    # Test với sample data
    sample_text = """
    Contact us at: john@example.com or call 0901234567
    Another email: admin@test.org, phone: +84987654321
    """
    
    contacts = parser.extract_emails_phones(sample_text)
    print("Extracted contacts:", contacts)
    
    # Test CSV parsing
    # parser.parse_csv_with_validation('sample.csv', ['name', 'email', 'age'])
    
    # Test export
    # sample_data = [{'name': 'John', 'age': 30}, {'name': 'Jane', 'age': 25}]
    # parser.export_to_format(sample_data, 'json', 'output.json')
```

## BÀI TẬP NUMPY

### Bài tập về nhà 4: Phân tích dữ liệu sensor IoT (60 phút)

**Mục tiêu:** Xử lý time series data và tính toán thống kê nâng cao với NumPy

**Yêu cầu:**
Phân tích dữ liệu từ hệ thống sensors theo dõi môi trường (nhiệt độ, độ ẩm, áp suất) trong 30 ngày

```python
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Tạo dữ liệu mô phỏng
np.random.seed(789)
n_days = 30
n_hours_per_day = 24
n_sensors = 5
sensor_names = ['Living Room', 'Bedroom', 'Kitchen', 'Bathroom', 'Garden']

# TODO 1: Tạo dữ liệu time series (15 phút)
def generate_sensor_data():
    """
    Tạo dữ liệu 3D array: (days, hours, sensors, measurements)
    measurements: [temperature, humidity, pressure]
    
    Yêu cầu:
    - Temperature: 20-35°C với cycle ngày/đêm
    - Humidity: 30-80% với noise
    - Pressure: 990-1020 hPa với trend theo thời tiết
    - Garden sensor có pattern khác (ảnh hưởng thời tiết ngoài trời)
    """
    pass

# TODO 2: Phân tích anomaly detection (20 phút)
def detect_anomalies(data, threshold=2.5):
    """
    Detect anomalies using statistical methods:
    - Z-score method
    - IQR method  
    - Moving average deviation
    - Return indices of anomalous readings
    """
    pass

def analyze_sensor_correlation(data):
    """
    Phân tích correlation giữa các sensors:
    - Correlation matrix giữa sensors
    - Cross-correlation theo thời gian
    - Tìm sensors có pattern tương đồng nhất
    """
    pass

# TODO 3: Phân tích patterns và trends (15 phút)
def analyze_daily_patterns(data):
    """
    Phân tích pattern hàng ngày:
    - Average hourly pattern cho mỗi sensor
    - Identify peak hours cho từng measurement
    - Weekly patterns (if data spans weeks)
    """
    pass

def calculate_comfort_index(temperature, humidity):
    """
    Tính chỉ số comfort dựa trên temperature và humidity:
    Comfort Index = T - 0.4(T - 10)(1 - H/100)
    Với T = temperature (°C), H = humidity (%)
    
    Categories:
    - Comfortable: 18-24
    - Slightly uncomfortable: 24-27 or 15-18  
    - Uncomfortable: >27 or <15
    """
    pass

# TODO 4: Advanced analytics (10 phút)
def predict_next_day_average(data):
    """
    Simple prediction cho ngày tiếp theo:
    - Linear trend extrapolation
    - Moving average method
    - Seasonal decomposition
    """
    pass

def calculate_energy_efficiency_score(temperature_data, target_temp=23):
    """
    Tính efficiency score dựa trên độ lệch từ target temperature:
    - Score = 100 - average_absolute_deviation_from_target
    - Bonus points for stable temperature (low std)
    """
    pass

# Main analysis
if __name__ == "__main__":
    # Generate data
    sensor_data = generate_sensor_data()
    
    print("=== SENSOR DATA ANALYSIS ===")
    print(f"Data shape: {sensor_data.shape}")
    
    # Detect anomalies
    anomalies = detect_anomalies(sensor_data)
    print(f"Detected {len(anomalies)} anomalies")
    
    # Correlation analysis
    correlations = analyze_sensor_correlation(sensor_data)
    
    # Daily patterns
    daily_stats = analyze_daily_patterns(sensor_data)
    
    # Comfort analysis
    for sensor_idx, sensor_name in enumerate(sensor_names):
        temp_data = sensor_data[:, :, sensor_idx, 0]
        humid_data = sensor_data[:, :, sensor_idx, 1]
        comfort = calculate_comfort_index(temp_data, humid_data)
        efficiency = calculate_energy_efficiency_score(temp_data)
        
        print(f"\n{sensor_name}:")
        print(f"  Average comfort index: {np.mean(comfort):.2f}")
        print(f"  Energy efficiency score: {efficiency:.2f}")
    
    # Predictions
    predictions = predict_next_day_average(sensor_data)
    print(f"\nNext day predictions: {predictions}")
```

### Bài tập về nhà 5: Phân tích hình ảnh với NumPy (50 phút)

**Mục tiêu:** Xử lý và phân tích dữ liệu hình ảnh sử dụng NumPy arrays

**Yêu cầu:**
Tạo các functions để xử lý ảnh cơ bản và phân tích đặc trưng hình ảnh

```python
import numpy as np
import matplotlib.pyplot as plt

# TODO 1: Image generation và basic operations (15 phút)
def create_test_images():
    """
    Tạo các test images:
    - Gradient image (256x256)
    - Checkerboard pattern (256x256)  
    - Circle with noise (256x256)
    - Random texture (256x256)
    Return dictionary of images
    """
    images = {}
    
    # Gradient image (0-255 từ trái sang phải)
    # TODO: Implement gradient
    
    # Checkerboard pattern (8x8 squares)
    # TODO: Implement checkerboard
    
    # Circle với gaussian noise
    # TODO: Implement noisy circle
    
    # Random texture
    # TODO: Implement random texture
    
    return images

def basic_image_operations(image):
    """
    Implement các operations cơ bản:
    - Rotate 90, 180, 270 degrees
    - Flip horizontal/vertical
    - Resize using nearest neighbor và bilinear interpolation
    - Crop center region
    """
    operations = {}
    
    # TODO: Implement each operation
    
    return operations

# TODO 2: Filters và convolution (20 phút)  
def apply_convolution(image, kernel):
    """
    Apply convolution filter (without using scipy):
    - Implement 2D convolution from scratch
    - Handle edge cases với padding
    """
    pass

def create_filters():
    """
    Tạo các common filters:
    - Edge detection (Sobel X, Sobel Y, Laplacian)
    - Blur (Gaussian, Box filter)
    - Sharpen filter
    - Custom filters
    """
    filters = {}
    
    # Sobel X filter
    filters['sobel_x'] = np.array([[-1, 0, 1],
                                   [-2, 0, 2], 
                                   [-1, 0, 1]])
    
    # TODO: Implement other filters
    
    return filters

def apply_all_filters(image, filters):
    """Apply tất cả filters và return results"""
    results = {}
    for filter_name, kernel in filters.items():
        results[filter_name] = apply_convolution(image, kernel)
    return results

# TODO 3: Image analysis và feature extraction (15 phút)
def calculate_image_statistics(image):
    """
    Tính toán image statistics:
    - Histogram (256 bins)
    - Mean, std, min, max intensity
    - Entropy (measure of randomness)
    - Contrast metrics
    - Local Binary Pattern features
    """
    stats = {}
    
    # Basic stats
    stats['mean'] = np.mean(image)
    stats['std'] = np.std(image)
    stats['min'] = np.min(image)
    stats['max'] = np.max(image)
    
    # Histogram
    stats['histogram'] = np.histogram(image.flatten(), bins=256, range=(0, 255))[0]
    
    # TODO: Implement entropy calculation
    # Entropy = -sum(p * log2(p)) where p is probability
    
    # TODO: Implement contrast metrics
    # RMS contrast = sqrt(mean((image - mean)^2))
    
    return stats

def detect_edges_and_corners(image):
    """
    Detect edges và corners:
    - Gradient magnitude và direction
    - Harris corner detection (simplified)
    - Edge linking
    """
    # TODO: Implement edge detection
    pass

def segment_image(image, method='threshold'):
    """
    Image segmentation:
    - Simple thresholding (Otsu's method)
    - Region growing (simplified)
    - Watershed algorithm (basic version)
    """
    if method == 'threshold':
        # TODO: Implement Otsu's thresholding
        pass
    elif method == 'region_growing':
        # TODO: Implement region growing
        pass

# Visualization helpers
def display_image_analysis(original, processed_images, statistics):
    """
    Create comprehensive visualization:
    - Original image
    - Processed versions
    - Histograms
    - Statistics plots
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # TODO: Implement visualization
    pass

# Main analysis
if __name__ == "__main__":
    print("=== IMAGE PROCESSING WITH NUMPY ===")
    
    # Create test images
    test_images = create_test_images()
    
    # Process each image
    for img_name, image in test_images.items():
        print(f"\nProcessing {img_name}...")
        
        # Basic operations
        operations = basic_image_operations(image)
        
        # Apply filters
        filters = create_filters()
        filtered_images = apply_all_filters(image, filters)
        
        # Calculate statistics
        stats = calculate_image_statistics(image)
        
        print(f"  Mean intensity: {stats['mean']:.2f}")
        print(f"  Std deviation: {stats['std']:.2f}")
        print(f"  Dynamic range: {stats['min']} - {stats['max']}")
        
        # Edge and corner detection
        edges_corners = detect_edges_and_corners(image)
        
        # Segmentation
        segmented = segment_image(image, method='threshold')
        
        # Visualization
        display_image_analysis(image, filtered_images, stats)
    
    plt.show()

# Bonus: Compare với built-in functions
def compare_with_builtin():
    """
    So sánh implementation tự viết với built-in functions:
    - scipy.ndimage
    - skimage functions
    - OpenCV equivalents
    """
    pass
```

---

**Lưu ý cho học viên:**
- Mỗi bài tập được thiết kế để mất 40-60 phút
- Nên làm từng phần một và test thoroughly
- Có thể tham khảo documentation nhưng cố gắng implement từ scratch
- Bài tập 4 và 5 khá nâng cao, có thể làm theo nhóm
- Submit code kèm theo screenshot/plots cho bài tập visualization

**Tiêu chí chấm điểm:**
- Code chạy được và cho kết quả đúng (40%)
- Code structure và readability (20%)
- Handle edge cases và error handling (20%) 
- Documentation và comments (10%)
- Creativity và bonus features (10%)

### Tài liệu tham khảo:
- **Python Official Documentation**: https://docs.python.org/3/
- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **NumPy Documentation**: https://numpy.org/doc/
- **Matplotlib Documentation**: https://matplotlib.org/stable/
- **Jupyter Notebook**: https://jupyter.org/documentation

---

## PHỤ LỤC: CODE TEMPLATES VÀ CHEAT SHEET

### A. Template cho Data Analysis Project

```python
# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATA
def load_data(file_path):
    """Load data from various formats"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)

# 2. EXPLORE DATA  
def explore_data(df):
    """Basic data exploration"""
    print("=== DATA INFO ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())
    print("\n=== DATA TYPES ===")
    print(df.dtypes)
    print("\n=== STATISTICAL SUMMARY ===")
    print(df.describe())

# 3. CLEAN DATA
def clean_data(df):
    """Basic data cleaning"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    # Option 1: Drop rows with missing values
    # df = df.dropna()
    
    # Option 2: Fill missing values
    # df = df.fillna(method='forward')
    
    return df

# 4. VISUALIZE DATA
def create_visualizations(df):
    """Create basic visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram for numeric column
    numeric_col = df.select_dtypes(include=[np.number]).columns[0]
    axes[0, 0].hist(df[numeric_col].dropna(), bins=20)
    axes[0, 0].set_title(f'Distribution of {numeric_col}')
    
    # Count plot for categorical column
    cat_col = df.select_dtypes(include=['object']).columns[0]
    df[cat_col].value_counts().plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title(f'Count of {cat_col}')
    
    plt.tight_layout()
    plt.show()

# Main analysis function
def main_analysis(file_path):
    """Main analysis pipeline"""
    # Load data
    df = load_data(file_path)
    
    # Explore data
    explore_data(df)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Visualize data
    create_visualizations(df_clean)
    
    return df_clean

# Usage
# df = main_analysis('your_data.csv')
```

### B. Python for C# Developers Cheat Sheet

| **Concept** | **C#** | **Python** |
|-------------|--------|------------|
| **Variables** | `int age = 25;` | `age = 25` |
| **String formatting** | `$"Hello {name}"` | `f"Hello {name}"` |
| **Arrays/Lists** | `int[] arr = {1,2,3};` | `arr = [1, 2, 3]` |
| **Dictionary** | `Dictionary<string, int>` | `dict = {'key': value}` |
| **For loop** | `for(int i=0; i<n; i++)` | `for i in range(n):` |
| **Functions** | `public int Add(int a, int b)` | `def add(a, b):` |
| **Classes** | `public class Person { }` | `class Person:` |
| **Null check** | `if (obj != null)` | `if obj is not None:` |
| **Exception handling** | `try { } catch { }` | `try: except:` |

### C. Pandas Quick Reference

```python
# READING DATA
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
df = pd.read_json('file.json')

# BASIC INFO
df.head()          # First 5 rows
df.tail()          # Last 5 rows  
df.info()          # Data types and null counts
df.describe()      # Statistical summary
df.shape           # (rows, columns)
df.columns         # Column names

# SELECTING DATA
df['column']       # Select single column
df[['col1', 'col2']]  # Select multiple columns
df.iloc[0]         # Select by position
df.loc[0]          # Select by index
df[df['age'] > 25] # Filter rows

# GROUPING AND AGGREGATION
df.groupby('column').sum()
df.groupby('column').agg({'col1': 'mean', 'col2': 'sum'})
df.pivot_table(values='value', index='row', columns='col')

# MISSING DATA
df.isnull().sum()  # Count missing values
df.dropna()        # Remove rows with missing values
df.fillna(value)   # Fill missing values

# SORTING
df.sort_values('column')
df.sort_values(['col1', 'col2'], ascending=[True, False])
```

### D. Common Matplotlib Patterns

```python
import matplotlib.pyplot as plt

# BASIC SETUP
plt.figure(figsize=(10, 6))
plt.style.use('seaborn')  # Use seaborn style

# LINE PLOT
plt.plot(x, y, marker='o', linestyle='-', color='blue', label='Data')
plt.title('Title')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.legend()
plt.grid(True)

# BAR PLOT
plt.bar(categories, values, color='skyblue')
plt.xticks(rotation=45)

# HISTOGRAM  
plt.hist(data, bins=20, alpha=0.7, color='green')

# SCATTER PLOT
plt.scatter(x, y, c=colors, s=sizes, alpha=0.6)
plt.colorbar()

# SUBPLOTS
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(x, y)
axes[0, 1].bar(x, y)
plt.tight_layout()

# SAVE FIGURE
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

### E. Error Handling Best Practices

```python
# FILE OPERATIONS
try:
    df = pd.read_csv('data.csv')
    print("Data loaded successfully")
except FileNotFoundError:
    print("File not found. Please check the file path.")
except pd.errors.EmptyDataError:
    print("The file is empty.")
except Exception as e:
    print(f"An error occurred: {e}")

# DATA OPERATIONS
def safe_division(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
    except TypeError:
        print("Invalid input types")
        return None

# VALIDATION
def validate_dataframe(df, required_columns):
    """Validate if DataFrame has required columns"""
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    return True
```

---

## HƯỚNG DẪN CHO GIẢNG VIÊN

### Chuẩn bị trước buổi học:
1. **Môi trường**: Đảm bảo tất cả máy đã cài đặt Python, Jupyter Notebook
2. **Dữ liệu**: Chuẩn bị file dữ liệu mẫu (CSV, Excel) cho thực hành
3. **Code**: Kiểm tra tất cả code examples hoạt động chính xác

### Gợi ý dẫn dắt:
1. **Mở đầu**: So sánh với C# để học viên dễ tiếp cận
2. **Demo**: Thực hiện live coding để học viên theo dõi
3. **Tương tác**: Đặt câu hỏi trong quá trình giảng
4. **Thực hành**: Đi kiểm tra và hỗ trợ từng nhóm

### Xử lý tình huống:
- **Học viên chậm**: Cung cấp code template
- **Lỗi technical**: Có backup code và dữ liệu
- **Câu hỏi khó**: Ghi chú để trả lời sau hoặc tìm hiểu thêm

### Mở rộng nâng cao:
- **Seaborn** cho visualization đẹp hơn
- **Plotly** cho interactive charts  
- **Scikit-learn** cho machine learning cơ bản
- **Streamlit** cho tạo web app đơn giản

---

**LƯU Ý**: Bài học này được thiết kế linh hoạt. Giảng viên có thể điều chỉnh thời gian và độ khó tùy theo trình độ lớp học.
