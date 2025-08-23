import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from faker import Faker
import os

# Khởi tạo Faker
fake = Faker('vi_VN')  # Vietnamese locale
Faker.seed(42)
np.random.seed(42)
random.seed(42)

def create_directories():
    """Tạo các thư mục cần thiết"""
    directories = ['data/raw', 'data/processed', 'data/cleaned']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Created directories:", directories)

# ============================================================================
# BÀI TẬP 1: BASIC DATA CLEANING - CUSTOMERS DATA
# ============================================================================

def generate_customers_data():
    """Tạo dữ liệu khách hàng với các vấn đề thường gặp"""
    
    n_customers = 1000
    data = []
    
    # Tạo dữ liệu khách hàng
    for i in range(n_customers):
        # Một số customer_id trùng lặp (5%)
        if random.random() < 0.05 and i > 0:
            customer_id = f"CUST_{random.randint(1, i)}"
        else:
            customer_id = f"CUST_{i+1}"
        
        # Tên với các vấn đề formatting
        name = fake.name()
        if random.random() < 0.1:  # 10% tên có vấn đề
            name = random.choice([
                name.upper(),  # All caps
                name.lower(),  # All lowercase
                f"  {name}  ",  # Extra spaces
                name.replace(" ", ""),  # No spaces
                f"{name}123",  # With numbers
                ""  # Empty name
            ])
        
        # Email với các vấn đề
        email = fake.email().lower()
        if random.random() < 0.15:  # 15% email có vấn đề
            email = random.choice([
                email.replace("@", "AT"),  # Invalid format
                email.upper(),  # Uppercase
                f"  {email}  ",  # Extra spaces
                "invalid-email",  # Completely invalid
                "",  # Empty
                None  # Null
            ])
        
        # Phone với nhiều format khác nhau
        phone_formats = [
            "0{:9d}".format(random.randint(100000000, 999999999)),  # 0xxxxxxxxx
            "+84{:9d}".format(random.randint(100000000, 999999999)),  # +84xxxxxxxxx
            "{:3d}-{:3d}-{:4d}".format(random.randint(100, 999), random.randint(100, 999), random.randint(1000, 9999)),  # xxx-xxx-xxxx
            "({:3d}) {:3d}-{:4d}".format(random.randint(100, 999), random.randint(100, 999), random.randint(1000, 9999)),  # (xxx) xxx-xxxx
        ]
        phone = random.choice(phone_formats)
        
        if random.random() < 0.08:  # 8% phone có vấn đề
            phone = random.choice([
                "123",  # Too short
                "abcd-efgh-ijkl",  # Letters
                "",  # Empty
                None  # Null
            ])
        
        # Age với outliers và missing values
        age = random.randint(18, 80)
        if random.random() < 0.05:  # 5% age có vấn đề
            age = random.choice([
                random.randint(150, 200),  # Impossible age
                random.randint(-10, 0),    # Negative age
                None  # Missing
            ])
        
        # City với inconsistent naming
        cities = ["Hà Nội", "TP.HCM", "Đà Nẵng", "Hải Phòng", "Cần Thơ"]
        city = random.choice(cities)
        if random.random() < 0.1:  # 10% city có vấn đề formatting
            if city == "Hà Nội":
                city = random.choice(["Ha Noi", "HANOI", "ha noi", "Hanoi"])
            elif city == "TP.HCM":
                city = random.choice(["Ho Chi Minh", "HCMC", "Sai Gon", "tp hcm"])
        
        # Registration date với missing values
        reg_date = fake.date_between(start_date='-3y', end_date='today')
        if random.random() < 0.05:  # 5% missing dates
            reg_date = None
        else:
            # Một số format khác nhau
            if random.random() < 0.3:
                reg_date = reg_date.strftime('%d/%m/%Y')  # DD/MM/YYYY
            elif random.random() < 0.3:
                reg_date = reg_date.strftime('%Y-%m-%d')  # YYYY-MM-DD
            else:
                reg_date = reg_date.strftime('%m-%d-%Y')  # MM-DD-YYYY
        
        data.append({
            'customer_id': customer_id,
            'name': name,
            'email': email,
            'phone': phone,
            'age': age,
            'city': city,
            'registration_date': reg_date
        })
    
    # Tạo DataFrame và lưu file
    df = pd.DataFrame(data)
    
    # Thêm một số hàng hoàn toàn trống
    empty_rows = pd.DataFrame([{col: None for col in df.columns} for _ in range(20)])
    df = pd.concat([df, empty_rows], ignore_index=True)
    
    df.to_csv('data/raw/customers.csv', index=False)
    print(f"✓ Generated customers.csv with {len(df)} rows")
    return df

# ============================================================================
# BÀI TẬP 2: SALES & PRODUCTS DATA
# ============================================================================

def generate_sales_data():
    """Tạo dữ liệu sales với các vấn đề"""
    
    n_orders = 2000
    customer_ids = [f"CUST_{i+1}" for i in range(500)]  # 500 customers
    product_ids = [f"PROD_{i+1:04d}" for i in range(100)]  # 100 products
    
    data = []
    
    for i in range(n_orders):
        order_id = f"ORD_{i+1:06d}"
        
        # Một số order_id trùng lặp
        if random.random() < 0.03:  # 3% duplicates
            order_id = f"ORD_{random.randint(1, max(1, i-100)):06d}"
        
        customer_id = random.choice(customer_ids)
        product_id = random.choice(product_ids)
        
        # Quantity với outliers
        quantity = random.randint(1, 10)
        if random.random() < 0.02:  # 2% outliers
            quantity = random.choice([0, -1, 100, 1000])  # Invalid quantities
        
        # Order date với missing và invalid values
        order_date = fake.date_between(start_date='-1y', end_date='today')
        if random.random() < 0.03:  # 3% missing dates
            order_date = None
        elif random.random() < 0.05:  # 5% invalid dates
            order_date = random.choice([
                "2024-13-45",  # Invalid date
                "invalid-date",
                "2023-02-30"  # February 30th
            ])
        else:
            # Random date formats
            formats = ['%Y-%m-%d', '%d/%m/%Y', '%m-%d-%Y']
            order_date = order_date.strftime(random.choice(formats))
        
        data.append({
            'order_id': order_id,
            'customer_id': customer_id,
            'product_id': product_id,
            'quantity': quantity,
            'order_date': order_date
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/raw/sales.csv', index=False)
    print(f"✓ Generated sales.csv with {len(df)} rows")
    return df

def generate_products_data():
    """Tạo dữ liệu products trong format JSON"""
    
    categories = ["Electronics", "Clothing", "Books", "Home & Garden", "Sports"]
    products = []
    
    for i in range(100):
        product_id = f"PROD_{i+1:04d}"
        
        # Product name với các vấn đề
        product_names = [
            f"Product {i+1}",
            f"Awesome {fake.word().title()} Device",
            f"{fake.company()} {fake.word().title()}",
            f"Premium {fake.color_name()} {fake.word().title()}"
        ]
        product_name = random.choice(product_names)
        
        if random.random() < 0.05:  # 5% tên có vấn đề
            product_name = random.choice([
                "",  # Empty
                None,  # Null
                "Product123!@#$%",  # Special characters
                product_name.upper(),  # All caps
            ])
        
        category = random.choice(categories)
        if random.random() < 0.08:  # 8% category có vấn đề
            category = random.choice([
                category.lower(),  # Lowercase
                f"{category}s",    # Plural form
                "Other",           # Different category
                None              # Missing
            ])
        
        # Price với various formats và issues
        price = round(random.uniform(10, 1000), 2)
        if random.random() < 0.07:  # 7% price có vấn đề
            price = random.choice([
                0,      # Zero price
                -10.5,  # Negative price
                None,   # Missing price
                "invalid"  # Non-numeric
            ])
        
        products.append({
            'product_id': product_id,
            'product_name': product_name,
            'category': category,
            'price': price
        })
    
    # Lưu file JSON
    with open('data/raw/products.json', 'w', encoding='utf-8') as f:
        json.dump(products, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Generated products.json with {len(products)} products")
    return products

# ============================================================================
# BÀI TẬP 3: MULTI-SOURCE EMPLOYEE DATA
# ============================================================================

def generate_employees_hr_data():
    """Tạo dữ liệu HR của nhân viên"""
    
    departments = ["IT", "Finance", "Marketing", "Sales", "HR", "Operations"]
    n_employees = 300
    data = []
    
    for i in range(n_employees):
        employee_id = f"EMP{i+1:04d}"
        
        # Name với formatting issues
        name = fake.name()
        if random.random() < 0.1:  # 10% có vấn đề
            name = random.choice([
                name.upper(),
                name.lower(),
                f"  {name}  ",
                f"{name} Jr.",
                f"Dr. {name}",
                ""
            ])
        
        department = random.choice(departments)
        if random.random() < 0.05:  # 5% department inconsistent
            if department == "IT":
                department = random.choice(["Information Technology", "Tech", "IT Dept"])
        
        # Hire date với missing values
        hire_date = fake.date_between(start_date='-10y', end_date='-1y')
        if random.random() < 0.03:  # 3% missing
            hire_date = None
        else:
            hire_date = hire_date.strftime('%Y-%m-%d')
        
        data.append({
            'employee_id': employee_id,
            'name': name,
            'department': department,
            'hire_date': hire_date
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/raw/employees_hr.csv', index=False)
    print(f"✓ Generated employees_hr.csv with {len(df)} rows")
    return df

def generate_salaries_data():
    """Tạo dữ liệu salary trong JSON format"""
    
    employee_ids = [f"EMP{i+1:04d}" for i in range(300)]
    salaries = []
    
    for emp_id in employee_ids:
        # Skip một số employees để tạo missing data
        if random.random() < 0.05:  # 5% missing salary data
            continue
        
        base_salary = random.randint(30000, 150000)
        bonus = random.randint(0, 20000)
        
        # Một số salary có vấn đề
        if random.random() < 0.03:  # 3% problematic salaries
            base_salary = random.choice([
                0,           # Zero salary
                -5000,       # Negative
                1000000,     # Unrealistic high
                None         # Missing
            ])
        
        salaries.append({
            'employee_id': emp_id,
            'base_salary': base_salary,
            'bonus': bonus
        })
    
    # Add some duplicate employee_ids
    for _ in range(10):
        duplicate_entry = random.choice(salaries).copy()
        duplicate_entry['base_salary'] *= 1.1  # Slightly different salary
        salaries.append(duplicate_entry)
    
    with open('data/raw/salaries.json', 'w') as f:
        json.dump(salaries, f, indent=2)
    
    print(f"✓ Generated salaries.json with {len(salaries)} entries")
    return salaries

def generate_performance_data():
    """Tạo dữ liệu performance"""
    
    employee_ids = [f"EMP{i+1:04d}" for i in range(300)]
    data = []
    
    for emp_id in employee_ids:
        # Skip một số để tạo missing data
        if random.random() < 0.08:  # 8% missing
            continue
        
        # Performance score (1-10)
        performance_score = round(random.uniform(1, 10), 1)
        if random.random() < 0.02:  # 2% invalid scores
            performance_score = random.choice([
                0,      # Zero score
                11,     # Above max
                -1,     # Negative
                None    # Missing
            ])
        
        # Last review date
        last_review = fake.date_between(start_date='-2y', end_date='today')
        if random.random() < 0.05:  # 5% missing dates
            last_review = None
        else:
            last_review = last_review.strftime('%Y-%m-%d')
        
        data.append({
            'employee_id': emp_id,
            'performance_score': performance_score,
            'last_review_date': last_review
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/raw/performance.csv', index=False)
    print(f"✓ Generated performance.csv with {len(df)} rows")
    return df

# ============================================================================
# BÀI TẬP NÂNG CAO: E-COMMERCE DATA
# ============================================================================

def generate_ecommerce_customers():
    """Tạo dữ liệu khách hàng e-commerce phức tạp"""
    
    n_customers = 1500
    countries = ["Vietnam", "Thailand", "Singapore", "Malaysia", "Indonesia"]
    data = []
    
    for i in range(n_customers):
        customer_id = f"ECUST_{i+1:06d}"
        
        # Tên với các vấn đề đa dạng
        name = fake.name()
        if random.random() < 0.12:  # 12% có vấn đề
            issues = [
                name.upper(),
                name.lower(),
                f"  {name}   ",  # Multiple spaces
                name.replace(" ", "_"),
                f"{name}_{random.randint(1, 999)}",
                "Test User",
                "Demo Account",
                f"Admin {name}",
                "",
                None
            ]
            name = random.choice(issues)
        
        # Email với nhiều vấn đề
        email = fake.email()
        if random.random() < 0.18:  # 18% có vấn đề
            issues = [
                email.replace("@", "[AT]"),
                email.replace(".", "[DOT]"),
                email.upper(),
                f" {email} ",
                f"{email}.backup",
                "invalid@",
                "@invalid.com",
                "test@test",
                "admin@company.com",
                "",
                None
            ]
            email = random.choice(issues)
        
        # Phone với format đa dạng
        phone_patterns = [
            f"+84{random.randint(100000000, 999999999)}",
            f"0{random.randint(100000000, 999999999)}",
            f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
            f"+84 {random.randint(10, 99)} {random.randint(100, 999)} {random.randint(1000, 9999)}",
        ]
        phone = random.choice(phone_patterns)
        
        if random.random() < 0.1:  # 10% có vấn đề
            phone = random.choice([
                "123456",
                "abcdefghij",
                "+84-invalid-phone",
                "",
                None,
                "0000000000"
            ])
        
        # Registration date với format đa dạng
        reg_date = fake.date_between(start_date='-5y', end_date='today')
        if random.random() < 0.04:  # 4% missing
            reg_date = None
        else:
            formats = [
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m-%d-%Y',
                '%Y/%m/%d',
                '%d-%m-%Y'
            ]
            reg_date = reg_date.strftime(random.choice(formats))
        
        country = random.choice(countries)
        
        data.append({
            'customer_id': customer_id,
            'name': name,
            'email': email,
            'phone': phone,
            'registration_date': reg_date,
            'country': country
        })
    
    # Thêm duplicates
    for _ in range(50):
        duplicate = random.choice(data).copy()
        duplicate['customer_id'] = f"ECUST_{len(data)+1:06d}"
        data.append(duplicate)
    
    df = pd.DataFrame(data)
    df.to_csv('data/raw/ecommerce_customers.csv', index=False)
    print(f"✓ Generated ecommerce_customers.csv with {len(df)} rows")
    return df

def generate_ecommerce_orders_json():
    """Tạo dữ liệu orders trong nested JSON format"""
    
    customer_ids = [f"ECUST_{i+1:06d}" for i in range(1550)]
    product_ids = [f"EPROD_{i+1:04d}" for i in range(200)]
    
    orders = []
    
    for i in range(3000):  # 3000 orders
        order_id = f"EORD_{i+1:08d}"
        customer_id = random.choice(customer_ids)
        
        # Order date
        order_date = fake.date_between(start_date='-2y', end_date='today')
        order_datetime = datetime.combine(order_date, fake.time())
        
        # Order status
        statuses = ["completed", "pending", "cancelled", "shipped", "processing"]
        status = random.choice(statuses)
        
        # Items trong order (nested structure)
        n_items = random.randint(1, 5)
        items = []
        
        for _ in range(n_items):
            item = {
                "product_id": random.choice(product_ids),
                "quantity": random.randint(1, 10),
                "unit_price": round(random.uniform(5, 500), 2)
            }
            
            # Một số item có vấn đề
            if random.random() < 0.05:  # 5%
                problematic_values = [
                    {"product_id": "", "quantity": 0, "unit_price": -10},
                    {"product_id": None, "quantity": -1, "unit_price": 0},
                    {"product_id": "INVALID", "quantity": 1000, "unit_price": None}
                ]
                item.update(random.choice(problematic_values))
            
            items.append(item)
        
        # Payment method
        payment_methods = ["credit_card", "paypal", "bank_transfer", "cash_on_delivery"]
        payment_method = random.choice(payment_methods)
        
        # Shipping address (nested)
        shipping_address = {
            "street": fake.street_address(),
            "city": fake.city(),
            "postal_code": fake.postcode(),
            "country": random.choice(["Vietnam", "Thailand", "Singapore"])
        }
        
        # Một số shipping address có vấn đề
        if random.random() < 0.08:  # 8%
            shipping_address = random.choice([
                {"street": "", "city": "", "postal_code": "", "country": ""},
                {"street": None, "city": None, "postal_code": None, "country": None},
                {}  # Empty address
            ])
        
        order = {
            "order_id": order_id,
            "customer_id": customer_id,
            "order_date": order_datetime.isoformat(),
            "status": status,
            "payment_method": payment_method,
            "shipping_address": shipping_address,
            "items": items
        }
        
        orders.append(order)
    
    # Một số orders missing key fields
    for _ in range(100):
        incomplete_order = {
            "order_id": f"EORD_{len(orders)+1:08d}",
            "customer_id": random.choice(customer_ids)
            # Missing other required fields
        }
        orders.append(incomplete_order)
    
    with open('data/raw/ecommerce_orders.json', 'w', encoding='utf-8') as f:
        json.dump(orders, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Generated ecommerce_orders.json with {len(orders)} orders")
    return orders

def generate_products_xml():
    """Tạo product catalog trong XML format"""
    
    # Tạo root element
    root = ET.Element("products")
    
    categories = ["Electronics", "Fashion", "Books", "Sports", "Home"]
    brands = ["Samsung", "Apple", "Nike", "Adidas", "Sony", "LG", "Canon", "Dell"]
    
    for i in range(200):
        product = ET.SubElement(root, "product")
        
        # Product ID
        product_id = ET.SubElement(product, "id")
        product_id.text = f"EPROD_{i+1:04d}"
        
        # Product name với vấn đề
        name = ET.SubElement(product, "name")
        product_name = f"{random.choice(brands)} {fake.word().title()} {random.choice(['Pro', 'Max', 'Plus', 'Lite', ''])}"
        
        if random.random() < 0.07:  # 7% có vấn đề
            product_name = random.choice([
                "",
                f"Product{i+1}",
                product_name.upper(),
                f"  {product_name}  ",
                None
            ])
        
        name.text = str(product_name) if product_name else ""
        
        # Category
        category = ET.SubElement(product, "category")
        cat_name = random.choice(categories)
        if random.random() < 0.05:  # 5% inconsistent
            cat_name = cat_name.lower() if random.random() < 0.5 else f"{cat_name}s"
        category.text = cat_name
        
        # Price với format issues
        price = ET.SubElement(product, "price")
        price_val = round(random.uniform(10, 2000), 2)
        
        if random.random() < 0.08:  # 8% có vấn đề
            price_formats = [
                f"${price_val}",
                f"{price_val} USD",
                f"{price_val:,.2f}",
                "0",
                "-10.5",
                ""
            ]
            price.text = random.choice(price_formats)
        else:
            price.text = str(price_val)
        
        # Description
        description = ET.SubElement(product, "description")
        desc_text = fake.text(max_nb_chars=200)
        if random.random() < 0.1:  # 10% missing description
            desc_text = ""
        description.text = desc_text
        
        # Stock quantity
        stock = ET.SubElement(product, "stock_quantity")
        stock_val = random.randint(0, 1000)
        if random.random() < 0.03:  # 3% có vấn đề
            stock_val = random.choice([-5, 10000, 0])
        stock.text = str(stock_val)
        
        # Weight (với missing values)
        if random.random() > 0.15:  # 85% có weight
            weight = ET.SubElement(product, "weight")
            weight.text = str(round(random.uniform(0.1, 50), 2))
    
    # Lưu XML file
    tree = ET.ElementTree(root)
    tree.write('data/raw/products_catalog.xml', encoding='utf-8', xml_declaration=True)
    print(f"✓ Generated products_catalog.xml with 200 products")
    
    return root

def generate_reviews_data():
    """Tạo dữ liệu reviews với text data"""
    
    product_ids = [f"EPROD_{i+1:04d}" for i in range(200)]
    customer_ids = [f"ECUST_{i+1:06d}" for i in range(1550)]
    
    data = []
    
    # Tạo review templates
    positive_reviews = [
        "Great product! Highly recommended.",
        "Excellent quality and fast delivery.",
        "Perfect! Exactly what I was looking for.",
        "Amazing product, will buy again.",
        "Outstanding quality and service."
    ]
    
    negative_reviews = [
        "Poor quality, not as described.",
        "Terrible product, waste of money.",
        "Not satisfied with the purchase.",
        "Quality is very disappointing.",
        "Would not recommend this product."
    ]
    
    neutral_reviews = [
        "It's okay, nothing special.",
        "Average product, meets basic needs.",
        "Decent quality for the price.",
        "Not bad, but could be better.",
        "Satisfactory purchase overall."
    ]
    
    for i in range(5000):  # 5000 reviews
        product_id = random.choice(product_ids)
        customer_id = random.choice(customer_ids)
        
        # Rating (1-5)
        rating = random.randint(1, 5)
        if random.random() < 0.03:  # 3% invalid ratings
            rating = random.choice([0, 6, -1, None])
        
        # Review text based on rating
        if rating and rating >= 4:
            review_text = random.choice(positive_reviews)
        elif rating and rating <= 2:
            review_text = random.choice(negative_reviews)
        else:
            review_text = random.choice(neutral_reviews)
        
        # Add some variety và issues
        if random.random() < 0.1:  # 10% có text issues
            modifications = [
                review_text.upper(),
                review_text.lower(),
                f"  {review_text}  ",
                f"{review_text} " * 3,  # Repeated text
                review_text.replace(" ", ""),
                "",  # Empty review
                "a" * 1000,  # Very long review
                "Review123!@#$%^&*()",  # Special characters
                None  # Null review
            ]
            review_text = random.choice(modifications)
        
        # Review date
        review_date = fake.date_between(start_date='-1y', end_date='today')
        if random.random() < 0.02:  # 2% missing dates
            review_date = None
        else:
            review_date = review_date.strftime('%Y-%m-%d')
        
        # Verified purchase flag
        verified_purchase = random.choice([True, False, None])
        
        data.append({
            'review_id': f"REV_{i+1:07d}",
            'product_id': product_id,
            'customer_id': customer_id,
            'rating': rating,
            'review_text': review_text,
            'review_date': review_date,
            'verified_purchase': verified_purchase
        })
    
    # Thêm duplicate reviews
    for _ in range(200):
        duplicate = random.choice(data).copy()
        duplicate['review_id'] = f"REV_{len(data)+1:07d}"
        data.append(duplicate)
    
    df = pd.DataFrame(data)
    df.to_csv('data/raw/product_reviews.csv', index=False)
    print(f"✓ Generated product_reviews.csv with {len(df)} reviews")
    return df

# ============================================================================
# TIME SERIES DATA - SENSOR DATA
# ============================================================================

def generate_sensor_data():
    """Tạo time series sensor data với anomalies"""
    
    # Tạo timestamp range (1 năm, mỗi giờ)
    start_date = datetime.now() - timedelta(days=365)
    timestamps = pd.date_range(start=start_date, periods=8760, freq='H')
    
    data = []
    
    for i, timestamp in enumerate(timestamps):
        # Base patterns - seasonal và daily cycles
        hour = timestamp.hour
        day_of_year = timestamp.timetuple().tm_yday
        
        # Temperature pattern (seasonal + daily cycle)
        seasonal_temp = 25 + 10 * np.sin(2 * np.pi * day_of_year / 365)  # Annual cycle
        daily_temp = 5 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
        base_temp = seasonal_temp + daily_temp + random.gauss(0, 2)  # Add noise
        
        # Humidity pattern (inverse relationship with temperature)
        base_humidity = 70 - (base_temp - 25) * 2 + random.gauss(0, 5)
        base_humidity = max(20, min(95, base_humidity))  # Clamp to realistic range
        
        # Pressure pattern (more stable)
        base_pressure = 1013 + random.gauss(0, 10)
        
        # Add anomalies và missing data
        temperature = base_temp
        humidity = base_humidity  
        pressure = base_pressure
        
        # Random sensor failures (missing data)
        if random.random() < 0.02:  # 2% missing temperature
            temperature = None
        if random.random() < 0.015:  # 1.5% missing humidity
            humidity = None
        if random.random() < 0.01:  # 1% missing pressure
            pressure = None
        
        # Sensor anomalies/outliers
        if random.random() < 0.005:  # 0.5% temperature outliers
            temperature = random.choice([
                base_temp + random.uniform(20, 40),  # Hot spike
                base_temp - random.uniform(15, 25),  # Cold spike
                -50,  # Impossible cold
                80,   # Impossible hot
            ])
        
        if random.random() < 0.003:  # 0.3% humidity outliers
            humidity = random.choice([
                150,  # Impossible high
                -10,  # Impossible low  
                0,    # Completely dry
            ])
        
        if random.random() < 0.002:  # 0.2% pressure outliers
            pressure = random.choice([
                pressure + random.uniform(100, 200),  # High pressure spike
                pressure - random.uniform(100, 200),  # Low pressure spike
                0,     # Sensor failure
                2000,  # Impossible high
            ])
        
        # Sensor drift (gradual degradation)
        if i > 4000:  # After ~5 months
            if random.random() < 0.0001:  # Very rare drift events
                temperature += random.uniform(-2, 2)
                humidity += random.uniform(-3, 3)
        
        data.append({
            'timestamp': timestamp,
            'device_id': 'SENSOR_001',
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'location': 'Lab_Room_A'
        })
    
    # Add second sensor with different patterns
    for i, timestamp in enumerate(timestamps):
        if random.random() < 0.1:  # 10% của data points để tạo sparse data
            continue
            
        # Slightly different base patterns
        hour = timestamp.hour
        day_of_year = timestamp.timetuple().tm_yday
        
        seasonal_temp = 23 + 8 * np.sin(2 * np.pi * day_of_year / 365 + np.pi/4)
        daily_temp = 3 * np.sin(2 * np.pi * hour / 24 + np.pi/6)
        base_temp = seasonal_temp + daily_temp + random.gauss(0, 1.5)
        
        base_humidity = 65 - (base_temp - 23) * 1.8 + random.gauss(0, 4)
        base_humidity = max(25, min(90, base_humidity))
        
        base_pressure = 1015 + random.gauss(0, 8)
        
        data.append({
            'timestamp': timestamp,
            'device_id': 'SENSOR_002', 
            'temperature': base_temp,
            'humidity': base_humidity,
            'pressure': base_pressure,
            'location': 'Lab_Room_B'
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.to_csv('data/raw/sensor_data.csv', index=False)
    print(f"✓ Generated sensor_data.csv with {len(df)} readings")
    return df

# ============================================================================
# FINANCIAL DATA - TRANSACTIONS & ACCOUNTS  
# ============================================================================

def generate_financial_transactions():
    """Tạo financial transaction data với fraud patterns"""
    
    account_ids = [f"ACC_{i+1:06d}" for i in range(1000)]
    transaction_types = ["deposit", "withdrawal", "transfer", "payment", "refund"]
    merchants = ["Amazon", "Walmart", "Starbucks", "Shell", "McDonald's", "ATM", "Bank Transfer"]
    
    data = []
    
    # Normal transactions
    for i in range(10000):
        transaction_id = f"TXN_{i+1:010d}"
        account_id = random.choice(account_ids)
        
        # Transaction type affects amount distribution
        txn_type = random.choice(transaction_types)
        
        if txn_type == "deposit":
            amount = round(random.uniform(100, 5000), 2)
        elif txn_type == "withdrawal":
            amount = -round(random.uniform(20, 1000), 2)
        elif txn_type == "transfer":
            amount = random.choice([1, -1]) * round(random.uniform(50, 2000), 2)
        elif txn_type == "payment":
            amount = -round(random.uniform(5, 500), 2)
        else:  # refund
            amount = round(random.uniform(10, 200), 2)
        
        # Add some amount formatting issues
        if random.random() < 0.05:  # 5% formatting issues
            amount_issues = [
                f"${abs(amount)}",  # With currency symbol
                f"{amount:,}",      # With commas
                f"({abs(amount)})" if amount < 0 else f"{amount}",  # Parentheses for negative
                str(amount).replace(".", ","),  # European decimal format
                0,  # Zero amount
                None  # Missing amount
            ]
            amount = random.choice(amount_issues)
        
        # Timestamp
        transaction_date = fake.date_time_between(start_date='-2y', end_date='now')
        
        # Add some timestamp issues  
        if random.random() < 0.02:  # 2% timestamp issues
            timestamp_issues = [
                transaction_date.strftime('%Y-%m-%d %H:%M:%S'),
                transaction_date.strftime('%d/%m/%Y %H:%M'),
                transaction_date.strftime('%m-%d-%Y'),
                "2024-02-30 25:70:80",  # Invalid datetime
                "",  # Empty
                None  # Missing
            ]
            transaction_date = random.choice(timestamp_issues)
        
        merchant = random.choice(merchants) if txn_type == "payment" else ""
        
        # Add description
        descriptions = [
            f"{txn_type.title()} - {merchant}",
            f"Online {txn_type}",
            f"{merchant} Purchase",
            "ATM Transaction",
            "Wire Transfer"
        ]
        description = random.choice(descriptions)
        
        if random.random() < 0.03:  # 3% description issues
            description = random.choice([
                "",  # Empty
                None,  # Missing
                "Transaction123!@#",  # Special chars
                description.upper(),  # All caps
                "a" * 200  # Too long
            ])
        
        data.append({
            'transaction_id': transaction_id,
            'account_id': account_id,
            'transaction_type': txn_type,
            'amount': amount,
            'transaction_date': transaction_date,
            'merchant': merchant,
            'description': description
        })
    
    # Add fraudulent patterns
    fraud_accounts = random.sample(account_ids, 20)  # 20 accounts with fraud
    
    for fraud_account in fraud_accounts:
        # Rapid succession transactions (fraud pattern)
        base_time = fake.date_time_between(start_date='-6M', end_date='now')
        
        for j in range(random.randint(5, 15)):
            fraud_time = base_time + timedelta(seconds=random.randint(1, 300))
            
            data.append({
                'transaction_id': f"TXN_{len(data)+1:010d}",
                'account_id': fraud_account,
                'transaction_type': 'withdrawal',
                'amount': -round(random.uniform(100, 500), 2),
                'transaction_date': fraud_time,
                'merchant': 'ATM',
                'description': 'ATM Withdrawal'
            })
        
        # Round number transactions (fraud indicator)
        for j in range(random.randint(3, 8)):
            data.append({
                'transaction_id': f"TXN_{len(data)+1:010d}",
                'account_id': fraud_account,
                'transaction_type': 'withdrawal', 
                'amount': -random.choice([100, 200, 500, 1000]),  # Exact round numbers
                'transaction_date': fake.date_time_between(start_date='-3M', end_date='now'),
                'merchant': '',
                'description': 'Cash Withdrawal'
            })
    
    # Add some duplicate transactions
    for _ in range(100):
        duplicate = random.choice(data).copy()
        duplicate['transaction_id'] = f"TXN_{len(data)+1:010d}"
        data.append(duplicate)
    
    df = pd.DataFrame(data)
    df.to_csv('data/raw/financial_transactions.csv', index=False)
    print(f"✓ Generated financial_transactions.csv with {len(df)} transactions")
    return df

def generate_account_data():
    """Tạo account balance data"""
    
    account_ids = [f"ACC_{i+1:06d}" for i in range(1000)]
    account_types = ["checking", "savings", "credit", "investment"]
    
    data = []
    
    for account_id in account_ids:
        account_type = random.choice(account_types)
        
        # Balance based on account type
        if account_type == "checking":
            balance = round(random.uniform(100, 10000), 2)
        elif account_type == "savings":
            balance = round(random.uniform(1000, 50000), 2)
        elif account_type == "credit":
            balance = -round(random.uniform(0, 5000), 2)  # Credit balance is negative
        else:  # investment
            balance = round(random.uniform(5000, 100000), 2)
        
        # Add balance issues
        if random.random() < 0.04:  # 4% have balance issues
            balance_issues = [
                f"${balance:,.2f}",  # With formatting
                str(balance).replace(".", ","),  # European format
                None,  # Missing
                "",   # Empty
                -999999,  # Impossible negative
                0     # Zero balance
            ]
            balance = random.choice(balance_issues)
        
        balance_date = fake.date_between(start_date='-1M', end_date='today')
        
        # Customer info
        customer_name = fake.name()
        if random.random() < 0.08:  # 8% name issues
            customer_name = random.choice([
                customer_name.upper(),
                customer_name.lower(), 
                f"  {customer_name}  ",
                "",
                None
            ])
        
        data.append({
            'account_id': account_id,
            'account_type': account_type,
            'customer_name': customer_name,
            'reported_balance': balance,
            'balance_date': balance_date,
            'status': random.choice(['active', 'inactive', 'closed', 'frozen'])
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/raw/account_balances.csv', index=False)
    print(f"✓ Generated account_balances.csv with {len(df)} accounts")
    return df

# ============================================================================
# MAIN FUNCTION - Generate All Sample Data
# ============================================================================

def generate_all_sample_data():
    """Generate all sample datasets for the exercises"""
    
    print("🚀 Starting sample data generation...")
    print("=" * 60)
    
    # Create directories
    create_directories()
    print()
    
    # Basic exercises data
    print("📊 Generating Basic Exercise Data:")
    print("-" * 40)
    generate_customers_data()
    generate_sales_data()
    generate_products_data()
    print()
    
    # Multi-source employee data
    print("👥 Generating Employee Data (Multi-source):")
    print("-" * 40) 
    generate_employees_hr_data()
    generate_salaries_data()
    generate_performance_data()
    print()
    
    # Advanced e-commerce data
    print("🛒 Generating E-commerce Data (Advanced):")
    print("-" * 40)
    generate_ecommerce_customers()
    generate_ecommerce_orders_json()
    generate_products_xml()
    generate_reviews_data()
    print()
    
    # Time series sensor data
    print("📡 Generating Time Series Sensor Data:")
    print("-" * 40)
    generate_sensor_data()
    print()
    
    # Financial data
    print("💰 Generating Financial Data:")
    print("-" * 40)
    generate_financial_transactions()
    generate_account_data()
    print()
    
    print("=" * 60)
    print("✅ All sample data generated successfully!")
    print("\nGenerated files structure:")
    print("""
data/
├── raw/
│   ├── customers.csv                 (Basic cleaning exercise)
│   ├── sales.csv                     (Sales analysis)
│   ├── products.json                 (JSON reading practice)
│   ├── employees_hr.csv             (Multi-source integration - HR)
│   ├── salaries.json                (Multi-source integration - Salaries)  
│   ├── performance.csv              (Multi-source integration - Performance)
│   ├── ecommerce_customers.csv      (Advanced e-commerce)
│   ├── ecommerce_orders.json        (Nested JSON structure)
│   ├── products_catalog.xml         (XML parsing practice)
│   ├── product_reviews.csv          (Text cleaning practice)
│   ├── sensor_data.csv              (Time series anomalies)
│   ├── financial_transactions.csv   (Fraud detection patterns)
│   └── account_balances.csv         (Financial reconciliation)
├── processed/                       (For intermediate files)
└── cleaned/                         (For final cleaned data)
    """)
    
    # Generate a data summary report
    generate_data_summary()

def generate_data_summary():
    """Generate a summary report of all datasets"""
    
    summary = {
        'Basic Exercise Data': {
            'customers.csv': {
                'purpose': 'Basic data cleaning practice',
                'rows': '~1,020',
                'issues': 'Missing values, duplicates, format inconsistencies, invalid emails/phones'
            },
            'sales.csv': {
                'purpose': 'Sales data analysis and joining practice', 
                'rows': '~2,000',
                'issues': 'Invalid dates, quantity outliers, duplicate orders'
            },
            'products.json': {
                'purpose': 'JSON reading and processing',
                'rows': '100',
                'issues': 'Missing values, price format inconsistencies, category variations'
            }
        },
        'Multi-source Employee Data': {
            'employees_hr.csv': {
                'purpose': 'HR master data',
                'rows': '300', 
                'issues': 'Name formatting, department inconsistencies'
            },
            'salaries.json': {
                'purpose': 'Salary information in JSON',
                'rows': '~295',
                'issues': 'Missing employees, duplicate entries, invalid salaries'
            },
            'performance.csv': {
                'purpose': 'Performance ratings',
                'rows': '~276',
                'issues': 'Missing employees, invalid scores, date issues'
            }
        },
        'Advanced E-commerce Data': {
            'ecommerce_customers.csv': {
                'purpose': 'Customer master with complex issues',
                'rows': '~1,550',
                'issues': 'Multiple email/phone formats, test accounts, country inconsistencies'
            },
            'ecommerce_orders.json': {
                'purpose': 'Nested order data with items',
                'rows': '~3,100', 
                'issues': 'Complex nested structure, missing fields, invalid items'
            },
            'products_catalog.xml': {
                'purpose': 'XML parsing practice',
                'rows': '200',
                'issues': 'Price formats, missing descriptions, stock inconsistencies'
            },
            'product_reviews.csv': {
                'purpose': 'Text data cleaning',
                'rows': '~5,200',
                'issues': 'Text formatting, invalid ratings, spam reviews, duplicates'
            }
        },
        'Time Series Data': {
            'sensor_data.csv': {
                'purpose': 'IoT sensor data with anomalies',
                'rows': '~15,700',
                'issues': 'Missing readings, sensor outliers, drift patterns, impossible values'
            }
        },
        'Financial Data': {
            'financial_transactions.csv': {
                'purpose': 'Transaction data with fraud patterns',
                'rows': '~10,500',
                'issues': 'Amount formatting, rapid transactions, round numbers, timestamp issues'
            },
            'account_balances.csv': {
                'purpose': 'Account balance reconciliation',
                'rows': '1,000',
                'issues': 'Balance formatting, missing data, account status inconsistencies'
            }
        }
    }
    
    # Save summary as JSON
    with open('data/DATA_SUMMARY.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("📋 Generated DATA_SUMMARY.json with detailed information about all datasets")

if __name__ == "__main__":
    # Install required packages if not available
    try:
        from faker import Faker
    except ImportError:
        print("Installing required package: faker")
        import subprocess
        subprocess.check_call(["pip", "install", "faker"])
        from faker import Faker
    
    generate_all_sample_data()
