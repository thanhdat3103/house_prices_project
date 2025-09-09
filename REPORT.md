# CS4680 Prompt Engineering — Assignment 1
## Machine Learning Exercise: House Prices

### 1) Xác định vấn đề
- **Bài toán chính (hồi quy):** Dự đoán `SalePrice` (USD) cho một căn nhà.
- **Bài toán phụ (phân loại):** Dự đoán nhà có **giá cao hơn trung vị** (`SalePrice > median`) hay không.
- **Biến mục tiêu (Target):** `SalePrice` (hồi quy), nhãn nhị phân (phân loại).
- **Đặc trưng (Features) được chọn (ban đầu):**
  - `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `YearBuilt`.

> Ghi chú: Bạn có thể mở rộng/tinh chỉnh đặc trưng để cải thiện mô hình (ví dụ: biến đổi log, mã hóa danh mục, xử lý thiếu).

### 2) Thu thập dữ liệu
- Nguồn: Kaggle — *House Prices: Advanced Regression Techniques* (`train.csv`).
- Kích thước: 1460 hàng, 81 cột.
- Chất lượng dữ liệu: Có thiếu giá trị ở một số cột (ví dụ: Basement, Garage), cần xử lý cơ bản.

### 3) Phát triển mô hình
- Hồi quy: `LinearRegression`, `RandomForestRegressor`.
- Phân loại: `LogisticRegression`, `RandomForestClassifier` (nhãn nhị phân dựa trên median của `SalePrice`).

### 4) Đánh giá mô hình
- **Hồi quy:** MAE, RMSE, R².
- **Phân loại:** Accuracy, Confusion Matrix, Classification Report.

> Kết quả chạy mẫu nằm trong `outputs/metrics.json` và biểu đồ trong `outputs/*.png`.
> Hãy bổ sung **nhận xét của bạn** về độ phù hợp của từng mô hình, ưu/nhược điểm, và đề xuất cải tiến (feature engineering, tuning, cross-validation).

### 5) Tài liệu & Mã nguồn
- Mã: `src/house_prices_ml.py` (được chú thích).
- Cách chạy: xem `README.md`.
- URL GitHub: (bạn điền sau khi đẩy repo).
