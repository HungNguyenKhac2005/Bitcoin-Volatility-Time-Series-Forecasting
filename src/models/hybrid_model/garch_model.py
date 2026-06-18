"""
Data Scientist: Nguyễn Khắc Hưng
Ngày: 30-3-2026
Mô tả: Mô hình GARCH (Generalized Autoregressive Conditional Heteroskedasticity) này là một trong ba thành phần
cốt lõi cấu thành nên mô hình lai (Hybrid Model) trong dự án dự báo độ biến động (volatility) của Bitcoin.
... (Phần docstring tôi giữ nguyên của bạn) ...
"""

# ==================== IMPORT THƯ VIỆN ====================
import sys
import os

# Lấy đường dẫn tuyệt đối của file garch_model.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Lấy đường dẫn của thư mục cha (thư mục 'src')
parent_dir = os.path.dirname(current_dir)

# Thêm thư mục 'src' vào danh sách tìm kiếm module của Python
sys.path.append(parent_dir)

# Bây giờ bạn có thể import thư mục utils bình thường
from utils.gridSearch_hungdeptraikt import GARCHGridSearch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import math # 🌟 THÊM THƯ VIỆN MATH ĐỂ TÍNH CĂN BẬC 2
import warnings
warnings.filterwarnings("ignore") # Tắt cảnh báo để vòng lặp chạy mượt


# ==================== LOAD VÀ CHUẨN BỊ DỮ LIỆU ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'garch_data', 'bitcoin_garch_dataset.csv')
data = pd.read_csv(CSV_PATH)

# Chia tập dữ liệu ra làm train và test với tỷ lệ 80/20
train_size = int(len(data)*0.8)
train_set = data['log_return'][:train_size]
test_set = data['log_return'][train_size:]
train_time = data['timestamp'][:train_size]
test_time = data['timestamp'][train_size:]

# lấy 100 mẫu dữ liệu của tập test để đánh giá model
test_set100 = test_set[:100]

# chia biến dự đoán volatility theo tỷ lệ 80/20 với train_size tương tự như trên để phần tử lấy giống nhau 
volatility_test = data['volatility_14_annual'][train_size:]
volatility_test100 = volatility_test[:100]
mean_volatility_test = volatility_test100.mean()

# ==================== GRID SEARCH CHỌN MODEL TỐI ƯU ====================
# Đã comment lại để tiết kiệm thời gian chạy


# ==================== DỰ ĐOÁN ROLLING WINDOW TRÊN TẬP TRAIN ====================
# Dùng cửa sổ trượt 3000 mẫu để dự đoán volatility từng bước trên tập train

preds_train = []
preds_train_time = []
preds_train_dic = {}
window = 3000

for i in tqdm(range(window, len(train_set)), desc="Train Set Progress", colour='green'):
    train_data = train_set[i-window:i]
    
    # 🌟 ĐÃ SỬA: Đưa về GARCH(1,1) tiêu chuẩn chống bùng nổ phương sai
    model = arch_model(
        train_data,
        p=1,
        q=1,
        dist='skewt'
    )
    model_fit = model.fit(disp='off', show_warning=False)
    prediction = model_fit.forecast(horizon=1)
    variance_forecast = prediction.variance.iloc[-1,0]
    
    # 🌟 ĐÃ SỬA: Quy đổi hệ quy chiếu (Từ % theo giờ -> Số thập phân thường niên)
    hourly_vol_pct = np.sqrt(variance_forecast)
    hourly_vol_decimal = hourly_vol_pct / 100.0
    annualized_volatility = hourly_vol_decimal * math.sqrt(8760)
    
    preds_train.append(annualized_volatility)
    preds_train_time.append(train_time.iloc[i])

preds_train_dic['timestamp'] = preds_train_time
preds_train_dic['values'] = preds_train
preds_train_df = pd.DataFrame(preds_train_dic)

# 🌟 CHÚ Ý: Đảm bảo lưu đúng tên và đúng đường dẫn mà XGBoost sẽ đọc
# Lưu thẳng vào thư mục results để đồng bộ
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
train_save_path = os.path.join(RESULTS_DIR, 'preds_train_garch.csv')

preds_train_df.to_csv(train_save_path, index=False)
print(f"✅ Đã lưu thành công file: {train_save_path}")


# ==================== DỰ ĐOÁN ROLLING WINDOW TRÊN TẬP TEST ====================
# Mỗi bước dự đoán dùng toàn bộ train + phần test đã qua, giới hạn 3000 mẫu gần nhất

preds_test = []
preds_test_time = []
preds_test_dic = {}
window = 3000
# Tối ưu hóa tốc độ: Chuyển toàn bộ cột log_return thành Numpy Array
all_log_returns = data['log_return'].values

for i in tqdm(range(len(test_set)), desc="Test Set Progress", colour='cyan'):
    current_idx = train_size + i 
    train_data = all_log_returns[current_idx - window : current_idx]
    
    # 🌟 ĐÃ SỬA: GARCH(1,1)
    model = arch_model(
        train_data,
        p=1,
        q=1,
        dist='skewt'
    )
    model_fit = model.fit(disp='off', show_warning=False)
    prediction = model_fit.forecast(horizon=1)
    variance_forecast = prediction.variance.iloc[-1,0]
    
    # 🌟 ĐÃ SỬA: Quy đổi hệ quy chiếu
    hourly_vol_pct = np.sqrt(variance_forecast)
    hourly_vol_decimal = hourly_vol_pct / 100.0
    annualized_volatility = hourly_vol_decimal * math.sqrt(8760)
    
    preds_test.append(annualized_volatility)
    preds_test_time.append(test_time.iloc[i])

preds_test_dic['timestamp'] = preds_test_time
preds_test_dic['values'] = preds_test
preds_test_df = pd.DataFrame(preds_test_dic)

test_save_path = os.path.join(RESULTS_DIR, 'preds_test_garch.csv')
preds_test_df.to_csv(test_save_path, index=False)
print(f"✅ Đã lưu thành công file: {test_save_path}")