import torch
import pandas as pd
import numpy as np
import pickle
import os
import joblib 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- IMPORT KIẾN TRÚC MÔ HÌNH ---
from lstm import LSTMModel

# ========================================================= #
# QUÁ TRÌNH INFERENCE: DỰ ĐOÁN, GIẢI MÃ SCALER VÀ ĐÁNH GIÁ
# ========================================================= #
def predict_LSTM_real():
    # 1. Cấu hình thiết bị (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Đang khởi chạy quá trình dự đoán Bi-LSTM trên thiết bị: {device}")

    # 2. Thiết lập cây thư mục (Path Management)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models')
    TRANSFORMER_DATA_DIR = os.path.join(DATA_DIR, 'transformer_data')
    
    X_TEST_PATH = os.path.join(TRANSFORMER_DATA_DIR, 'X_test.pkl')
    # Trỏ đúng đến file trọng số của mô hình LSTM
    MODEL_PATH = os.path.join(BASE_DIR, 'src', 'model_save', 'lstm_best.pt')
    SCALER_PATH = os.path.join(BASE_DIR, 'src', 'model_save', 'scaler.pkl')

    # =========================================================
    # BƯỚC 1: LOAD METADATA VÀ GROUND TRUTH
    # =========================================================
    print("⏳ Đang nạp metadata (timestamp & ground truth) để đồng bộ kiểm thử...")
    test_set_transformer_path = os.path.join(DATA_DIR, "results", "test_set_transformer_timestamp.csv")
    test_set_path = os.path.join(DATA_DIR, "results", "test_set.csv")

    if not os.path.exists(test_set_transformer_path):
        print(f"❌ Cảnh báo: Không tìm thấy file {test_set_transformer_path}. Vui lòng chạy luồng tiền xử lý trước!")
        return
        
    test_set_transformer_timestamp = pd.read_csv(test_set_transformer_path)
    test_set = pd.read_csv(test_set_path)

    print(f"✅ Nạp metadata thành công! (Tổng số mẫu Test: {len(test_set_transformer_timestamp)} dòng)")

    # =========================================================
    # BƯỚC 2: KHỞI TẠO MÔ HÌNH & CHẠY INFERENCE
    # =========================================================
    print("⏳ Đang nạp trọng số (weights) của mô hình Bi-LSTM tốt nhất...")
    # Khởi tạo kiến trúc mô hình (khớp với file huấn luyện)
    model = LSTMModel(input_dim=8).to(device) 
    
    if os.path.exists(MODEL_PATH):
        # Nạp state_dict an toàn (weights_only=True)
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
    else:
        print(f"❌ Lỗi: Không tìm thấy file trọng số tại {MODEL_PATH}")
        return
        
    # Chuyển mô hình sang chế độ suy luận (Tắt Dropout để dự đoán ổn định)
    model.eval()

    print("🧠 Đang xử lý tính toán (Forward Pass) trên tập Test...")
    with open(X_TEST_PATH, "rb") as f:
        X_test = pickle.load(f)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad(): # Tắt tính toán Gradient để tối ưu tốc độ và giải phóng VRAM
        preds_scaled = model(X_test_tensor).squeeze(1).cpu().numpy()

    # =========================================================
    # BƯỚC 3: GIẢI MÃ DỮ LIỆU (INVERSE SCALING)
    # =========================================================
    print("🔄 Đang khôi phục hệ quy chiếu (Inverse Transform) về giá trị Volatility thực tế...")
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        
        # Kỹ thuật Dummy Array: Tạo mảng 0 với kích thước 9 cột (8 features + 1 target)
        dummy_array = np.zeros((len(preds_scaled), 9))
        
        # Gắn kết quả dự đoán vào đúng vị trí của cột Volatility (cột cuối cùng - index 8)
        TARGET_COL_INDEX = 8
        dummy_array[:, TARGET_COL_INDEX] = preds_scaled
        
        # Dùng Scaler để chuyển đổi ngược và trích xuất lại cột mục tiêu
        dummy_inversed = scaler.inverse_transform(dummy_array)
        preds_real = dummy_inversed[:, TARGET_COL_INDEX]
    else:
        print("❌ Lỗi: Không tìm thấy file scaler.pkl!")
        return

    # Lưu trữ mảng dự đoán ra file .pkl phục vụ pha Stacking hoặc so sánh tổng hợp
    os.makedirs(MODELS_DIR, exist_ok=True)
    pkl_save_path = os.path.join(MODELS_DIR, 'preds_real_lstm.pkl')
    with open(pkl_save_path, 'wb') as f:
        pickle.dump(preds_real, f)
    print(f"✅ Đã lưu kết quả dự đoán (Dạng thô) vào: {pkl_save_path}")

    # =========================================================
    # BƯỚC 4: TÍNH TOÁN SAI SỐ & TRỰC QUAN HÓA (VISUALIZATION)
    # =========================================================
    # Tính toán các chỉ số đánh giá Metric
    mae = mean_absolute_error(test_set['volatility_14_annual'], preds_real)
    rmse = np.sqrt(mean_squared_error(test_set['volatility_14_annual'], preds_real))
    
    # Thiết lập và vẽ biểu đồ so sánh (Tối ưu hóa hiển thị)
    plt.figure(figsize=(14, 6))
    
    # Sử dụng màu 'deeppink' để hiển thị rõ hơn trên nền trắng, thay cho màu 'pink' nhạt
    plt.plot(test_set_transformer_timestamp['timestamp'], preds_real, color='deeppink', label='Bi-LSTM Predicted', linewidth=1.5)
    plt.plot(test_set_transformer_timestamp['timestamp'], test_set['volatility_14_annual'], color='black', label='Actual Volatility', alpha=0.7)
    
    # Định dạng trục X (Timestamp) để không bị lộn xộn
    ticks = np.linspace(0, len(test_set_transformer_timestamp['timestamp']) - 1, 10, dtype=int)
    plt.xticks(ticks, rotation=45)
    
    plt.title("So sánh Độ biến động (Volatility): Dự đoán bởi Bi-LSTM vs Thực tế")
    plt.xlabel("Thời gian (Timestamp)")
    plt.ylabel("Độ biến động (Annualized Volatility)")
    plt.legend()
    plt.tight_layout() # Đảm bảo các nhãn không bị cắt xén
    plt.show()

    # Báo cáo kết quả trên Console
    print("\n" + "="*50)
    print("📊 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH BI-LSTM (HỆ THỰC TẾ)")
    print("="*50)
    print(f"🔹 Mean Absolute Error (MAE)  : {mae:.5f}")
    print(f"🔹 Root Mean Squared Error (RMSE) : {rmse:.5f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    predict_LSTM_real()