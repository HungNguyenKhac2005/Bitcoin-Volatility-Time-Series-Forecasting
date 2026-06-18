import pandas as pd
import os

def create_early_fusion_dataset():
    """
    Quy trình Tiền xử lý (Preprocessing Pipeline) - Early Fusion
    Mục tiêu: Hợp nhất chuỗi dự báo thống kê từ mô hình GARCH vào bộ dữ liệu gốc.
    Biến 'garch_volatility' sẽ đóng vai trò như một đặc trưng tiền nghiệm (Prior Knowledge Feature)
    để nhúng (embed) vào không gian đầu vào của các mô hình Mạng Nơ-ron (Deep Learning).
    """
    # =========================================================
    # 1. THIẾT LẬP CÂY THƯ MỤC DỰ ÁN (PATH MANAGEMENT)
    # =========================================================
    # Tự động truy xuất thư mục gốc của dự án (Lùi lên 4 cấp từ file hiện tại)
    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )
        )
    )
    
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RESULTS_DIR = os.path.join(DATA_DIR, 'results')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

    print("⏳ Khởi tạo luồng hợp nhất dữ liệu (Early Fusion Pipeline)...")
    
    # Kiểm tra tính hợp lệ của thư mục trước khi thực thi
    if not os.path.exists(RESULTS_DIR) or not os.path.exists(PROCESSED_DIR):
        print("❌ Cảnh báo: Không tìm thấy thư mục 'results' hoặc 'processed'. Hãy kiểm tra lại kiến trúc dự án!")
        return

    # =========================================================
    # 2. TRÍCH XUẤT CHUỖI DỮ LIỆU TỪ MÔ HÌNH THỐNG KÊ (GARCH p=1, q=1)
    # =========================================================
    try:
        garch_train = pd.read_csv(os.path.join(RESULTS_DIR, 'preds_train_garch.csv'))
        garch_test = pd.read_csv(os.path.join(RESULTS_DIR, 'preds_test_garch.csv'))
    except FileNotFoundError as e:
        print(f"❌ Lỗi: Thiếu file dữ liệu GARCH - {e}")
        return

    print("✅ Đã nạp thành công chuỗi dự báo Volatility từ mô hình GARCH.")
    
    # Nối (Concatenate) tập Huấn luyện và Kiểm thử thành một chuỗi thời gian liên tục.
    # Sử dụng drop_duplicates để loại bỏ các điểm dữ liệu trùng lặp ở vùng giao thoa.
    garch_all = pd.concat([garch_train, garch_test]).drop_duplicates(subset=['timestamp'])
    
    # Chuẩn hóa lại tên trường dữ liệu (Feature Renaming) để dễ quản lý trong không gian vector
    garch_all.rename(columns={'values': 'garch_volatility'}, inplace=True)

    # =========================================================
    # 3. KẾT HỢP DỮ LIỆU (DATA MERGE & ALIGNMENT)
    # =========================================================
    print("⏳ Đang đồng bộ hóa trục thời gian (Time-axis Alignment) với dữ liệu gốc...")
    try:
        df_main = pd.read_csv(os.path.join(PROCESSED_DIR, 'Bitcoin_time_series_processed.csv'))
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file gốc 'Bitcoin_time_series_processed.csv'")
        return
        
    # Áp dụng Left Join trên cột 'timestamp' để đảm bảo cấu trúc dữ liệu chính (df_main) 
    # được bảo toàn tuyệt đối 100%, chỉ ánh xạ (map) thêm biến GARCH tương ứng vào.
    df_hybrid = pd.merge(df_main, garch_all[['timestamp', 'garch_volatility']], on='timestamp', how='left')

    # =========================================================
    # 4. XỬ LÝ NHIỄU KHUYẾT THIẾU (IMPUTATION CHO MỨC BÙ QUÁ KHỨ)
    # =========================================================
    # Do mô hình GARCH cần một độ trễ ban đầu (lag) để khởi động phương trình phương sai,
    # các dòng đầu tiên (khoảng 3000 dòng) thường bị khuyết dữ liệu (NaN).
    # Kỹ thuật Backward Fill (bfill) được sử dụng để nội suy lan truyền ngược, 
    # bảo vệ tính liên tục của chuỗi dữ liệu trước khi đưa vào Deep Learning.
    missing_count = df_hybrid['garch_volatility'].isna().sum()
    print(f"🛠️ Đang xử lý {missing_count} dòng dữ liệu khởi động bị khuyết (NaN)...")
    
    df_hybrid['garch_volatility'] = df_hybrid['garch_volatility'].bfill()

    # =========================================================
    # 5. XUẤT KẾT QUẢ VÀ LƯU TRỮ (DATA EXPORT)
    # =========================================================
    save_path = os.path.join(PROCESSED_DIR, 'Bitcoin_hybrid_features.csv')
    df_hybrid.to_csv(save_path, index=False)

    print("\n" + "="*50)
    print("🎯 HOÀN TẤT QUY TRÌNH HỢP NHẤT DỮ LIỆU EARLY FUSION")
    print("="*50)
    print(f"📍 Đường dẫn xuất file: {save_path}")
    print(f"📊 Đặc trưng (Feature) mới đã thêm: 'garch_volatility'")
    print(f"📈 Tổng số mẫu (Samples)   : {len(df_hybrid)} dòng")
    print("="*50 + "\n")

if __name__ == '__main__':
    create_early_fusion_dataset()