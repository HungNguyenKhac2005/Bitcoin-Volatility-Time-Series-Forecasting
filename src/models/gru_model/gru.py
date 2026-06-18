import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import numpy as np

# ========================================================= #
# 1. ĐỊNH NGHĨA MÔ HÌNH: TỐI GIẢN GRU (ULTRA-SIMPLE GRU)
# ========================================================= #
class GRUModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=8): 
        super().__init__()
        
        # ----------------------------------------------------
        # 1. Khối GRU (Gated Recurrent Unit):
        # Trích xuất đặc trưng chuỗi thời gian, giải quyết bài toán 
        # Vanishing Gradient của RNN truyền thống nhưng nhẹ hơn LSTM
        # ----------------------------------------------------
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        # ----------------------------------------------------
        # 2. Khối Fully Connected (Đầu ra):
        # Ánh xạ vector trạng thái ẩn cuối cùng thành 1 giá trị dự báo
        # ----------------------------------------------------
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        
        # Đẩy dữ liệu qua GRU.
        # 'out' chứa toàn bộ trạng thái ẩn (hidden states) của chuỗi
        out, _ = self.gru(x)
        
        # Trích xuất vector trạng thái tại bước thời gian cuối cùng (t)
        # Đây là vector mang thông tin đúc kết của toàn bộ chuỗi lịch sử
        last_hidden_state = out[:, -1, :] 
        
        # Đưa qua lớp Linear để xuất thẳng ra giá trị Volatility
        return self.fc(last_hidden_state)

# ========================================================= #
# 2. QUÁ TRÌNH HUẤN LUYỆN (TRAINING PIPELINE)
# ========================================================= #
def train_model():
    # Cấu hình thiết bị (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Siêu tham số (Hyperparameters)
    EPOCHS = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    MODEL_NAME = "gru" 

    # Thiết lập cây thư mục (Path Management) gọn gàng
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'transformer_data')
    SAVE_DIR = os.path.join(BASE_DIR, 'src', 'model_save')
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"⏳ Đang nạp dữ liệu cho mô hình {MODEL_NAME.upper()}...")
    
    # Hàm tiện ích load dữ liệu Pickle
    def load_data(x_name, y_name):
        with open(os.path.join(DATA_DIR, x_name), 'rb') as f: 
            x = pickle.load(f)
        with open(os.path.join(DATA_DIR, y_name), 'rb') as f: 
            y = pickle.load(f)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    # Nạp tập Train và Validation
    X_train, y_train = load_data('X_train.pkl', 'Y_train.pkl')
    X_val, y_val = load_data('X_validate.pkl', 'Y_validate.pkl')

    # Đóng gói DataLoader để tối ưu hóa quá trình đẩy batch vào VRAM
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    # Khởi tạo kiến trúc mạng nơ-ron
    model = GRUModel(input_dim=8, hidden_dim=8).to(device) 
    
    # Tối ưu hóa bằng AdamW (tích hợp L2 Regularization chống Overfitting)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Hàm suy hao (Loss Functions): HuberLoss kháng nhiễu outlier tốt hơn MSE
    criterion = nn.HuberLoss() 
    l1_loss = nn.L1Loss() # Dùng MAE làm Metric đánh giá độ chuẩn xác

    best_val_mae = float('inf')
    
    print(f"🚀 Bắt đầu huấn luyện {MODEL_NAME.upper()} (8 Features) trên thiết bị: {device}...")
    
    for epoch in range(EPOCHS):
        # --- GIAI ĐOẠN HUẤN LUYỆN (TRAINING) ---
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()               # Xóa bộ nhớ gradient
            outputs = model(batch_x).squeeze(1) # Suy luận (Forward pass)
            loss = criterion(outputs, batch_y)  # Tính sai số
            loss.backward()                     # Lan truyền ngược (Backward pass)
            optimizer.step()                    # Cập nhật trọng số
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)

        # --- GIAI ĐOẠN ĐÁNH GIÁ (VALIDATION) ---
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad(): # Tắt đồ thị gradient để tiết kiệm tài nguyên
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze(1)
                
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                val_mae += l1_loss(outputs, batch_y).item() * batch_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        
        # Log tiến trình huấn luyện
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train Loss (Huber): {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val MAE: {val_mae:.5f}")

        # Cơ chế Checkpoint: Chỉ lưu lại mô hình có chỉ số MAE thấp nhất
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_best.pt")
            torch.save({'model': model.state_dict()}, save_path)
            print(f"   => 🌟 Cập nhật mô hình tốt nhất (Best MAE: {best_val_mae:.5f})")

if __name__ == '__main__':
    train_model()