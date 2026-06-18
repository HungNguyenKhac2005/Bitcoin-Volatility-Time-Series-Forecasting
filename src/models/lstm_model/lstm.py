import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import numpy as np

# ========================================================= #
# 1. ĐỊNH NGHĨA MÔ HÌNH: ADVANCED BI-LSTM
# ========================================== #
class LSTMModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2):
        super().__init__()
        
        # ----------------------------------------------------
        # 1. Khối Trích xuất Đặc trưng Chuỗi: Bi-directional LSTM
        # Sử dụng Bidirectional để mô hình có thể học được bối cảnh 
        # từ cả quá khứ (forward) lẫn tương lai (backward) trong chuỗi
        # ----------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2, 
            bidirectional=True
        )
        
        # Vì là Bi-LSTM nên đầu ra sẽ bị nhân đôi kích thước (forward + backward)
        lstm_out_dim = hidden_dim * 2 
        
        # ----------------------------------------------------
        # 2. Khối Chuẩn hóa (Layer Normalization):
        # Ổn định phân phối của các hidden states, giúp mô hình hội tụ 
        # nhanh hơn và chống hiện tượng bùng nổ Gradient (Exploding Gradient)
        # ----------------------------------------------------
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        
        # ----------------------------------------------------
        # 3. Khối Neural Truyền thẳng (MLP Head):
        # Đóng vai trò tổng hợp các đặc trưng đã được trích xuất để xuất ra dự báo
        # ----------------------------------------------------
        self.fc_block = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2), # Lọc nhiễu và chống học vẹt (Overfitting)
            nn.Linear(32, 1) # Xuất ra 1 giá trị Volatility duy nhất
        )

    def forward(self, x):
        # Đẩy dữ liệu qua mạng Bi-LSTM
        # out shape: (batch_size, seq_len, num_directions * hidden_size)
        out, _ = self.lstm(x)
        
        # Trích xuất bối cảnh tại bước thời gian (time step) cuối cùng của chuỗi
        out = out[:, -1, :] 
        
        # Chuẩn hóa để ổn định luồng dữ liệu trước khi vào MLP
        out = self.layer_norm(out)
        
        # Đẩy qua MLP Head để ra kết quả dự báo cuối cùng
        # (Không dùng Softplus/Sigmoid ở layer cuối để cho phép mô hình dự báo dải giá trị linh hoạt)
        return self.fc_block(out)

# ========================================================= #
# 2. QUÁ TRÌNH HUẤN LUYỆN (TRAINING PIPELINE)
# ========================================== #
def train_model():
    # Cấu hình thiết bị (Tối ưu hóa chạy trên GPU nếu có)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Siêu tham số (Hyperparameters)
    EPOCHS = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    MODEL_NAME = "lstm" 

    # Thiết lập cây thư mục thông minh
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

    # Nạp dữ liệu
    X_train, y_train = load_data('X_train.pkl', 'Y_train.pkl')
    X_val, y_val = load_data('X_validate.pkl', 'Y_validate.pkl')

    # Đóng gói DataLoader để quản lý bộ nhớ RAM/VRAM hiệu quả
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    # Khởi tạo mô hình
    model = LSTMModel(input_dim=8).to(device) 
    
    # Tối ưu hóa: Dùng AdamW để có tính năng Weight Decay (L2 Regularization) tốt hơn Adam truyền thống
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Loss functions: HuberLoss cho quá trình backward (kháng outlier), L1Loss để đánh giá
    criterion = nn.HuberLoss(delta=1.0) 
    l1_loss = nn.L1Loss() 

    best_val_mae = float('inf')
    
    print(f"🚀 Bắt đầu huấn luyện {MODEL_NAME.upper()} (Advanced Bi-LSTM) trên thiết bị: {device}...")
    
    for epoch in range(EPOCHS):
        # --- GIAI ĐOẠN HUẤN LUYỆN (TRAINING) ---
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()               # Reset gradients
            outputs = model(batch_x).squeeze(1) # Suy luận
            loss = criterion(outputs, batch_y)  # Tính sai số Huber
            loss.backward()                     # Lan truyền ngược
            optimizer.step()                    # Cập nhật trọng số
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)

        # --- GIAI ĐOẠN ĐÁNH GIÁ (VALIDATION) ---
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad(): # Tắt theo dõi gradient
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze(1)
                
                loss = criterion(outputs, batch_y)
                mae = l1_loss(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                val_mae += mae.item() * batch_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        # In log tiến trình
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val MAE: {val_mae:.5f}")

        # Cơ chế Checkpoint: Lưu model tốt nhất dựa trên MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_best.pt")
            torch.save({'model': model.state_dict()}, save_path)
            print(f"   => 🌟 Đã lưu mô hình tốt nhất mới! (Best MAE: {best_val_mae:.5f})")

if __name__ == '__main__':
    train_model()