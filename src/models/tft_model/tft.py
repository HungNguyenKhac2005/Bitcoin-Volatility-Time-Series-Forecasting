import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import numpy as np

# ========================================================= #
# 1. ĐỊNH NGHĨA MÔ HÌNH: LITE TEMPORAL FUSION TRANSFORMER (TFT)
# ========================================================= #
class LiteTFTModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, n_heads=4):
        super().__init__()
        
        # ----------------------------------------------------
        # 1. Khối Xử lý Cục bộ (Local Processing): LSTM
        # Dùng LSTM để trích xuất các mẫu hình tuần tự (sequential patterns) 
        # và sự phụ thuộc ngắn hạn (short-term dependencies) trong chuỗi thời gian.
        # ----------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        
        # ----------------------------------------------------
        # 2. Khối Đối chiếu Toàn cục (Global Processing): Multihead Attention
        # Nhận đầu ra từ LSTM và tìm kiếm các mối liên hệ dài hạn (long-term 
        # correlations) xuyên suốt toàn bộ chuỗi thông qua cơ chế Tự chú ý.
        # ----------------------------------------------------
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=n_heads, 
            batch_first=True, 
            dropout=0.2
        )
        
        # ----------------------------------------------------
        # 3. Khối Chuẩn hóa (Add & Norm):
        # Kết hợp Residual Connection (Cộng dồn) và Layer Normalization
        # Giúp bảo toàn thông tin gốc và ổn định quá trình huấn luyện
        # ----------------------------------------------------
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # ----------------------------------------------------
        # 4. Khối Bộ giải mã (MLP Decoder Head):
        # Tổng hợp các luồng thông tin để đưa ra dự báo Volatility cuối cùng.
        # ----------------------------------------------------
        self.fc_block = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2), # Chống học vẹt (Overfitting)
            nn.Linear(32, 1) # Đầu ra tuyến tính (Linear), cho phép dự báo giá trị linh hoạt
        )

    def forward(self, x):
        # Bước 1: Trích xuất đặc trưng ngắn hạn qua LSTM
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        
        # Bước 2: Tìm kiếm liên kết dài hạn qua Self-Attention
        # Query, Key, Value đều là lstm_out
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Bước 3: Cơ chế Add & Norm (Residual Connection + Layer Normalization)
        # Giảm thiểu Vanishing Gradient và giữ lại luồng thông tin cục bộ của LSTM
        enriched_out = self.layer_norm(lstm_out + attn_output)
        
        # Bước 4: Trích xuất vector thông tin tại bước thời gian cuối cùng (t)
        final_out = enriched_out[:, -1, :]
        
        # Bước 5: Đưa qua khối MLP để ra kết quả dự báo
        return self.fc_block(final_out)

# ========================================================= #
# 2. QUÁ TRÌNH HUẤN LUYỆN (TRAINING PIPELINE)
# ========================================================= #
def train_model():
    # Cấu hình phần cứng (Tự động nhận diện GPU nếu có)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Siêu tham số (Hyperparameters)
    EPOCHS = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    MODEL_NAME = "tft" 

    # Quản lý đường dẫn (Path Management) thống nhất
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'transformer_data')
    SAVE_DIR = os.path.join(BASE_DIR, 'src', 'model_save')
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"⏳ Đang nạp dữ liệu cho mô hình {MODEL_NAME.upper()}...")
    
    # Hàm đọc dữ liệu Pickle
    def load_data(x_name, y_name):
        with open(os.path.join(DATA_DIR, x_name), 'rb') as f: 
            x = pickle.load(f)
        with open(os.path.join(DATA_DIR, y_name), 'rb') as f: 
            y = pickle.load(f)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    # Nạp dữ liệu
    X_train, y_train = load_data('X_train.pkl', 'Y_train.pkl')
    X_val, y_val = load_data('X_validate.pkl', 'Y_validate.pkl')

    # Đóng gói dữ liệu thành các Batch để tối ưu VRAM
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    # Khởi tạo mô hình Lite TFT
    model = LiteTFTModel(input_dim=8).to(device) 
    
    # Tối ưu hóa với AdamW (Bao gồm L2 Regularization để phạt các trọng số quá lớn)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Hàm mất mát: HuberLoss kháng outlier (bất thường) tốt hơn MSE
    criterion = nn.HuberLoss(delta=1.0) 
    l1_loss = nn.L1Loss() # Dùng MAE để theo dõi hiệu suất thực

    best_val_mae = float('inf')
    
    print(f"🚀 Bắt đầu huấn luyện {MODEL_NAME.upper()} (Lite Temporal Fusion Transformer) trên thiết bị: {device}...")
    
    for epoch in range(EPOCHS):
        # --- GIAI ĐOẠN HUẤN LUYỆN (TRAINING) ---
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()               # Reset gradients
            outputs = model(batch_x).squeeze(1) # Suy luận
            loss = criterion(outputs, batch_y)  # Đánh giá sai số Huber
            loss.backward()                     # Lan truyền ngược
            optimizer.step()                    # Cập nhật tham số
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)

        # --- GIAI ĐOẠN KIỂM ĐỊNH (VALIDATION) ---
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad(): # Tắt tính toán gradient
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze(1)
                
                loss = criterion(outputs, batch_y)
                mae = l1_loss(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                val_mae += mae.item() * batch_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        # In nhật ký huấn luyện
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val MAE: {val_mae:.5f}")

        # Cơ chế Checkpoint: Chỉ lưu mô hình có sai số MAE thấp nhất trên tập Validate
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_best.pt")
            torch.save({'model': model.state_dict()}, save_path)
            print(f"   => 🌟 Đã lưu cấu hình mô hình tốt nhất (Best MAE: {best_val_mae:.5f})")

if __name__ == '__main__':
    train_model()