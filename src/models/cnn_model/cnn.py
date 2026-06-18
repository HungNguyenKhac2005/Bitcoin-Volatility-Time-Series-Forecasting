import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import numpy as np

# ========================================== #
# 1. ĐỊNH NGHĨA MÔ HÌNH: DEEP CNN 1D
# ========================================== #
class CNN1DModel(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        
        # ----------------------------------------------------
        # Khối Tích chập 1: Bắt các xu hướng vi mô (Ngắn hạn)
        # Input: (Batch, 8, 168)
        # ----------------------------------------------------
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), # Chuẩn hóa dữ liệu, giúp model hội tụ nhanh và ổn định
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # Giảm chiều dài chuỗi đi một nửa: 168 -> 84
        )
        
        # ----------------------------------------------------
        # Khối Tích chập 2: Bắt các mẫu hình trung hạn
        # Input: (Batch, 32, 84)
        # ----------------------------------------------------
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # Giảm chiều: 84 -> 42
        )
        
        # ----------------------------------------------------
        # Khối Tích chập 3: Nhận diện chu kỳ phức tạp dài hạn
        # Input: (Batch, 64, 42)
        # ----------------------------------------------------
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # Giảm chiều: 42 -> 21
        )
        
        # ----------------------------------------------------
        # Khối Fully Connected (Não bộ suy luận)
        # ----------------------------------------------------
        # Tính toán kích thước sau khi ép phẳng: 
        # Chiều dài chuỗi còn 21, số lượng bộ lọc là 128 -> 128 * 21 = 2688
        self.flatten_size = 128 * 21
        
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3), # Kỹ thuật Dropout quan trọng để tránh học vẹt dữ liệu Crypto
            nn.Linear(256, 1)  # Dự báo ra 1 giá trị duy nhất
        )

    def forward(self, x):
        # Đảo chiều để phù hợp với Conv1d (batch, channels, seq_len)
        # Từ (Batch, 168, 8) thành (Batch, 8, 168)
        x = x.permute(0, 2, 1) 
        
        # Đi qua các khối trích xuất đặc trưng
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Đưa vào khối Fully Connected để dự đoán
        out = self.fc_block(x)
        return out

# ========================================== #
# 2. QUÁ TRÌNH HUẤN LUYỆN (TRAINING LOOP)
# ========================================== #
def train_model():
    # Cấu hình thiết bị và tham số
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    
    # Tên model để tự động đặt tên file lưu
    MODEL_NAME = "cnn" 

    # Setup paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'transformer_data')
    SAVE_DIR = os.path.join(BASE_DIR, 'src', 'model_save')
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"⏳ Đang nạp dữ liệu cho {MODEL_NAME.upper()}...")
    
    # Hàm load dữ liệu
    def load_data(x_name, y_name):
        with open(os.path.join(DATA_DIR, x_name), 'rb') as f: 
            x = pickle.load(f)
        with open(os.path.join(DATA_DIR, y_name), 'rb') as f: 
            y = pickle.load(f)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    X_train, y_train = load_data('X_train.pkl', 'Y_train.pkl')
    X_val, y_val = load_data('X_validate.pkl', 'Y_validate.pkl')

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    # Init model, optimizer, loss
    model = CNN1DModel(input_dim=8).to(device) 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0) 
    l1_loss = nn.L1Loss() 

    best_val_mae = float('inf')
    
    print(f"🚀 Bắt đầu huấn luyện {MODEL_NAME.upper()} SÂU (Deep CNN 1D) trên {device}...")
    
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze(1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)

        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze(1)
                
                loss = criterion(outputs, batch_y)
                mae = l1_loss(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                val_mae += mae.item() * batch_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val MAE: {val_mae:.5f}")

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_best.pt")
            torch.save({'model': model.state_dict()}, save_path)
            print(f"   => 🌟 Đã lưu model tốt nhất mới! (MAE: {best_val_mae:.5f})")

if __name__ == '__main__':
    train_model()