

"""
AI Engineer: Nguyễn Khắc Hưng
Ngày: 30-3-2026
Mô tả: Mô hình Transformer này là một trong ba thành phần cốt lõi cấu thành nên mô hình lai (Hybrid Model) trong dự án dự báo độ 
biến động (volatility) của Bitcoin. Khác với các kiến trúc Transformer truyền thống trong NLP hay Computer Vision, phiên bản này được 
thiết kế và tinh chỉnh chuyên biệt cho bài toán chuỗi thời gian (Time Series), nó giúp Hybird model có thể học được những cú shock của 
thị trường như chiến tranh, dịch bệnh thuế gia tăng , v.v.v, nó giúp hybrid model học những dữ liệu đa chiều phức tạp, mỗi head tập chung 
khai thác 1 khía cạnh giúp model học được hét sự ảnh hưởng của feature vào target thông qua lớp Multi-Head Attention, nó còn giúp hybird 
model học được những quan hệ phi tuyến phức tạp thông qua mang nơ-ron fully conected layer, nó giúp hybird model học được hết những quan 
hệ phức tạp cái mà các model truyền thống nhưARIMA, SARIMA và GARCH không làm được, khi kết hợp model transformer này với model Garch đã 
train phía trên thì nó sẽ tối ưu hiệu quả học và cho kết quả tốt hơn rất nhiều lần 

"""

# ================================================================ #
#                        IMPORT LIBRARIES                          #
# ================================================================ #

# import các lớp trong thư viện pytorch
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# import các thư viện xử lý dữ liệu
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math

# import các thư viện trực quan hóa dữ liệu
import matplotlib.pyplot as plt
import seaborn as sns

# import các thư viện hệ thống và đường dẫn
import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'        
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    
import shutil

# import thư viện đóng gói sản phẩm và ux/ui cho trương trình
import pickle
from tqdm import tqdm

# import thư viện lấy tham số và đánh giá
from sklearn.metrics import mean_absolute_error
from argparse import ArgumentParser


# ================================================================ #
#                        CONFIGURATIONS                            # 
# ================================================================ #

def get_args():

    """
    aims to: Lấy các tham số cần thiết để training model trực tiếp từ termianl thông qua câu lệnh --key value

    Returns:
        args: Namespace chứa các tham số mà mình truyền vào (epoch, batch-size, ....)
    """

    parser = ArgumentParser(description="Transformer for time series")
    parser.add_argument("--epochs", type=int, default=100, help='Number of epochs')
    parser.add_argument("--batch_size", type=int, default=256, help='Number of batch size')
    parser.add_argument("--num_layer", type=int, default=3, help='Number of layers')
    parser.add_argument("--input_dim", type=int, default=8, help='Number of input_dim')
    parser.add_argument("--d_model", type=int, default=64, help='Number of d_models')
    parser.add_argument("--n_head", type=int, default=4, help='Number of heads')
    parser.add_argument("--d_ff", type=int, default=256, help='Number of d_ff')
    args = parser.parse_args()
    return args

# Base Directories
# Cài đặt đường dẫn
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
)

# Data Directories
DATA_DIR = os.path.join(BASE_DIR,'data')
# Data Path
X_TRAIN_DATA_PATH = os.path.join(os.path.join(DATA_DIR,'transformer_data'),'X_train.pkl')
Y_TRAIN_DATA_PATH = os.path.join(os.path.join(DATA_DIR,'transformer_data'),'Y_train.pkl')
X_VALIDATE_DATA_PATH = os.path.join(os.path.join(DATA_DIR,'transformer_data'),'X_validate.pkl')
Y_VALIDATE_DATA_PATH = os.path.join(os.path.join(DATA_DIR,'transformer_data'),'Y_validate.pkl')

# Model Save Directories
MODEL_SAVE_DIR = os.path.join(os.path.join(BASE_DIR, 'src'),'model_save')
# Model Path
MODEL_BEST_PATH = os.path.join(MODEL_SAVE_DIR,'model_best.pt')
MODEL_LAST_PATH = os.path.join(MODEL_SAVE_DIR,'model_last.pt')

# Tensorboard Directories
# Note: Dùng đường dẫn tuyệt đối do Tensorflow không hỗ trợ đường dẫn tiếng việt
TENSORBOARD_DIR = os.path.join(os.path.join(BASE_DIR, 'src'),'tensorboard')
# TENSORBOARD_DIR = os.path.join(os.path.join(BASE_DIR, 'src'),'tensorboard')
# Tensorboard Path
TENSORBOARD_PATH = os.path.join(TENSORBOARD_DIR,'tensorboard')


# ================================================================ #
#                        MODEL DEFINITION                          #
# ================================================================ #

class EmbeddingBlock(nn.Module):

    """
    Khối mã hóa đầu vào và thêm thông tin vị trí (Positional Encoding).

    Chuyển đổi dữ liệu chuỗi thời gian thô thành các vector trong không gian
    d_model chiều, đồng thời gán thông tin vị trí để model phân biệt được
    thứ tự thời gian giữa các bước.

    Args:
        input_dim (int): Số chiều của dữ liệu đầu vào (số features).
        d_model   (int): Số chiều của không gian embedding.
        max_len   (int): Độ dài chuỗi tối đa hỗ trợ Positional Encoding.

    Returns:
        Tensor shape (batch_size, window_size, d_model)
    """

    def __init__(self, input_dim, d_model, max_len=500):
        super().__init__()
        self.embedding = nn.Linear(
            input_dim,
            d_model)
        self.pos_encoding = self._get_positional_encoding(
            max_len, 
            d_model)
        self.d_model = d_model
    def _get_positional_encoding(self, max_len, d_model):
        pos = torch.arange(0, max_len).unsqueeze(1)
        i = torch.arange(0,d_model,2)
        angle_rates = pos / (10000**(i/d_model))
        pe = torch.zeros(max_len,d_model)
        pe[:,0::2] = torch.sin(angle_rates)
        pe[:, 1::2] = torch.cos(angle_rates)
        return pe.unsqueeze(0)
    def forward(self, x):
        window_slicing = x.shape[1]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:,:window_slicing,:].to(x.device)
        return x
    

class EncoderBlock(nn.Module):

    """
    Một tầng Encoder của kiến trúc Transformer.

    Gồm 2 thành phần chính:
        - Multi-Head Self-Attention: học các mối quan hệ giữa các bước thời gian.
        - Feed-Forward Network (FFN): trích xuất đặc trưng phi tuyến.
    Mỗi thành phần đều có Layer Normalization và Dropout theo chuẩn "Add & Norm".

    Args:
        d_model  (int)  : Số chiều embedding.
        n_heads  (int)  : Số đầu Attention.
        d_ff     (int)  : Số chiều ẩn của Feed-Forward Network.
        dropout  (float): Tỷ lệ Dropout (mặc định: 0.1).

    Returns:
        Tensor shape (batch_size, window_size, d_model)
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff,d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x):
        seq_len = x.shape[1] # Lấy độ dài cửa sổ (window_size)
        
        # Tạo mặt nạ che tương lai (Causal Mask)
        # Các giá trị ở tam giác trên sẽ là -inf, tam giác dưới là 0
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        
        # Đưa mask vào Attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x
    


class EncoderBlocks(nn.Module):

    """
    Ngăn xếp nhiều tầng EncoderBlock nối tiếp nhau.

    Cho phép xây dựng Encoder sâu hơn bằng cách lặp lại EncoderBlock
    num_layers lần, giúp model học được các đặc trưng ở nhiều mức độ trừu tượng.

    Args:
        num_layers (int)  : Số tầng EncoderBlock.
        d_model    (int)  : Số chiều embedding.
        n_heads    (int)  : Số đầu Attention.
        d_ff       (int)  : Số chiều ẩn của Feed-Forward Network.
        dropout    (float): Tỷ lệ Dropout (mặc định: 0.1).

    Returns:
        Tensor shape (batch_size, window_size, d_model)
    """
    
    def __init__(self, num_layers ,d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.Layers = nn.ModuleList()
        for i in range(num_layers):
            self.Layers.append(EncoderBlock(d_model, n_heads, d_ff, dropout))
    def forward(self, x):
        for i in range(len(self.Layers)):
            x = self.Layers[i](x)
        return x
    

class OutputBlock(nn.Module):
    """
    Khối đầu ra — chiếu vector đặc trưng về giá trị dự báo.

    Lấy vector tại bước thời gian cuối cùng, chiếu tuyến tính về 1 chiều.
    (Đã gỡ bỏ Softplus để mô hình có thể dự đoán được các giá trị âm 
    sau khi dữ liệu đã được đi qua Scaler).

    Args:
        d_model (int): Số chiều embedding đầu vào.

    Returns:
        Tensor shape (batch_size, 1)
    """
        
    def __init__(self,d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)
        # 🌟 ĐÃ GỠ BỎ: self.activation = nn.Softplus()
        
    def forward(self,x):
        x = x[:,-1,:]
        x = self.fc(x)
        # 🌟 ĐÃ GỠ BỎ SOFTPLUS: Trả về trực tiếp output của Linear (cho phép số âm)
        return x
    

class TimeseriesTransformer(nn.Module):

    """
    Model Transformer hoàn chỉnh cho bài toán dự báo chuỗi thời gian.

    Kết nối tuần tự 3 khối:
        1. EmbeddingBlock  : Mã hóa đầu vào + Positional Encoding.
        2. EncoderBlocks   : Học đặc trưng qua num_layers tầng Attention.
        3. OutputBlock     : Chiếu về giá trị dự báo cuối cùng.

    Args:
        num_layers (int): Số tầng EncoderBlock.
        input_dim  (int): Số chiều dữ liệu đầu vào.
        d_model    (int): Số chiều embedding.
        n_heads    (int): Số đầu Attention.
        d_ff       (int): Số chiều ẩn của Feed-Forward Network.

    Returns:
        Tensor shape (batch_size, 1) — giá trị dự báo của model.
    """
     
    def __init__(self, num_layers, input_dim, d_model, n_heads, d_ff):
        super().__init__()
        self.embeddingblock = EmbeddingBlock(input_dim, d_model)
        self.encoderblocks = EncoderBlocks(num_layers, d_model, n_heads, d_ff)
        self.outputblock = OutputBlock(d_model)
    def forward(self, x):
        x = self.embeddingblock(x)
        x = self.encoderblocks(x)
        x = self.outputblock(x)
        return x

# ================================================================ #
#                        TRAINING LOOP                             #
# ================================================================ #

if __name__ == '__main__':

    args = get_args() # Lấy các tham số từ terminal

    # Kiểm tra và thiết lập thiết bị tính toán CPU/GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Thông tin Card đồ họa GPU")
        print("Tổng số GPU có: ", torch.cuda.device_count())
        print("Tên GPU của máy: ", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        condition = int(input("Không có Card đồ họa Nvidia, nhấn 1 + Enter để dừng lại quá trình training: "))
        if(condition == 1):
            sys.exit()

    # Load dữ liệu training và tạo DataLoader
    with open(X_TRAIN_DATA_PATH,"rb") as f:
        X_train = pickle.load(f)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    # print(X_train.shape)
    # print(X_train.device)

    with open(Y_TRAIN_DATA_PATH,"rb") as f:
        Y_train = pickle.load(f)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
    # print(Y_train.shape)
    # print(Y_train.device)

    train_set = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,   # Xáo trộn dữ liệu sau mỗi epoch
        num_workers=0,
        drop_last=True  # Bỏ batch cuối nếu không đủ batch size
    )

    # Load dữ liệu validation và tạo DataLoader
    with open(X_VALIDATE_DATA_PATH,"rb") as f:
        X_validate = pickle.load(f)
    X_validate = torch.tensor(X_validate, dtype=torch.float32).to(device)
    # print(X_validate.shape)

    with open(Y_VALIDATE_DATA_PATH,"rb") as f:
        Y_validate = pickle.load(f)
    Y_validate = torch.tensor(Y_validate, dtype=torch.float32).to(device)
    # print(Y_validate.shape)

    validate_set = TensorDataset(X_validate, Y_validate)
    # print(validate_set[0])
    validate_loader = DataLoader(
        dataset=validate_set,
        batch_size=args.batch_size,
        shuffle=False,      # Không xáo trộn dữ liệu để đảm báo tính nhất quán khi đánh giá
        num_workers=0,
        drop_last=False     # Dữ lại toàn bộ dữ liệu validation
    )

    # Khởi tạo thư mục tensorboard 
    if os.path.isdir(TENSORBOARD_DIR):
        shutil.rmtree(TENSORBOARD_DIR)  # Xóa thư mục đi nếu đã tồn tại tránh log cũ
        os.makedirs(TENSORBOARD_DIR)
    else:
        os.makedirs(TENSORBOARD_DIR)

    # Khởi tạo thư mục Model save nếu chưa tồn tại
    if not os.path.isdir(MODEL_SAVE_DIR):
        os.mkdir(MODEL_SAVE_DIR)

    writer = SummaryWriter(TENSORBOARD_DIR) # khởi tạo tensorboard writer
    
    model = TimeseriesTransformer(  # Khởi tạo model transformer
        args.num_layer,             # Số lớp Encoder
        args.input_dim,             # Số chiều đầu vào
        args.d_model,               # Số chiều embedding
        args.n_head,                # Số đầu vào Attention
        args.d_ff)                  # Số chiều feed forward

    # Chuyển model lên GPU để huấn luyện, thoát nếu không có GPU
    if(device.type == "cuda"):
        model.to(device)
        print("Đã chuyển model lên GPU Nvidia")
    else:
        print("Máy không có GPU Nvidia, sẽ tự động thoát trương trình")
        sys.exit()

    # Định nghĩa hàm Loss, Optimizer và Learninf rate scheduler
    criterion = nn.HuberLoss(delta=1.0)     # Sử dụng hàm Loss Huber loss, vì nó ít nhạy cảm với outlier hơn mse

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,            # Learning rate khởi đầu giúp model học nhanh hơn
        weight_decay=1e-4)  # Learning rate cuối giúp model tránh bị overfitting
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,                        # Learning tối đa trong chu kì 
        steps_per_epoch=len(train_loader),  # số bước mỗi epoch
        epochs=args.epochs                           # Tổng số epoch huấn luyện
    )
    
    num_iters = len(train_loader) # Tổng số interation trong mỗi epoch

    # Loading check_point nếu model đã được lưu trước đó, tiếp tục training với thông số bên trong check_point
    if os.path.isfile(MODEL_LAST_PATH):
        check_points = torch.load(MODEL_LAST_PATH)
        start_epoch = check_points["epoch"]                 # Epoch đã dừng lại
        model.load_state_dict(check_points['model'])        # khôi phục trọng số model
        optimizer.load_state_dict(check_points['optimizer'])# Khôi phục trạng thái optimizer
        if 'scheduler' in check_points:
            scheduler.load_state_dict(check_points['scheduler'])
        print("đã load checkpoint thành công")
    else:
        start_epoch = 0 # Bắt đầu training từ 0
        print("ko có checkpoint")

    finish_epochs = args.epochs # Tổng số epoch cần huấn luyện
    best_MAE = float('inf')     # Khởi tạo MAE tốt nhất, dùng để lưu model tốt nhất

    for epoch in range(start_epoch, finish_epochs):

        # Training phase
        model.train()
        progress_bar = tqdm(train_loader,
                            bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
                            colour='cyan',
                            ascii=False)
        for iter, (data, value_real) in enumerate(progress_bar):
            outputs = model(data).squeeze(1)            # Forward pass
            loss_value = criterion(outputs, value_real) # tính Loss

            # Cập nhật thông tin hiện thị lên progress bar
            progress_bar.set_description(f"🚀 Epoch [{epoch+1}]")
            progress_bar.set_postfix({
                "Iter": iter + 1,
                "Loss": f"{loss_value:.4f}",
                "GPU_Mem": f"{torch.cuda.memory_allocated()/1024**2:.0f}MB"
            })

            writer.add_scalar('train/loss', loss_value,epoch*num_iters+iter) # Ghi loss lên tensorboard

            # Backward và cập nhật trọng số
            optimizer.zero_grad()   # Xóa gradient cũ
            loss_value.backward()   # Tính gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()        # Cập nhật trọng số
            scheduler.step()        # Cập nhật learning rate

        # Validation phase
        model.eval()
        all_predictions = []
        all_values = []
        for iter, (data, value) in enumerate(validate_loader):
            all_values.extend(value)
            with torch.no_grad():                       # Tắt gradient 
                prediction = model(data).squeeze(1)     # forward pass
                indices = prediction.cpu()              # Chuyển kết quả sang CPU
                all_predictions.extend(indices)
                loss_value = criterion(prediction,value)

        # Chuyển list tensor về dạng số thực đẻ tính metrics bằng sklearn
        all_predictions = [pre.item() for pre in all_predictions]
        all_values = [val.item() for val in all_values]
        # print(all_predictions)
        # print("------------------")
        # print(all_values)
        # input()
        # print("Epoch: " + str(epoch))
        # print(mean_absolute_error(all_values, all_predictions))

        # Tính và ghi MAE lên tensorboard
        MAE = mean_absolute_error(all_values,all_predictions)
        print("MAE: ", MAE)
        writer.add_scalar('val/MAE',MAE,epoch)

        # Lưu check_points
        check_point = {
            'epoch': epoch + 1, 
            'model': model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(check_point,MODEL_LAST_PATH) # Luôn lưu check point mới nhất

        # Lưu model tốt nhất dựa trên MAE
        if MAE < best_MAE:
            torch.save(check_point, MODEL_BEST_PATH) 
            best_MAE = MAE
    writer.close() 


