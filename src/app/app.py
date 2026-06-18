import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import torch
import joblib

# Import model Transformer đã train
try:
    from transformer_model import TimeseriesTransformer, get_args
except ImportError:
    st.error("Missing transformer_model.py. Ensure it's in the same directory.")

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Risk Control Engine | Binance Style",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CACHE MODEL & SCALER
# ==========================================
@st.cache_resource
def load_ai_model():
    """Load model 1 lần duy nhất để tránh tràn VRAM khi rerender liên tục."""
    try:
        args = get_args()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        MODEL_PATH = os.path.join(BASE_DIR, 'src', 'model_save', 'model_best.pt')
        SCALER_PATH = os.path.join(BASE_DIR, 'src', 'model_save', 'scaler.pkl')

        # Khởi tạo kiến trúc
        model = TimeseriesTransformer(
            num_layers=args.num_layer, 
            input_dim=args.input_dim, 
            d_model=args.d_model, 
            n_heads=args.n_head, 
            d_ff=args.d_ff
        ).to(device)
        
        # Load weights
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        model.eval() 

        # Load scaler
        scaler = joblib.load(SCALER_PATH)

        return model, scaler, args, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

ai_model, ai_scaler, ai_args, ai_device = load_ai_model()

# ==========================================
# CUSTOM CSS (BINANCE DARK THEME)
# ==========================================
def inject_custom_css():
    bg_img = 'url("https://static.vecteezy.com/system/resources/previews/014/296/021/non_2x/binance-coin-banner-bnb-cryptocurrency-concept-banner-background-vector.jpg")' 
    st.markdown(f"""
        <style>
        .stApp {{ background-color: #0b0e11 !important; color: #E5E7EB; font-family: 'Inter', 'Segoe UI', sans-serif; }}
        [data-testid="stMain"] {{
            background-image: linear-gradient(rgba(11, 14, 17, 0.85), rgba(11, 14, 17, 0.95)), {bg_img} !important;
            background-size: cover !important; background-position: center right !important; 
            background-repeat: no-repeat !important; background-attachment: fixed !important;
        }}
        [data-testid="stHeader"] {{ background: transparent !important; }}
        [data-testid="stSidebar"] {{
            background-color: #0B0F19 !important; background-image: linear-gradient(180deg, #0B0F19 0%, #111827 100%) !important;
            border-right: 1px solid #1F2937 !important; box-shadow: 2px 0 10px rgba(0, 0, 0, 0.5) !important;
        }}
        [data-testid="stSidebar"] > div:first-child {{ background: transparent !important; }}
        #MainMenu, footer, header {{visibility: hidden;}}
        h1, h2, h3 {{ color: #F9FAFB !important; font-weight: 700 !important; }}
        p, span, label p {{ color: #E5E7EB !important; }}
        hr {{ border-color: #374151 !important; margin: 1.5rem 0 !important; }}
        
        /* Dashboard Cards */
        .glass-card {{
            background: rgba(30, 35, 41, 0.85) !important; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            border: 1px solid #374151 !important; border-radius: 8px; padding: 20px; margin-bottom: 20px; transition: all 0.3s ease;
        }}
        
        /* Notifications Scrollbar */
        .notify-panel {{ height: 800px; overflow-y: auto; padding-right: 10px; }}
        .notify-panel::-webkit-scrollbar {{ width: 6px; }}
        .notify-panel::-webkit-scrollbar-track {{ background: rgba(0,0,0,0.1); border-radius: 4px; }}
        .notify-panel::-webkit-scrollbar-thumb {{ background: #374151; border-radius: 4px; }}
        .notify-panel::-webkit-scrollbar-thumb:hover {{ background: #F3BA2F; }}
        
        /* Alert Boxes */
        .alert-box {{ background: rgba(11, 14, 17, 0.7); border: 1px solid #374151; border-left: 4px solid #374151; border-radius: 6px; padding: 15px; margin-bottom: 15px; }}
        .alert-danger {{ border-left-color: #F6465D; }}
        .alert-warning {{ border-left-color: #F3BA2F; }}
        .alert-safe {{ border-left-color: #0ECB81; }}
        .alert-email {{ border-left-color: #9333EA; background: rgba(147, 51, 234, 0.15); }}
        .alert-time {{ font-size: 0.75rem; color: #848e9c; margin-bottom: 5px; }}
        .alert-title {{ font-size: 0.95rem; font-weight: bold; color: #E5E7EB; margin: 0 0 5px 0; }}
        .alert-desc {{ font-size: 0.85rem; color: #9CA3AF; line-height: 1.4; margin: 0; }}
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ==========================================
# SESSION STATE MANAGEMENT
# ==========================================
# Lưu trữ trạng thái để tránh mất data khi Streamlit rerender
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'data_history' not in st.session_state:
    st.session_state.data_history = pd.DataFrame(columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volatility', 'BadDebt', 'MaxLev', 'ReqMargin'])
if 'alerts' not in st.session_state:
    st.session_state.alerts = [] 
if 'last_price' not in st.session_state:
    st.session_state.last_price = 95000.00 # Base price cho việc sinh nến

# Init dummy tensor (Batch=1, Seq_len=14, Features=7)
if 'current_input_tensor' not in st.session_state:
    seq_len = getattr(ai_args, 'seq_len', 14) if ai_args else 14
    input_features = ai_args.input_dim if ai_args else 7
    st.session_state.current_input_tensor = torch.randn(1, seq_len, input_features, dtype=torch.float32)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #F3BA2F;'>🛡️ AI Risk Engine</h2>", unsafe_allow_html=True)
    
    # Control Buttons
    col_play, col_stop = st.columns(2)
    if col_play.button("▶️ Start Live", use_container_width=True):
        st.session_state.is_running = True
    if col_stop.button("⏹️ Stop", use_container_width=True):
        st.session_state.is_running = False
        
    if st.button("🗑️ Reset Data", use_container_width=True):
        # Clear toàn bộ session
        st.session_state.data_history = pd.DataFrame(columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volatility', 'BadDebt', 'MaxLev', 'ReqMargin'])
        st.session_state.alerts = []
        st.session_state.last_price = 95000.00
        st.session_state.current_input_tensor = torch.randn(1, getattr(ai_args, 'seq_len', 14), getattr(ai_args, 'input_dim', 7), dtype=torch.float32)
        st.session_state.is_running = False
        st.rerun()

    st.markdown("---")
    st.markdown("### 💰 Financial Inputs")
    capital = st.slider("Portfolio Capital (USD)", min_value=10000, max_value=10000000, value=1000000, step=10000, format="$%d")
    leverage = st.slider("Current Leverage", min_value=1, max_value=100, value=20, step=1)
    
    status_color = "#0ECB81" if st.session_state.is_running else "#F6465D"
    status_text = "🟢 ENGINE RUNNING" if st.session_state.is_running else "🔴 ENGINE STOPPED"
    st.markdown(f"<div style='text-align:center; padding:10px; border:1px solid {status_color}; color:{status_color}; font-weight:bold; border-radius:5px;'>{status_text}</div>", unsafe_allow_html=True)

# ==========================================
# LIVE INFERENCE CORE
# ==========================================
if st.session_state.is_running and ai_model is not None:
    current_time = datetime.now()
    
    # 1. Update Input Tensor
    # Thêm nhiễu ngẫu nhiên vào timestep cuối cùng, trượt tensor đi 1 nhịp
    old_tensor = st.session_state.current_input_tensor
    input_features = ai_args.input_dim
    new_step_data = old_tensor[:, -1:, :] + torch.randn(1, 1, input_features) * 0.05
    new_tensor = torch.cat((old_tensor[:, 1:, :], new_step_data), dim=1)
    st.session_state.current_input_tensor = new_tensor 
    
    # 2. Forward Pass
    with torch.no_grad():
        pred_scaled = ai_model(new_tensor.to(ai_device)).squeeze().cpu().numpy()
        if pred_scaled.ndim > 0:
            pred_scaled = pred_scaled[-1]
        else:
            pred_scaled = float(pred_scaled)

    # 3. Inverse Transform & Scaling
    # Model predict ra Annualized Volatility, cần scale lại về khung 2-8% cho bài toán demo realtime
    dummy_array = np.zeros((1, 9))
    dummy_array[0, 8] = pred_scaled
    raw_vol = ai_scaler.inverse_transform(dummy_array)[0, 8]
    
    raw_vol_percent = abs(raw_vol) * 100
    new_vol = 2.0 + (raw_vol_percent % 60) / 10.0
    new_vol = np.clip(new_vol, 2.0, 8.0) # Đảm bảo vol luôn nằm trong khoảng 2% - 8%

    # 4. Exchange Rules
    # Cấu hình mức margin theo vol
    if new_vol < 3.0: max_lev, req_margin = 100, 1.0
    elif new_vol < 6.0: max_lev, req_margin = 25, 5.0
    elif new_vol < 12.0: max_lev, req_margin = 10, 15.0
    else: max_lev, req_margin = 3, 35.0

    # 5. Price Simulation
    # Giá chạy ngẫu nhiên nhưng chịu chi phối bởi độ lớn của vol (new_vol)
    prev_close = st.session_state.last_price
    price_change_pct = np.random.normal(0, (new_vol / 100) * 0.2) 
    new_close = prev_close * (1 + price_change_pct)
    
    # Giới hạn giá trong vùng 80k-110k
    if new_close > 110000:
        new_close = 110000 - abs(new_close - 110000)
    elif new_close < 80000:
        new_close = 80000 + abs(80000 - new_close)
        
    new_open = prev_close
    candle_spread = abs(prev_close * np.random.normal(0, (new_vol / 100) * 0.1)) # Tính độ dài râu nến
    new_high = max(new_open, new_close) + candle_spread
    new_low = min(new_open, new_close) - candle_spread
    st.session_state.last_price = new_close

    # 6. Risk Calculations
    total_loaned = capital * (leverage - 1) if leverage > 1 else 0
    bad_debt_est = total_loaned * (new_vol / 100) * (leverage / max_lev) * 0.1 # Ước tính nợ xấu theo đòn bẩy quá hạn
    required_deposit = capital * (req_margin / 100) # Margin call yêu cầu

    # Update df
    new_row = pd.DataFrame([{
        'Time': current_time, 'Open': new_open, 'High': new_high, 'Low': new_low, 'Close': new_close,
        'Volatility': new_vol, 'BadDebt': bad_debt_est, 'MaxLev': max_lev, 'ReqMargin': req_margin
    }])
    st.session_state.data_history = pd.concat([st.session_state.data_history, new_row], ignore_index=True)

    # 7. Alert Management
    time_str = current_time.strftime("%H:%M:%S")
    
    if leverage > max_lev:
        st.session_state.alerts.insert(0, {
            'type': 'danger', 'time': time_str, 'title': '🚨 CẢNH BÁO ĐÒN BẨY', 
            'desc': f'Volatility vọt lên {new_vol:.2f}%. Đòn bẩy {leverage}x đang rủi ro cực cao. Vui lòng hạ xuống {max_lev}x.'
        })
    
    if required_deposit > capital * 0.1:
        st.session_state.alerts.insert(0, {
            'type': 'email', 'time': time_str, 'title': '📧 MARGIN CALL EMAIL SENT', 
            'desc': f'Dự báo Volatility cao. Đã tự động gửi Email khẩn cấp yêu cầu người dùng nạp thêm ${required_deposit:,.0f}.'
        })
    elif new_vol > 6.0: 
        st.session_state.alerts.insert(0, {
            'type': 'warning', 'time': time_str, 'title': '⚠️ CẢNH BÁO NỢ XẤU', 
            'desc': f'Gia tốc giá tăng đột biến. Ước tính hệ thống đối mặt với khoản nợ xấu ${bad_debt_est:,.0f}.'
        })
        
    # Giới hạn danh sách alert để tránh tràn RAM
    st.session_state.alerts = st.session_state.alerts[:20]

# ==========================================
# EXTRACT LATEST METRICS FOR KPI
# ==========================================
if len(st.session_state.data_history) > 0:
    latest = st.session_state.data_history.iloc[-1]
    curr_vol, curr_bad_debt, curr_req_margin = latest['Volatility'], latest['BadDebt'], latest['ReqMargin']
else:
    curr_vol, curr_bad_debt, curr_req_margin = 0.0, 0.0, 0.0

total_loaned_display = capital * (leverage - 1) if leverage > 1 else 0
req_deposit_display = capital * (curr_req_margin / 100)

# ==========================================
# MAIN DASHBOARD RENDER
# ==========================================
st.markdown("<h1 style='text-align: center;'>AI-Powered Crypto Volatility Engine</h1>", unsafe_allow_html=True)
st.markdown("<div style='height:2px; background:linear-gradient(90deg, transparent, #F3BA2F, transparent); margin:10px 0 20px 0;'></div>", unsafe_allow_html=True)

# --- KPI CARDS ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="glass-card"><div class="kpi-title" style="color:#848e9c;">🏦 TỔNG TIỀN CHO VAY</div><div class="kpi-value" style="color:#0ECB81;">${total_loaned_display:,.0f}</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="glass-card" style="border-left: 3px solid #F6465D;"><div class="kpi-title" style="color:#848e9c;">⚠️ DỰ KIẾN NỢ XẤU</div><div class="kpi-value" style="color:#F6465D;">${curr_bad_debt:,.0f}</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="glass-card"><div class="kpi-title" style="color:#848e9c;">📈 AI VOLATILITY (LIVE)</div><div class="kpi-value" style="color:#F3BA2F;">{curr_vol:.2f}%</div></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="glass-card" style="border-left: 3px solid #9333EA;"><div class="kpi-title" style="color:#848e9c;">💰 YÊU CẦU NẠP (MARGIN CALL)</div><div class="kpi-value" style="color:#9333EA;">${req_deposit_display:,.0f}</div></div>""", unsafe_allow_html=True)

# --- BỐ CỤC CHÍNH ---
main_col_charts, main_col_notify = st.columns([7.5, 2.5])
df = st.session_state.data_history

with main_col_charts:
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        st.markdown("### 🕯️ Nến BTC & Dải thanh lý (Live)")
        fig_candle = go.Figure()
        if len(df) > 0:
            fig_candle.add_trace(go.Candlestick(x=df['Time'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], increasing_line_color='#0ECB81', decreasing_line_color='#F6465D'))
            margin_buffer = 1 / leverage if leverage > 0 else 1
            liq_lower = df['Close'].iloc[-1] * (1 - margin_buffer) # Đường thanh lý
            fig_candle.add_hline(y=liq_lower, line_dash="dot", line_color="#F3BA2F", annotation_text=f"Vùng thanh lý (Long)", annotation_font_color="#F3BA2F")
        
        fig_candle.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False, yaxis=dict(autorange=True, tickformat="$,.0f"), height=350)
        st.plotly_chart(fig_candle, use_container_width=True)

    with row1_col2:
        st.markdown("### 📊 AI Volatility (Từ Model .pt)")
        fig_vol = go.Figure()
        if len(df) > 0:
            fig_vol.add_trace(go.Scatter(x=df['Time'], y=df['Volatility'], mode='lines+markers', line=dict(color='#F3BA2F', width=3), fill='tozeroy', fillcolor='rgba(243, 186, 47, 0.1)'))
            if df['Volatility'].max() > 10:
                fig_vol.add_hline(y=12, line_dash="dash", line_color="#F6465D", annotation_text="Vùng Nguy hiểm (>12%)")
        
        fig_vol.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange=True), height=350)
        st.plotly_chart(fig_vol, use_container_width=True)

    with row2_col1:
        st.markdown("### ⚠️ Nợ Xấu Ước Tính (Real-time)")
        fig_bd = go.Figure()
        if len(df) > 0:
            fig_bd.add_trace(go.Scatter(x=df['Time'], y=df['BadDebt'], mode='lines', line=dict(color='#F6465D', width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(246, 70, 93, 0.2)'))
        fig_bd.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange=True, tickformat="$,.0f"), height=350)
        st.plotly_chart(fig_bd, use_container_width=True)

    with row2_col2:
        st.markdown("### ⚙️ Max Leverage Allowed (Real-time)")
        fig_lev = go.Figure()
        if len(df) > 0:
            fig_lev.add_trace(go.Scatter(x=df['Time'], y=df['MaxLev'], mode='lines+markers', line=dict(color='#0ECB81', width=3, shape='hv'))) 
            fig_lev.add_hline(y=leverage, line_dash="solid", line_color="#F9FAFB", annotation_text=f"User Leverage ({leverage}x)")
        fig_lev.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Multiplier (x)", height=350)
        st.plotly_chart(fig_lev, use_container_width=True)

# --- NOTIFICATIONS PANEL ---
with main_col_notify:
    st.markdown("### 🔔 System Live Alerts")
    alerts_html = ""
    
    if len(st.session_state.alerts) == 0:
        alerts_html = "<div style='color:#848e9c; text-align:center; margin-top:50px;'>Đang chờ tín hiệu từ AI...</div>"
    else:
        # Render các alert từ danh sách
        for alert in st.session_state.alerts:
            alerts_html += f"""
            <div class="alert-box alert-{alert['type']}">
                <div class="alert-time">{alert['time']}</div>
                <p class="alert-title">{alert['title']}</p>
                <p class="alert-desc">{alert['desc']}</p>
            </div>"""
            
    st.markdown(f"""<div class="glass-card notify-panel">{alerts_html}</div>""", unsafe_allow_html=True)

# ==========================================
# LOOP THỜI GIAN THỰC (AUTO RERUN)
# ==========================================
if st.session_state.is_running:
    time.sleep(2) # Delay 2s để demo
    st.rerun() # Trigger Streamlit rerun toàn bộ app