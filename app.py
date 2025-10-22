# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import sys
import os

# Thêm path để import modules
sys.path.append('src')
sys.path.append('models')
sys.path.append('utils')

try:
    from src.data_collection import BinanceDataCollector
    from src.data_processing import DataProcessor
    from src.visualization import CryptoVisualizer
    from models.price_predictor import PricePredictor
    from utils.helpers import format_currency, calculate_performance_metrics
except ImportError as e:
    st.error(f"📦 Lỗi import: {e}")
    st.info("🔄 Đang cài đặt dependencies...")
    try:
        os.system("pip install -r requirements.txt")
        st.rerun()
    except:
        st.error("❌ Không thể cài đặt dependencies tự động")

# Cấu hình page
st.set_page_config(
    page_title="Phân Tích Crypto - B22DCCN132",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF4B4B;
    }
    .positive {
        color: #00FF00;
    }
    .negative {
        color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

# Tiêu đề ứng dụng
st.markdown('<h1 class="main-header">🚀 PHÂN TÍCH DỮ LIỆU CRYPTO</h1>', unsafe_allow_html=True)
st.markdown("**Sinh viên:** Nguyễn Mạnh Dũng - **MSSV:** B22DCCN132 - **Môn:** Khai phá dữ liệu")
st.markdown("---")

# Khởi tạo classes
@st.cache_resource
def init_classes():
    return (
        BinanceDataCollector(),
        DataProcessor(), 
        CryptoVisualizer(),
        PricePredictor()
    )

collector, processor, visualizer, predictor = init_classes()

# Sidebar
st.sidebar.header("⚙️ CÀI ĐẶT PHÂN TÍCH")

# Lựa chọn symbol
symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT']
selected_symbol = st.sidebar.selectbox("💰 Chọn Coin:", symbols, index=0)

# Lựa chọn timeframe
timeframe = st.sidebar.selectbox("⏰ Khung thời gian:", ['1d', '4h', '1h', '15m'], index=0)

# Số lượng nến
limit = st.sidebar.slider("📊 Số lượng nến:", 100, 500, 200)

# Lựa chọn phân tích nâng cao
st.sidebar.header("🔧 CÔNG CỤ NÂNG CAO")
enable_prediction = st.sidebar.checkbox("🔮 Dự đoán giá", value=True)
enable_technical = st.sidebar.checkbox("📈 Phân tích kỹ thuật", value=True)
enable_portfolio = st.sidebar.checkbox("💼 Phân tích portfolio", value=True)

# Nút tải dữ liệu
if st.sidebar.button("🔄 TẢI DỮ LIỆU THẬT", type="primary"):
    with st.spinner("🔄 Đang kết nối Binance API..."):
        try:
            # Hiển thị progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔗 Đang kết nối Binance API...")
            progress_bar.progress(20)
            
            # Lấy dữ liệu chính
            main_data = collector.get_klines_data(
                symbol=selected_symbol, 
                interval=timeframe, 
                limit=limit
            )
            
            progress_bar.progress(60)
            status_text.text("📊 Đang xử lý dữ liệu...")
            
            if main_data is not None and not main_data.empty:
                # Xử lý dữ liệu
                processed_data = processor.calculate_technical_indicators(main_data)
                
                progress_bar.progress(80)
                
                # Lấy dữ liệu portfolio nếu được chọn
                if enable_portfolio:
                    status_text.text("💼 Đang lấy portfolio data...")
                    portfolio_symbols = ['BTCUSDT', 'ETHUSDT']  # Giảm số lượng
                    portfolio_data = collector.get_multiple_symbols(portfolio_symbols, timeframe, 100)
                    
                    if portfolio_data:
                        portfolio_df, correlation_matrix, portfolio_metrics = processor.create_portfolio_data(portfolio_data)
                    else:
                        portfolio_df, correlation_matrix, portfolio_metrics = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                else:
                    portfolio_df, correlation_matrix, portfolio_metrics = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                
                # Lưu vào session state
                st.session_state.processed_data = processed_data
                st.session_state.portfolio_df = portfolio_df
                st.session_state.correlation_matrix = correlation_matrix
                st.session_state.portfolio_metrics = portfolio_metrics
                st.session_state.symbol = selected_symbol
                
                progress_bar.progress(100)
                status_text.text("✅ Hoàn thành!")
                
                st.success(f"✅ Đã tải dữ liệu THẬT thành công cho {selected_symbol}!")
                st.info(f"📅 Dữ liệu từ {processed_data['open_time'].min().strftime('%d/%m/%Y')} đến {processed_data['open_time'].max().strftime('%d/%m/%Y')}")
                
            else:
                progress_bar.progress(0)
                status_text.text("❌ Thất bại!")
                st.error("""
                ❌ **Không thể kết nối Binance API!** 
                
                **Nguyên nhân có thể:**
                - 🌐 **Firewall** chặn kết nối từ server Streamlit
                - ⏰ **Timeout** kết nối
                - 🔄 **Rate limit** từ Binance
                
                **Giải pháp:**
                - Thử lại sau 1-2 phút
                - Giảm số lượng nến
                - Dùng khung thời gian lớn hơn (1d thay vì 1h)
                """)
                
        except Exception as e:
            st.error(f"❌ Lỗi hệ thống: {str(e)}")

# Hiển thị nội dung chính
if 'processed_data' in st.session_state:
    data = st.session_state.processed_data
    portfolio_df = st.session_state.portfolio_df
    corr_matrix = st.session_state.correlation_matrix
    portfolio_metrics = st.session_state.portfolio_metrics
    
    # Tab chính (GIỮ NGUYÊN phần này từ code cũ)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 TỔNG QUAN", "📈 BIỂU ĐỒ STATIC", "🎨 BIỂU ĐỒ TƯƠNG TÁC", 
        "🔍 PHÂN TÍCH KỸ THUẬT", "🔮 DỰ ĐOÁN", "📖 STORYTELLING"
    ])
    
    with tab1:
        st.header("📊 TỔNG QUAN THỊ TRƯỜNG")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = data['close'].iloc[-1]
            price_change = data['price_change_pct'].iloc[-1] if 'price_change_pct' in data.columns else 0
            change_color = "positive" if price_change > 0 else "negative"
            st.metric(
                "💰 Giá Hiện Tại", 
                f"${current_price:.2f}",
                f"{price_change:+.2f}%"
            )
        
        with col2:
            volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].mean()
            volume_ratio = (volume / avg_volume - 1) * 100
            st.metric(
                "📈 Khối Lượng", 
                f"{volume:,.0f}",
                f"{volume_ratio:+.1f}% vs TB"
            )
        
        with col3:
            rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 0
            rsi_status = "QUÁ MUA" if rsi > 70 else "QUÁ BÁN" if rsi < 30 else "BÌNH THƯỜNG"
            st.metric(
                "🎯 RSI", 
                f"{rsi:.1f}",
                rsi_status
            )
        
        with col4:
            volatility = data['volatility_7d'].iloc[-1] if 'volatility_7d' in data.columns else 0
            st.metric(
                "⚡ Độ Biến Động", 
                f"{volatility:.2f}%",
                "7 ngày"
            )
        
        # Price chart và thông tin cơ bản
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Biểu Đồ Giá")
            fig_price = visualizer.create_interactive_line(data, title=f"Giá {selected_symbol}")
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col2:
            st.subheader("📋 Thông Tin Cơ Bản")
            
            # Performance metrics
            metrics = calculate_performance_metrics(data)
            
            st.write(f"**📅 Thời gian:** {data['open_time'].min().strftime('%d/%m/%Y')} - {data['open_time'].max().strftime('%d/%m/%Y')}")
            st.write(f"**📊 Tổng số nến:** {len(data)}")
            st.write(f"**📈 Tổng lợi nhuận:** {metrics.get('total_return', 0):.2f}%")
            st.write(f"**🎯 Sharpe Ratio:** {metrics.get('sharpe_ratio', 0):.2f}")
            
            # Trading signals
            if 'combined_signal' in data.columns:
                signal = data['combined_signal'].iloc[-1]
                if signal > 1:
                    st.success("🎯 TÍN HIỆU: MUA")
                elif signal < -1:
                    st.error("🎯 TÍN HIỆU: BÁN")
                else:
                    st.info("🎯 TÍN HIỆU: GIỮ")
        
        # Dữ liệu thô
        st.subheader("📄 Dữ Liệu Thô (10 dòng gần nhất)")
        st.dataframe(data.tail(10), use_container_width=True)
    
    # CÁC TAB CÒN LẠI GIỮ NGUYÊN NHƯ CODE CŨ
    with tab2:
        st.header("📊 BIỂU ĐỒ STATIC - YÊU CẦU 3")
        # ... (giữ nguyên nội dung tab2 từ code cũ)
        
    with tab3:
        st.header("🎨 BIỂU ĐỒ TƯƠNG TÁC - YÊU CẦU 4") 
        # ... (giữ nguyên nội dung tab3 từ code cũ)
        
    with tab4:
        st.header("🔍 PHÂN TÍCH KỸ THUẬT")
        # ... (giữ nguyên nội dung tab4 từ code cũ)
        
    with tab5:
        st.header("🔮 DỰ ĐOÁN GIÁ")
        # ... (giữ nguyên nội dung tab5 từ code cũ)
        
    with tab6:
        st.header("📖 STORYTELLING - YÊU CẦU 5")
        # ... (giữ nguyên nội dung tab6 từ code cũ)

else:
    # Welcome screen
    st.info("👈 **Vui lòng chọn tham số và nhấn 'TẢI DỮ LIỆU THẬT' để bắt đầu phân tích**")
    
    st.warning("""
    **⚠️ Lưu ý quan trọng:**
    - Ứng dụng đang thử kết nối **Binance API thật**
    - Có thể mất 10-30 giây để lấy dữ liệu
    - Nếu thất bại, vui lòng thử lại hoặc giảm số lượng nến
    """)

# Footer
st.markdown("---")
st.markdown(
    "**Bài tập giữa kỳ - Môn Khai phá dữ liệu** • "
    "**Nguyễn Mạnh Dũng - B22DCCN132** • "
    "**PTIT - 2024**"
)