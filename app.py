# Thêm đoạn này ở đầu file app.py
import streamlit as st
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
    st.error(f"📦 Đang cài đặt dependencies... {e}")
    # Tự động cài đặt nếu thiếu
    os.system("pip install -r requirements.txt")
    st.rerun()

# Phần còn lại của code giữ nguyên...
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

from src.data_collection import BinanceDataCollector
from src.data_processing import DataProcessor
from src.visualization import CryptoVisualizer
from models.price_predictor import PricePredictor
from utils.helpers import format_currency, calculate_performance_metrics

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
limit = st.sidebar.slider("📊 Số lượng nến:", 100, 1000, 500)

# Lựa chọn phân tích nâng cao
st.sidebar.header("🔧 CÔNG CỤ NÂNG CAO")
enable_prediction = st.sidebar.checkbox("🔮 Dự đoán giá", value=True)
enable_technical = st.sidebar.checkbox("📈 Phân tích kỹ thuật", value=True)
enable_portfolio = st.sidebar.checkbox("💼 Phân tích portfolio", value=True)

# Nút tải dữ liệu
if st.sidebar.button("🔄 TẢI DỮ LIỆU", type="primary"):
    with st.spinner("🔄 Đang tải dữ liệu từ Binance..."):
        try:
            # Lấy dữ liệu chính
            main_data = collector.get_klines_data(
                symbol=selected_symbol, 
                interval=timeframe, 
                limit=limit
            )
            
            if main_data is not None and not main_data.empty:
                # Xử lý dữ liệu
                processed_data = processor.calculate_technical_indicators(main_data)
                
                # Lấy dữ liệu portfolio nếu được chọn
                if enable_portfolio:
                    portfolio_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
                    portfolio_data = collector.get_multiple_symbols(portfolio_symbols, timeframe, 300)
                    portfolio_df, correlation_matrix, portfolio_metrics = processor.create_portfolio_data(portfolio_data)
                else:
                    portfolio_df, correlation_matrix, portfolio_metrics = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                
                # Lưu vào session state
                st.session_state.processed_data = processed_data
                st.session_state.portfolio_df = portfolio_df
                st.session_state.correlation_matrix = correlation_matrix
                st.session_state.portfolio_metrics = portfolio_metrics
                st.session_state.symbol = selected_symbol
                
                st.success(f"✅ Đã tải dữ liệu thành công cho {selected_symbol}!")
                
            else:
                st.error("❌ Không thể tải dữ liệu. Vui lòng thử lại!")
                
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")

# Hiển thị nội dung chính
if 'processed_data' in st.session_state:
    data = st.session_state.processed_data
    portfolio_df = st.session_state.portfolio_df
    corr_matrix = st.session_state.correlation_matrix
    portfolio_metrics = st.session_state.portfolio_metrics
    
    # Tab chính
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
            price_change = data['price_change_pct'].iloc[-1]
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
            status_color = "negative" if rsi > 70 else "positive" if rsi < 30 else ""
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
            st.write(f"**📈 Tổng lợi nhuận:** {metrics['total_return']:.2f}%")
            st.write(f"**🎯 Sharpe Ratio:** {metrics['sharpe_ratio']:.2f}")
            st.write(f"**📉 Max Drawdown:** {metrics['max_drawdown']:.2f}%")
            
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
    
    with tab2:
        st.header("📊 BIỂU ĐỒ STATIC - YÊU CẦU 3")
        
        # YÊU CẦU 3.1: Histogram & Boxplot
        st.subheader("📊 1. Histogram & Boxplot - Phân Phối Biến Động")
        col1, col2 = st.columns(2)
        
        with col1:
            dist_column = st.selectbox("Chọn cột phân phối:", 
                                     ['price_change_pct', 'volume', 'RSI', 'volatility_7d'])
        
        fig_hist_box = visualizer.create_histogram_boxplot(data, dist_column)
        st.pyplot(fig_hist_box)
        
        # YÊU CẦU 3.2: Line & Area
        st.subheader("📈 2. Line & Area Chart - Xu Hướng Giá")
        fig_line_area = visualizer.create_line_area_chart(data)
        st.pyplot(fig_line_area)
        
        # YÊU CẦU 3.3: Scatter + Regression
        st.subheader("🔍 3. Scatter Plot & Hồi Quy")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("Trục X:", ['volume', 'RSI', 'MACD', 'volatility_7d'])
        with col2:
            y_axis = st.selectbox("Trục Y:", ['price_change_pct', 'close', 'volume_ratio'])
        with col3:
            hue_axis = st.selectbox("Màu sắc:", ['RSI', 'price_change_pct', 'volume_ratio'])
        
        fig_scatter = visualizer.create_scatter_regression(data, x_axis, y_axis, hue_axis)
        st.pyplot(fig_scatter)
        
        # YÊU CẦU 3.4: Heatmap
        if not corr_matrix.empty:
            st.subheader("🔥 4. Heatmap Tương Quan")
            fig_heatmap = visualizer.create_heatmap(corr_matrix)
            st.pyplot(fig_heatmap)
        
        # YÊU CẦU 3.5: Treemap
        if not portfolio_df.empty:
            st.subheader("🌳 5. Treemap Portfolio")
            fig_treemap = visualizer.create_treemap(portfolio_df)
            st.plotly_chart(fig_treemap, use_container_width=True)
        
        # YÊU CẦU 3.6: WordCloud
        if not portfolio_df.empty:
            st.subheader("☁️ 6. WordCloud Thị Trường")
            fig_wordcloud = visualizer.create_wordcloud(portfolio_df)
            st.pyplot(fig_wordcloud)
    
    with tab3:
        st.header("🎨 BIỂU ĐỒ TƯƠNG TÁC - YÊU CẦU 4")
        
        # YÊU CẦU 4: 3 biểu đồ tương tác
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 1. Line Chart Tương Tác")
            fig_interactive_line = visualizer.create_interactive_line(data)
            st.plotly_chart(fig_interactive_line, use_container_width=True)
            
            st.subheader("🕯️ 2. Candlestick Tương Tác")
            fig_candlestick = visualizer.create_interactive_candlestick(data)
            st.plotly_chart(fig_candlestick, use_container_width=True)
        
        with col2:
            st.subheader("🌟 3. 3D Scatter Tương Tác")
            if not portfolio_df.empty:
                fig_3d_scatter = visualizer.create_interactive_3d_scatter(portfolio_df)
                st.plotly_chart(fig_3d_scatter, use_container_width=True)
            else:
                st.info("📊 Bật phân tích portfolio để xem 3D scatter")
            
            st.subheader("📊 4. Dashboard Kỹ Thuật")
            fig_technical = visualizer.create_technical_analysis_dashboard(data)
            st.plotly_chart(fig_technical, use_container_width=True)
    
    with tab4:
        st.header("🔍 PHÂN TÍCH KỸ THUẬT")
        
        if enable_technical:
            # Chỉ số kỹ thuật
            st.subheader("📈 Chỉ Số Kỹ Thuật")
            
            # Lọc các chỉ số quan trọng
            tech_cols = ['open_time', 'SMA_7', 'SMA_25', 'RSI', 'MACD', 'MACD_signal', 
                        'BB_upper', 'BB_lower', 'volatility_7d', 'volume_ratio', 'combined_signal']
            
            available_tech_cols = [col for col in tech_cols if col in data.columns]
            tech_data = data[available_tech_cols].tail(20)
            
            st.dataframe(tech_data.style.format({
                'SMA_7': '{:.2f}', 'SMA_25': '{:.2f}', 'RSI': '{:.1f}',
                'MACD': '{:.4f}', 'MACD_signal': '{:.4f}', 
                'BB_upper': '{:.2f}', 'BB_lower': '{:.2f}',
                'volatility_7d': '{:.2f}%', 'volume_ratio': '{:.2f}'
            }), use_container_width=True)
            
            # Correlation analysis
            st.subheader("🔗 Phân Tích Tương Quan")
            numeric_cols = ['close', 'volume', 'price_change_pct', 'RSI', 'MACD', 'volatility_7d']
            available_numeric = [col for col in numeric_cols if col in data.columns]
            
            if available_numeric:
                correlation_analysis = data[available_numeric].corr()
                fig_corr = px.imshow(correlation_analysis, 
                                   color_continuous_scale='RdBu_r',
                                   aspect='auto',
                                   title='Heatmap Tương Quan Các Chỉ Số')
                st.plotly_chart(fig_corr, use_container_width=True)
        
        else:
            st.info("🔧 Bật 'Phân tích kỹ thuật' trong sidebar để xem nội dung này")
    
    with tab5:
        st.header("🔮 DỰ ĐOÁN GIÁ")
        
        if enable_prediction:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🤖 Huấn luyện Model")
                model_type = st.selectbox("Chọn model:", ['random_forest', 'linear_regression'])
                lookback = st.slider("Lookback periods:", 5, 30, 10)
                
                if st.button("🎯 Train Model", type="primary"):
                    with st.spinner("Đang huấn luyện model..."):
                        success = predictor.train(data, model_type)
                        if success:
                            st.success("✅ Model đã được huấn luyện!")
                            
                            # Hiển thị feature importance
                            if predictor.feature_importance is not None:
                                st.subheader("📊 Feature Importance")
                                fig_importance = px.bar(
                                    predictor.feature_importance.head(10),
                                    x='importance',
                                    y='feature',
                                    orientation='h',
                                    title='Top 10 Features Quan Trọng Nhất'
                                )
                                st.plotly_chart(fig_importance, use_container_width=True)
            
            with col2:
                st.subheader("🔮 Dự Đoán Tương Lai")
                future_periods = st.slider("Số periods dự đoán:", 1, 10, 5)
                
                if st.button("🌠 Dự Đoán Giá"):
                    if predictor.model is not None:
                        predictions = predictor.predict(data, future_periods)
                        
                        if predictions:
                            # Tạo timeline cho predictions
                            last_date = data['open_time'].iloc[-1]
                            future_dates = [last_date + timedelta(days=i+1) for i in range(future_periods)]
                            
                            # Tạo chart
                            fig_pred = go.Figure()
                            
                            # Historical data
                            fig_pred.add_trace(go.Scatter(
                                x=data['open_time'],
                                y=data['close'],
                                mode='lines',
                                name='Historical',
                                line=dict(color='blue')
                            ))
                            
                            # Predictions
                            fig_pred.add_trace(go.Scatter(
                                x=future_dates,
                                y=predictions,
                                mode='lines+markers',
                                name='Predictions',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            fig_pred.update_layout(
                                title=f'Dự Đoán Giá {selected_symbol}',
                                xaxis_title='Thời Gian',
                                yaxis_title='Giá (USDT)'
                            )
                            
                            st.plotly_chart(fig_pred, use_container_width=True)
                            
                            # Hiển thị bảng predictions
                            pred_df = pd.DataFrame({
                                'Ngày': future_dates,
                                'Giá Dự Đoán (USDT)': predictions
                            })
                            st.dataframe(pred_df.style.format({'Giá Dự Đoán (USDT)': '{:.2f}'}))
                    
                    else:
                        st.warning("⚠️ Vui lòng train model trước khi dự đoán")
            
            # Model evaluation
            if predictor.model is not None:
                st.subheader("📊 Đánh Giá Model")
                evaluation = predictor.evaluate(data)
                
                if evaluation:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("MAE", f"{evaluation['MAE']:.4f}")
                    with col2:
                        st.metric("RMSE", f"{evaluation['RMSE']:.4f}")
                    with col3:
                        st.metric("Độ Chính Xác (2%)", f"{evaluation['Accuracy_2%']:.1f}%")
        
        else:
            st.info("🔧 Bật 'Dự đoán giá' trong sidebar để xem nội dung này")
    
    with tab6:
        st.header("📖 STORYTELLING - YÊU CẦU 5")
        
        # YÊU CẦU 5: Storytelling tự động
        storytelling_content = f"""
        # 📊 BÁO CÁO PHÂN TÍCH {selected_symbol}
        
        **Thời gian tạo:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}  
        **Symbol:** {selected_symbol}  
        **Khung thời gian:** {timeframe}  
        **Phạm vi dữ liệu:** {len(data)} nến ({data['open_time'].min().strftime('%d/%m/%Y')} - {data['open_time'].max().strftime('%d/%m/%Y')})
        
        ## 🎯 TÓM TẮT EXECUTIVE
        
        ### 📈 TÌNH HÌNH HIỆN TẠI
        - **Giá hiện tại:** ${data['close'].iloc[-1]:.2f}
        - **Biến động 24h:** {data['price_change_pct'].iloc[-1]:+.2f}%
        - **Xu hướng chính:** {'🟢 BULLISH' if data['close'].iloc[-1] > data['close'].iloc[0] else '🔴 BEARISH'}
        - **Khối lượng giao dịch:** {data['volume'].iloc[-1]:,.0f}
        
        ### ⚠️ CẢNH BÁO RỦI RO
        - **Độ biến động:** {data['volatility_7d'].iloc[-1] if 'volatility_7d' in data.columns else 0:.2f}%
        - **Mức RSI:** {data['RSI'].iloc[-1] if 'RSI' in data.columns else 0:.1f} - {'🔴 QUÁ MUA' if data['RSI'].iloc[-1] > 70 else '🟢 QUÁ BÁN' if data['RSI'].iloc[-1] < 30 else '🟡 TRUNG LẬP'}
        - **VaR 95% (1 ngày):** {data['daily_var_95'].iloc[-1] if 'daily_var_95' in data.columns else 0:.2f}%
        
        ## 📊 PHÂN TÍCH KỸ THUẬT CHI TIẾT
        
        ### 🎯 TÍN HIỆU GIAO DỊCH
        """
        
        # Thêm tín hiệu giao dịch
        if 'combined_signal' in data.columns:
            signal = data['combined_signal'].iloc[-1]
            if signal > 2:
                storytelling_content += "**🔴 TÍN HIỆU: BÁN MẠNH** - Hầu hết các indicator cho thấy overbought\n"
            elif signal > 1:
                storytelling_content += "**🟡 TÍN HIỆU: BÁN NHẸ** - Có dấu hiệu overbought\n"
            elif signal < -2:
                storytelling_content += "**🟢 TÍN HIỆU: MUA MẠNH** - Hầu hết các indicator cho thấy oversold\n"
            elif signal < -1:
                storytelling_content += "**🟢 TÍN HIỆU: MUA NHẸ** - Có dấu hiệu oversold\n"
            else:
                storytelling_content += "**⚪ TÍN HIỆU: TRUNG LẬP** - Không có tín hiệu rõ ràng\n"
        
        # Thêm phân tích indicators
        storytelling_content += f"""
        ### 📈 CHỈ SỐ KỸ THUẬT QUAN TRỌNG
        
        **Moving Averages:**
        - MA7: ${data['SMA_7'].iloc[-1] if 'SMA_7' in data.columns else 0:.2f}
        - MA25: ${data['SMA_25'].iloc[-1] if 'SMA_25' in data.columns else 0:.2f}
        - **Tín hiệu:** {'🟢 BULLISH' if data['SMA_7'].iloc[-1] > data['SMA_25'].iloc[-1] else '🔴 BEARISH'}
        
        **Momentum:**
        - RSI: {data['RSI'].iloc[-1] if 'RSI' in data.columns else 0:.1f}
        - MACD: {data['MACD'].iloc[-1] if 'MACD' in data.columns else 0:.4f}
        - **Phân tích:** {'Momentum tăng' if data['RSI'].iloc[-1] > 50 else 'Momentum giảm'}
        
        **Khối lượng:**
        - Volume hiện tại: {data['volume'].iloc[-1]:,.0f}
        - Volume trung bình: {data['volume'].mean():,.0f}
        - **Nhận định:** {'🟢 Cao hơn trung bình' if data['volume'].iloc[-1] > data['volume'].mean() else '🔴 Thấp hơn trung bình'}
        
        ## 💡 KHUYẾN NGHỊ CHIẾN LƯỢC
        
        ### 🎯 NGẮN HẠN (1-7 ngày)
        """
        
        # Khuyến nghị ngắn hạn
        if 'RSI' in data.columns and 'volatility_7d' in data.columns:
            if data['RSI'].iloc[-1] > 70 and data['volatility_7d'].iloc[-1] > 5:
                storytelling_content += "**🔴 BÁN** - Thị trường overbought và biến động cao\n"
            elif data['RSI'].iloc[-1] < 30 and data['volatility_7d'].iloc[-1] < 3:
                storytelling_content += "**🟢 MUA** - Thị trường oversold và ổn định\n"
            else:
                storytelling_content += "**🟡 GIỮ** - Chờ tín hiệu rõ ràng hơn\n"
        
        storytelling_content += f"""
        ### 📅 TRUNG HẠN (1-4 tuần)
        - **Mục tiêu giá:** ${data['resistance_20'].iloc[-1] if 'resistance_20' in data.columns else data['close'].iloc[-1] * 1.1:.2f}
        - **Điểm dừng lỗ:** ${data['support_20'].iloc[-1] if 'support_20' in data.columns else data['close'].iloc[-1] * 0.9:.2f}
        - **Tỷ lệ R:R:** {(data['resistance_20'].iloc[-1] - data['close'].iloc[-1]) / (data['close'].iloc[-1] - data['support_20'].iloc[-1]) if 'resistance_20' in data.columns and 'support_20' in data.columns else 1:.2f}:1
        
        ## 🎯 KẾT LUẬN
        
        Thị trường {selected_symbol} đang trong xu hướng **{'TĂNG' if data['close'].iloc[-1] > data['close'].mean() else 'GIẢM'}** 
        với mức biến động **{'CAO' if data['volatility_7d'].iloc[-1] > 5 else 'TRUNG BÌNH' if data['volatility_7d'].iloc[-1] > 2 else 'THẤP'}**.
        
        **Khuyến nghị tổng thể:** {'NÊN MUA' if data['RSI'].iloc[-1] < 40 and data['close'].iloc[-1] > data['SMA_25'].iloc[-1] else 'NÊN BÁN' if data['RSI'].iloc[-1] > 60 else 'THEO DÕI THÊM'}
        
        *Lưu ý: Đây là phân tích tự động, nhà đầu tư nên cân nhắc kỹ trước khi ra quyết định.*
        """
        
        # Hiển thị storytelling
        st.markdown(storytelling_content)
        
        # Nút export
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("📥 Export Báo Cáo PDF"):
                # Trong thực tế, bạn có thể sử dụng libraries như weasyprint
                st.success("✅ Đã xuất báo cáo! (Tính năng PDF đang phát triển)")
        
        with col2:
            if st.button("🔄 Cập Nhật Báo Cáo"):
                st.rerun()

else:
    # Welcome screen
    st.info("👈 **Vui lòng chọn tham số và nhấn 'TẢI DỮ LIỆU' để bắt đầu phân tích**")
    
    # Hiển thị hướng dẫn
    with st.expander("📖 HƯỚNG DẪN SỬ DỤNG"):
        st.markdown("""
        ## 🚀 Hướng Dẫn Sử Dụng Ứng Dụng
        
        ### 1. 🎯 CÀI ĐẶT PHÂN TÍCH
        - **Chọn Coin:** Lựa chọn cryptocurrency muốn phân tích
        - **Khung thời gian:** Chọn khung thời gian phù hợp (1D, 4H, 1H, 15M)
        - **Số lượng nến:** Số lượng dữ liệu historical cần phân tích
        
        ### 2. 🔧 CÔNG CỤ NÂNG CAO
        - **Dự đoán giá:** Sử dụng machine learning để dự đoán xu hướng giá
        - **Phân tích kỹ thuật:** Hiển thị các chỉ số kỹ thuật nâng cao
        - **Phân tích portfolio:** So sánh nhiều coin cùng lúc
        
        ### 3. 📊 CÁC TÍNH NĂNG CHÍNH
        - **Tổng quan:** Metrics và biểu đồ giá cơ bản
        - **Biểu đồ Static:** 6 loại biểu đồ theo yêu cầu đề bài
        - **Biểu đồ Tương tác:** 4 biểu đồ Plotly interactive
        - **Phân tích Kỹ thuật:** Các indicator và tín hiệu giao dịch
        - **Dự đoán:** Machine learning price prediction
        - **Storytelling:** Báo cáo tự động với insights
        
        ### 4. 💡 MẸO SỬ DỤNG
        - Luôn bắt đầu với việc tải dữ liệu mới nhất
        - Sử dụng caching để tải nhanh hơn
        - Export báo cáo để lưu trữ kết quả phân tích
        """)

# Footer
st.markdown("---")
st.markdown(
    "**Bài tập giữa kỲ - Môn Khai phá dữ liệu** • "
    "**Nguyễn Mạnh Dũng - B22DCCN132** • "
    "**PTIT - 2024**"
)