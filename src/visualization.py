# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from wordcloud import WordCloud
import folium
from folium import plugins
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

class CryptoVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
        self.plotly_template = 'plotly_white'
    
    # YÊU CẦU 3: STATIC VISUALIZATIONS
    def create_histogram_boxplot(self, df, column='price_change_pct', title="Phân Phối Biến Động Giá"):
        """YÊU CẦU 3.1: Histogram và Boxplot nâng cao"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        data = df[column].dropna()
        
        # Histogram với KDE
        ax1.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        data.plot.kde(ax=ax1, color='red', linewidth=2)
        ax1.set_title(f'Histogram với KDE - {title}', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Biến Động (%)')
        ax1.set_ylabel('Mật Độ')
        ax1.grid(True, alpha=0.3)
        ax1.legend(['KDE', 'Histogram'])
        
        # Boxplot
        box_plot = ax2.boxplot(data, patch_artist=True, vert=True)
        box_plot['boxes'][0].set_facecolor('lightgreen')
        ax2.set_title(f'Boxplot - {title}', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Biến Động (%)')
        ax2.grid(True, alpha=0.3)
        
        # Violin plot
        ax3.violinplot(data, showmeans=True, showmedians=True)
        ax3.set_title(f'Violin Plot - {title}', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Biến Động (%)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_line_area_chart(self, df, time_col='open_time', value_col='close', 
                             title="Biểu Đồ Giá Theo Thời Gian", ma_periods=[7, 25]):
        """YÊU CẦU 3.2: Line và Area chart nâng cao"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Line chart với Moving Averages
        ax1.plot(df[time_col], df[value_col], color='#FF6B6B', linewidth=2, label='Giá Đóng')
        
        # Thêm các đường MA
        colors = ['#4ECDC4', '#45B7D1', '#96CEB4']
        for i, period in enumerate(ma_periods):
            ma_col = f'SMA_{period}'
            if ma_col in df.columns:
                ax1.plot(df[time_col], df[ma_col], color=colors[i], linewidth=1.5, 
                        label=f'MA {period}', linestyle='--')
        
        ax1.set_title(f'Line Chart với MA - {title}', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Giá (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Area chart với volume
        ax2.fill_between(df[time_col], df[value_col], alpha=0.3, color='#4ECDC4')
        ax2.plot(df[time_col], df[value_col], color='#4ECDC4', linewidth=1, label='Giá')
        
        # Thêm volume như bar chart
        ax2_vol = ax2.twinx()
        ax2_vol.bar(df[time_col], df['volume'] if 'volume' in df.columns else 0, 
                   alpha=0.2, color='gray', label='Volume')
        ax2_vol.set_ylabel('Volume')
        
        ax2.set_title(f'Area Chart với Volume - {title}', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Thời Gian')
        ax2.set_ylabel('Giá (USDT)')
        ax2.legend(loc='upper left')
        ax2_vol.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_scatter_regression(self, df, x_col='volume', y_col='price_change_pct', 
                                hue_col='RSI', title="Phân Tán & Hồi Quy"):
        """YÊU CẦU 3.3: Scatter plot với hồi quy nâng cao"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot với màu sắc
        scatter = ax1.scatter(df[x_col], df[y_col], 
                             c=df[hue_col] if hue_col in df.columns else 'blue', 
                             cmap='viridis', alpha=0.6, s=50)
        ax1.set_xlabel(x_col.replace('_', ' ').title())
        ax1.set_ylabel(y_col.replace('_', ' ').title())
        ax1.set_title(f'Scatter Plot - {title}', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Regression line
        if len(df) > 1:
            valid_data = df[[x_col, y_col]].dropna()
            if len(valid_data) > 1:
                z = np.polyfit(valid_data[x_col], valid_data[y_col], 1)
                p = np.poly1d(z)
                ax1.plot(valid_data[x_col], p(valid_data[x_col]), "r--", alpha=0.8, linewidth=2, label='Hồi quy')
                ax1.legend()
        
        plt.colorbar(scatter, ax=ax1, label=hue_col)
        
        # Hexbin plot
        hb = ax2.hexbin(df[x_col], df[y_col], gridsize=30, cmap='Blues', alpha=0.7)
        ax2.set_xlabel(x_col.replace('_', ' ').title())
        ax2.set_ylabel(y_col.replace('_', ' ').title())
        ax2.set_title(f'Hexbin Plot - {title}', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(hb, ax=ax2, label='Mật độ điểm')
        
        return fig
    
    def create_heatmap(self, correlation_matrix, title="Heatmap Tương Quan"):
        """YÊU CẦU 3.4: Heatmap tương quan nâng cao"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Correlation heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=ax1, mask=mask)
        ax1.set_title(f'Correlation Heatmap - {title}', fontweight='bold', pad=20)
        
        # Clustered heatmap
        sns.clustermap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                      fmt='.2f', figsize=(8, 8))
        
        return fig
    
    def create_treemap(self, portfolio_df, title="Treemap Vốn Hóa Thị Trường"):
        """YÊU CẦU 3.5: Treemap nâng cao"""
        # Tính market cap giả định
        market_cap_data = portfolio_df.groupby('coin_name').agg({
            'volume': 'mean',
            'close': 'mean',
            'price_change_pct': 'mean'
        }).reset_index()
        market_cap_data['market_cap'] = market_cap_data['volume'] * market_cap_data['close']
        market_cap_data['return_color'] = market_cap_data['price_change_pct']
        
        fig = px.treemap(market_cap_data,
                        path=['coin_name'],
                        values='market_cap',
                        color='return_color',
                        color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=0,
                        title=title,
                        hover_data=['volume', 'price_change_pct'])
        
        fig.update_layout(height=600, 
                         coloraxis_colorbar=dict(title="Lợi nhuận (%)"))
        fig.update_traces(textinfo="label+value+percent parent")
        
        return fig
    
    def create_wordcloud(self, portfolio_df, title="WordCloud Thị Trường Crypto"):
        """YÊU CẦU 3.6: WordCloud nâng cao"""
        # Tạo tần suất từ với trọng số
        word_freq = {}
        for coin in portfolio_df['coin_name'].unique():
            coin_data = portfolio_df[portfolio_df['coin_name'] == coin]
            coin_volume = coin_data['volume'].mean()
            coin_return = abs(coin_data['price_change_pct'].mean())
            # Kết hợp volume và volatility
            word_freq[coin] = max(1, int(coin_volume / 1000 * (1 + coin_return/100)))
        
        wordcloud = WordCloud(width=1000, height=500,
                             background_color='black',
                             colormap='plasma',
                             max_words=50,
                             relative_scaling=0.5,
                             random_state=42).generate_from_frequencies(word_freq)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20, color='white')
        fig.patch.set_facecolor('black')
        
        return fig
    
    # YÊU CẦU 4: INTERACTIVE VISUALIZATIONS
    def create_interactive_line(self, df, time_col='open_time', value_col='close', 
                              title="Biểu Đồ Giá Tương Tác"):
        """Biểu đồ tương tác 1: Line chart nâng cao"""
        fig = go.Figure()
        
        # Đường giá chính
        fig.add_trace(go.Scatter(
            x=df[time_col],
            y=df[value_col],
            mode='lines',
            name='Giá Đóng',
            line=dict(color='#FF6B6B', width=2),
            hovertemplate='<b>Thời gian:</b> %{x}<br><b>Giá:</b> $%{y:.2f}<extra></extra>'
        ))
        
        # Thêm các đường MA nếu có
        ma_columns = [col for col in df.columns if col.startswith('SMA_') or col.startswith('EMA_')]
        colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#F7DC6F', '#DDA0DD']
        
        for i, col in enumerate(ma_columns[:5]):  # Giới hạn 5 đường
            fig.add_trace(go.Scatter(
                x=df[time_col],
                y=df[col],
                mode='lines',
                name=col,
                line=dict(color=colors[i % len(colors)], width=1, dash='dash'),
                opacity=0.7
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            xaxis_title='Thời Gian',
            yaxis_title='Giá (USDT)',
            template=self.plotly_template,
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_interactive_candlestick(self, df, title="Biểu Đồ Nến Tương Tác"):
        """Biểu đồ tương tác 2: Candlestick nâng cao"""
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df['open_time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Giá',
            increasing_line_color='#2E8B57',
            decreasing_line_color='#DC143C'
        ))
        
        # Thêm Bollinger Bands nếu có
        if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['open_time'],
                y=df['BB_upper'],
                line=dict(color='rgba(255, 107, 107, 0.5)', width=1),
                name='BB Upper',
                opacity=0.7
            ))
            
            fig.add_trace(go.Scatter(
                x=df['open_time'],
                y=df['BB_lower'],
                line=dict(color='rgba(255, 107, 107, 0.5)', width=1),
                name='BB Lower',
                fill='tonexty',
                opacity=0.3
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            xaxis_title='Thời Gian',
            yaxis_title='Giá (USDT)',
            template=self.plotly_template,
            height=600,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def create_interactive_3d_scatter(self, portfolio_df, title="3D Scatter Tương Tác"):
        """Biểu đồ tương tác 3: 3D Scatter nâng cao"""
        # Chuẩn bị dữ liệu mới nhất
        latest_data = portfolio_df.sort_values('open_time').groupby('coin_name').last().reset_index()
        
        fig = px.scatter_3d(latest_data,
                           x='volume',
                           y='price_change_pct', 
                           z='RSI',
                           color='coin_name',
                           size='close',
                           hover_name='coin_name',
                           title=title,
                           labels={
                               'volume': 'Khối Lượng',
                               'price_change_pct': 'Biến Động (%)',
                               'RSI': 'RSI',
                               'close': 'Giá'
                           },
                           size_max=30,
                           opacity=0.8)
        
        fig.update_layout(
            height=700,
            scene=dict(
                xaxis_title='Volume (Tỷ lệ)',
                yaxis_title='Biến Động 24h (%)',
                zaxis_title='RSI'
            )
        )
        
        return fig
    
    def create_technical_analysis_dashboard(self, df, title="Dashboard Phân Tích Kỹ Thuật"):
        """Dashboard phân tích kỹ thuật tương tác"""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Biểu Đồ Giá với MA', 'RSI', 'MACD', 'Volume'),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # 1. Price with MA
        fig.add_trace(go.Candlestick(
            x=df['open_time'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name='Price'
        ), row=1, col=1)
        
        if 'SMA_7' in df.columns:
            fig.add_trace(go.Scatter(x=df['open_time'], y=df['SMA_7'], 
                                   name='MA 7', line=dict(color='orange')), row=1, col=1)
        if 'SMA_25' in df.columns:
            fig.add_trace(go.Scatter(x=df['open_time'], y=df['SMA_25'], 
                                   name='MA 25', line=dict(color='blue')), row=1, col=1)
        
        # 2. RSI
        if 'RSI' in df.columns:
            fig.add_trace(go.Scatter(x=df['open_time'], y=df['RSI'], 
                                   name='RSI', line=dict(color='purple')), row=2, col=1)
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # 3. MACD
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            fig.add_trace(go.Scatter(x=df['open_time'], y=df['MACD'], 
                                   name='MACD', line=dict(color='blue')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['open_time'], y=df['MACD_signal'], 
                                   name='Signal', line=dict(color='red')), row=3, col=1)
        
        # 4. Volume
        if 'volume' in df.columns:
            colors = ['red' if row['close'] < row['open'] else 'green' 
                     for _, row in df.iterrows()]
            fig.add_trace(go.Bar(x=df['open_time'], y=df['volume'], 
                               name='Volume', marker_color=colors), row=4, col=1)
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig