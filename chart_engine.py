import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.transforms as transforms
import io
import base64
from scipy import stats

class ChartAnnotation(Enum):
    ORDER_BLOCK = "order_block"
    FAIR_VALUE_GAP = "fair_value_gap"
    BREAK_OF_STRUCTURE = "break_of_structure"
    CHANGE_OF_CHARACTER = "change_of_character"
    LIQUIDITY_ZONE = "liquidity_zone"
    FIBONACCI_LEVEL = "fibonacci_level"
    WYCKOFF_PHASE = "wyckoff_phase"
    SUPPLY_DEMAND_ZONE = "supply_demand_zone"

class MarketStructure(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"

@dataclass
class SMCAnnotation:
    annotation_type: ChartAnnotation
    symbol: str
    timeframe: str
    timestamp: datetime
    price_levels: List[float]
    confidence: float
    metadata: Dict[str, Any]
    visual_style: Dict[str, Any]

@dataclass
class FibonacciLevels:
    level_236: float
    level_382: float  
    level_500: float
    level_618: float
    level_786: float
    level_886: float
    extension_127: float
    extension_161: float
    extension_261: float

@dataclass
class WyckoffPhase:
    phase: str  # accumulation, markup, distribution, markdown
    start_time: datetime
    end_time: datetime
    price_range: Tuple[float, float]
    volume_profile: Dict[str, float]
    confirmation_signals: List[str]

class ProfessionalChartEngine:
    def __init__(self):
        self.annotation_styles = self.setup_annotation_styles()
        self.color_schemes = self.setup_color_schemes()
        self.chart_templates = self.setup_chart_templates()
        logging.info("Professional Chart Engine initialized")
    
    def setup_annotation_styles(self) -> Dict[ChartAnnotation, Dict]:
        """Setup professional annotation styles"""
        return {
            ChartAnnotation.ORDER_BLOCK: {
                'color': '#FF6B6B',
                'alpha': 0.3,
                'linewidth': 2,
                'linestyle': '--',
                'label': 'Order Block'
            },
            ChartAnnotation.FAIR_VALUE_GAP: {
                'color': '#4ECDC4',
                'alpha': 0.4,
                'linewidth': 1,
                'linestyle': '-',
                'label': 'FVG'
            },
            ChartAnnotation.BREAK_OF_STRUCTURE: {
                'color': '#45B7D1',
                'alpha': 0.6,
                'linewidth': 3,
                'linestyle': '-',
                'label': 'BOS'
            },
            ChartAnnotation.CHANGE_OF_CHARACTER: {
                'color': '#96CEB4',
                'alpha': 0.5,
                'linewidth': 2,
                'linestyle': '-.',
                'label': 'CHOCH'
            },
            ChartAnnotation.LIQUIDITY_ZONE: {
                'color': '#FECA57',
                'alpha': 0.3,
                'linewidth': 2,
                'linestyle': ':',
                'label': 'Liquidity'
            },
            ChartAnnotation.FIBONACCI_LEVEL: {
                'color': '#FF9FF3',
                'alpha': 0.4,
                'linewidth': 1,
                'linestyle': '--',
                'label': 'Fib'
            },
            ChartAnnotation.WYCKOFF_PHASE: {
                'color': '#54A0FF',
                'alpha': 0.2,
                'linewidth': 2,
                'linestyle': '-',
                'label': 'Wyckoff'
            },
            ChartAnnotation.SUPPLY_DEMAND_ZONE: {
                'color': '#5F27CD',
                'alpha': 0.3,
                'linewidth': 2,
                'linestyle': '-',
                'label': 'S/D Zone'
            }
        }
    
    def setup_color_schemes(self) -> Dict[str, Dict]:
        """Setup professional color schemes"""
        return {
            'institutional_blue': {
                'background': '#0f172a',
                'grid': '#1e293b',
                'text': '#f8fafc',
                'primary': '#1e40af',
                'secondary': '#3b82f6',
                'success': '#10b981',
                'warning': '#f59e0b',
                'danger': '#ef4444'
            },
            'professional_dark': {
                'background': '#1a1b26',
                'grid': '#24283b',
                'text': '#c0caf5',
                'primary': '#7aa2f7',
                'secondary': '#bb9af7',
                'success': '#9ece6a',
                'warning': '#e0af68',
                'danger': '#f7768e'
            },
            'trading_classic': {
                'background': '#2d2d2d',
                'grid': '#3d3d3d',
                'text': '#e0e0e0',
                'primary': '#ff6b6b',
                'secondary': '#4ecdc4',
                'success': '#45B7D1',
                'warning': '#FECA57',
                'danger': '#ff9ff3'
            }
        }
    
    def setup_chart_templates(self) -> Dict[str, Dict]:
        """Setup professional chart templates"""
        return {
            'institutional': {
                'figsize': (16, 10),
                'dpi': 150,
                'style': 'dark_background',
                'font_family': 'DejaVu Sans',
                'font_size': 10
            },
            'mobile_optimized': {
                'figsize': (12, 8),
                'dpi': 120,
                'style': 'dark_background',
                'font_family': 'DejaVu Sans',
                'font_size': 8
            },
            'presentation': {
                'figsize': (20, 12),
                'dpi': 200,
                'style': 'dark_background',
                'font_family': 'DejaVu Sans',
                'font_size': 12
            }
        }
    
    def create_professional_chart(self, price_data: pd.DataFrame, 
                                annotations: List[SMCAnnotation],
                                template: str = 'institutional',
                                color_scheme: str = 'institutional_blue') -> str:
        """Create professional trading chart with annotations"""
        try:
            # Setup chart style
            plt.style.use('dark_background')
            template_config = self.chart_templates[template]
            colors = self.color_schemes[color_scheme]
            
            # Create figure and axis
            fig, (price_ax, volume_ax) = plt.subplots(
                2, 1, 
                figsize=template_config['figsize'],
                gridspec_kw={'height_ratios': [3, 1]},
                dpi=template_config['dpi']
            )
            
            # Plot price data
            self.plot_candlestick(price_ax, price_data, colors)
            
            # Plot volume
            self.plot_volume(volume_ax, price_data, colors)
            
            # Add annotations
            for annotation in annotations:
                self.add_annotation(price_ax, annotation, colors)
            
            # Add professional elements
            self.add_chart_elements(price_ax, volume_ax, price_data, colors)
            
            # Convert to base64 for web display
            chart_base64 = self.fig_to_base64(fig)
            plt.close(fig)
            
            return chart_base64
            
        except Exception as e:
            logging.error(f"Error creating professional chart: {e}")
            return None
    
    def plot_candlestick(self, ax, price_data: pd.DataFrame, colors: Dict):
        """Plot professional candlestick chart"""
        # Calculate OHLC data
        opens = price_data['open'].values
        highs = price_data['high'].values
        lows = price_data['low'].values
        closes = price_data['close'].values
        dates = price_data.index
        
        # Define colors for bullish/bearish candles
        bull_color = colors['success']
        bear_color = colors['danger']
        
        # Plot candlesticks
        for i, (date, open_val, high, low, close) in enumerate(zip(dates, opens, highs, lows, closes)):
            color = bull_color if close >= open_val else bear_color
            
            # Draw candle body
            body_bottom = min(open_val, close)
            body_top = max(open_val, close)
            body_height = body_top - body_bottom
            
            if body_height > 0:
                rect = Rectangle(
                    (i - 0.3, body_bottom), 0.6, body_height,
                    facecolor=color, edgecolor=color, alpha=0.8
                )
                ax.add_patch(rect)
            
            # Draw wicks
            ax.plot([i, i], [low, body_bottom], color=color, linewidth=1, alpha=0.8)
            ax.plot([i, i], [body_top, high], color=color, linewidth=1, alpha=0.8)
        
        # Set labels and formatting
        ax.set_ylabel('Price', fontsize=12, color=colors['text'])
        ax.grid(True, alpha=0.3, color=colors['grid'])
        ax.set_facecolor(colors['background'])
        
        # Format x-axis with dates
        self.format_xaxis_dates(ax, dates)
    
    def plot_volume(self, ax, price_data: pd.DataFrame, colors: Dict):
        """Plot volume bars"""
        if 'volume' in price_data.columns:
            volumes = price_data['volume'].values
            dates = price_data.index
            
            # Color volume bars based on price movement
            colors_list = []
            for i in range(len(price_data)):
                if i == 0:
                    colors_list.append(colors['secondary'])
                else:
                    if price_data['close'].iloc[i] >= price_data['open'].iloc[i]:
                        colors_list.append(colors['success'])
                    else:
                        colors_list.append(colors['danger'])
            
            ax.bar(range(len(volumes)), volumes, color=colors_list, alpha=0.7)
            ax.set_ylabel('Volume', fontsize=10, color=colors['text'])
            ax.set_facecolor(colors['background'])
            ax.grid(True, alpha=0.2, color=colors['grid'])
    
    def add_annotation(self, ax, annotation: SMCAnnotation, colors: Dict):
        """Add SMC annotation to chart"""
        style = self.annotation_styles[annotation.annotation_type]
        
        if annotation.annotation_type == ChartAnnotation.ORDER_BLOCK:
            self.add_order_block_annotation(ax, annotation, style, colors)
        elif annotation.annotation_type == ChartAnnotation.FAIR_VALUE_GAP:
            self.add_fvg_annotation(ax, annotation, style, colors)
        elif annotation.annotation_type == ChartAnnotation.BREAK_OF_STRUCTURE:
            self.add_bos_annotation(ax, annotation, style, colors)
        elif annotation.annotation_type == ChartAnnotation.CHANGE_OF_CHARACTER:
            self.add_choch_annotation(ax, annotation, style, colors)
        elif annotation.annotation_type == ChartAnnotation.LIQUIDITY_ZONE:
            self.add_liquidity_annotation(ax, annotation, style, colors)
        elif annotation.annotation_type == ChartAnnotation.FIBONACCI_LEVEL:
            self.add_fibonacci_annotation(ax, annotation, style, colors)
        elif annotation.annotation_type == ChartAnnotation.WYCKOFF_PHASE:
            self.add_wyckoff_annotation(ax, annotation, style, colors)
        elif annotation.annotation_type == ChartAnnotation.SUPPLY_DEMAND_ZONE:
            self.add_supply_demand_annotation(ax, annotation, style, colors)
    
    def add_order_block_annotation(self, ax, annotation: SMCAnnotation, style: Dict, colors: Dict):
        """Add Order Block annotation"""
        price_levels = annotation.price_levels
        if len(price_levels) >= 2:
            # Draw rectangle for order block
            x_center = annotation.metadata.get('x_position', len(ax.get_xticks()) // 2)
            width = annotation.metadata.get('width', 5)
            
            rect = Rectangle(
                (x_center - width/2, min(price_levels)),
                width, max(price_levels) - min(price_levels),
                linewidth=style['linewidth'],
                edgecolor=style['color'],
                facecolor=style['color'],
                alpha=style['alpha'],
                linestyle=style['linestyle']
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(x_center, max(price_levels), 'OB', 
                   color=style['color'], fontsize=9, ha='center', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], 
                           edgecolor=style['color'], alpha=0.8))
    
    def add_fvg_annotation(self, ax, annotation: SMCAnnotation, style: Dict, colors: Dict):
        """Add Fair Value Gap annotation"""
        price_levels = annotation.price_levels
        if len(price_levels) >= 2:
            # Draw FVG area
            x_center = annotation.metadata.get('x_position', len(ax.get_xticks()) // 2)
            width = annotation.metadata.get('width', 3)
            
            rect = Rectangle(
                (x_center - width/2, min(price_levels)),
                width, max(price_levels) - min(price_levels),
                linewidth=style['linewidth'],
                edgecolor=style['color'],
                facecolor=style['color'],
                alpha=style['alpha'],
                linestyle=style['linestyle']
            )
            ax.add_patch(rect)
            
            # Add FVG label
            ax.text(x_center, np.mean(price_levels), 'FVG', 
                   color=style['color'], fontsize=8, ha='center', va='center',
                   rotation=90, alpha=0.8)
    
    def add_bos_annotation(self, ax, annotation: SMCAnnotation, style: Dict, colors: Dict):
        """Add Break of Structure annotation"""
        price_levels = annotation.price_levels
        if len(price_levels) >= 2:
            # Draw BOS line
            x_position = annotation.metadata.get('x_position', len(ax.get_xticks()) // 2)
            
            ax.axhline(y=price_levels[0], color=style['color'], 
                      linestyle=style['linestyle'], linewidth=style['linewidth'],
                      alpha=style['alpha'])
            
            # Add arrow indicating direction
            direction = annotation.metadata.get('direction', 'up')
            arrow_y = price_levels[0]
            arrow_dy = 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            
            if direction == 'up':
                ax.annotate('', xy=(x_position, arrow_y + arrow_dy),
                           xytext=(x_position, arrow_y),
                           arrowprops=dict(arrowstyle='->', color=style['color'], lw=2))
            else:
                ax.annotate('', xy=(x_position, arrow_y - arrow_dy),
                           xytext=(x_position, arrow_y),
                           arrowprops=dict(arrowstyle='->', color=style['color'], lw=2))
            
            # Add BOS label
            ax.text(x_position, price_levels[0], 'BOS', 
                   color=style['color'], fontsize=9, ha='center', va='bottom' if direction == 'up' else 'top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], 
                           edgecolor=style['color'], alpha=0.8))
    
    def add_choch_annotation(self, ax, annotation: SMCAnnotation, style: Dict, colors: Dict):
        """Add Change of Character annotation"""
        price_levels = annotation.price_levels
        if len(price_levels) >= 2:
            # Draw CHOCH line
            x_position = annotation.metadata.get('x_position', len(ax.get_xticks()) // 2)
            
            ax.axhline(y=price_levels[0], color=style['color'], 
                      linestyle=style['linestyle'], linewidth=style['linewidth'],
                      alpha=style['alpha'])
            
            # Add CHOCH label
            ax.text(x_position, price_levels[0], 'CHOCH', 
                   color=style['color'], fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], 
                           edgecolor=style['color'], alpha=0.8))
    
    def add_liquidity_annotation(self, ax, annotation: SMCAnnotation, style: Dict, colors: Dict):
        """Add Liquidity Zone annotation"""
        price_levels = annotation.price_levels
        if len(price_levels) >= 2:
            # Draw liquidity zone
            x_center = annotation.metadata.get('x_position', len(ax.get_xticks()) // 2)
            width = annotation.metadata.get('width', 10)
            
            rect = Rectangle(
                (x_center - width/2, min(price_levels)),
                width, max(price_levels) - min(price_levels),
                linewidth=style['linewidth'],
                edgecolor=style['color'],
                facecolor=style['color'],
                alpha=style['alpha'],
                linestyle=style['linestyle'],
                hatch='///'
            )
            ax.add_patch(rect)
            
            # Add liquidity label
            ax.text(x_center, np.mean(price_levels), 'Liquidity', 
                   color=style['color'], fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], 
                           edgecolor=style['color'], alpha=0.8))
    
    def add_fibonacci_annotation(self, ax, annotation: SMCAnnotation, style: Dict, colors: Dict):
        """Add Fibonacci levels annotation"""
        price_levels = annotation.price_levels
        if len(price_levels) >= 2:
            # Draw Fibonacci levels
            fib_levels = annotation.metadata.get('fib_levels', {})
            fib_labels = ['0.236', '0.382', '0.5', '0.618', '0.786', '0.886']
            
            for i, level in enumerate(fib_levels.values()):
                if i < len(fib_labels):
                    ax.axhline(y=level, color=style['color'], 
                              linestyle=style['linestyle'], linewidth=style['linewidth'],
                              alpha=style['alpha'] * 0.7)
                    
                    # Add Fibonacci labels
                    ax.text(ax.get_xlim()[1] * 0.98, level, fib_labels[i],
                           color=style['color'], fontsize=7, ha='right', va='center',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=colors['background'], 
                                   edgecolor=style['color'], alpha=0.6))
    
    def add_wyckoff_annotation(self, ax, annotation: SMCAnnotation, style: Dict, colors: Dict):
        """Add Wyckoff phase annotation"""
        price_levels = annotation.price_levels
        if len(price_levels) >= 2:
            # Draw Wyckoff phase area
            x_center = annotation.metadata.get('x_position', len(ax.get_xticks()) // 2)
            width = annotation.metadata.get('width', 15)
            
            rect = Rectangle(
                (x_center - width/2, min(price_levels)),
                width, max(price_levels) - min(price_levels),
                linewidth=style['linewidth'],
                edgecolor=style['color'],
                facecolor=style['color'],
                alpha=style['alpha'],
                linestyle=style['linestyle']
            )
            ax.add_patch(rect)
            
            # Add Wyckoff label
            phase = annotation.metadata.get('phase', 'Accumulation')
            ax.text(x_center, np.mean(price_levels), f'Wyckoff: {phase}', 
                   color=style['color'], fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], 
                           edgecolor=style['color'], alpha=0.8))
    
    def add_supply_demand_annotation(self, ax, annotation: SMCAnnotation, style: Dict, colors: Dict):
        """Add Supply/Demand Zone annotation"""
        price_levels = annotation.price_levels
        if len(price_levels) >= 2:
            # Draw supply/demand zone
            x_center = annotation.metadata.get('x_position', len(ax.get_xticks()) // 2)
            width = annotation.metadata.get('width', 8)
            zone_type = annotation.metadata.get('zone_type', 'supply')
            
            rect = Rectangle(
                (x_center - width/2, min(price_levels)),
                width, max(price_levels) - min(price_levels),
                linewidth=style['linewidth'],
                edgecolor=style['color'],
                facecolor=style['color'],
                alpha=style['alpha'],
                linestyle=style['linestyle'],
                hatch='xx' if zone_type == 'supply' else 'oo'
            )
            ax.add_patch(rect)
            
            # Add zone label
            label = 'Supply' if zone_type == 'supply' else 'Demand'
            ax.text(x_center, np.mean(price_levels), label, 
                   color=style['color'], fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], 
                           edgecolor=style['color'], alpha=0.8))
    
    def add_chart_elements(self, price_ax, volume_ax, price_data: pd.DataFrame, colors: Dict):
        """Add professional chart elements"""
        # Add title
        symbol = price_data.attrs.get('symbol', 'Unknown')
        timeframe = price_data.attrs.get('timeframe', 'Unknown')
        price_ax.set_title(f'{symbol} - {timeframe}', 
                          fontsize=14, color=colors['text'], pad=20)
        
        # Add grid
        price_ax.grid(True, alpha=0.2, color=colors['grid'])
        volume_ax.grid(True, alpha=0.2, color=colors['grid'])
        
        # Add background color
        price_ax.set_facecolor(colors['background'])
        volume_ax.set_facecolor(colors['background'])
        
        # Add price statistics
        self.add_price_statistics(price_ax, price_data, colors)
    
    def add_price_statistics(self, ax, price_data: pd.DataFrame, colors: Dict):
        """Add price statistics to chart"""
        if len(price_data) > 0:
            current_price = price_data['close'].iloc[-1]
            high_24h = price_data['high'].max()
            low_24h = price_data['low'].min()
            change_24h = ((current_price - price_data['open'].iloc[0]) / price_data['open'].iloc[0]) * 100
            
            stats_text = (f"Current: {current_price:.5f}\n"
                         f"24h High: {high_24h:.5f}\n"
                         f"24h Low: {low_24h:.5f}\n"
                         f"Change: {change_24h:+.2f}%")
            
            # Add stats box
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, color=colors['text'], va='top', ha='left',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['background'], 
                           edgecolor=colors['primary'], alpha=0.8))
    
    def format_xaxis_dates(self, ax, dates):
        """Format x-axis with dates"""
        if len(dates) > 0:
            # Simple date formatting for now
            ax.set_xlim(0, len(dates))
            ax.set_xticks(np.linspace(0, len(dates)-1, min(10, len(dates))))
            
            # Format x-axis labels
            if hasattr(dates, 'strftime'):
                date_labels = [d.strftime('%m/%d %H:%M') for d in dates[::len(dates)//10]]
                ax.set_xticklabels(date_labels, rotation=45)
    
    def fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', 
                   facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return image_base64
    
    def detect_smc_patterns(self, price_data: pd.DataFrame) -> List[SMCAnnotation]:
        """Detect SMC patterns in price data"""
        annotations = []
        
        try:
            # Detect Order Blocks
            ob_annotations = self.detect_order_blocks(price_data)
            annotations.extend(ob_annotations)
            
            # Detect Fair Value Gaps
            fvg_annotations = self.detect_fair_value_gaps(price_data)
            annotations.extend(fvg_annotations)
            
            # Detect Break of Structure
            bos_annotations = self.detect_break_of_structure(price_data)
            annotations.extend(bos_annotations)
            
            # Detect Fibonacci Levels
            fib_annotations = self.detect_fibonacci_levels(price_data)
            annotations.extend(fib_annotations)
            
            # Detect Liquidity Zones
            liquidity_annotations = self.detect_liquidity_zones(price_data)
            annotations.extend(liquidity_annotations)
            
        except Exception as e:
            logging.error(f"Error detecting SMC patterns: {e}")
        
        return annotations
    
    def detect_order_blocks(self, price_data: pd.DataFrame) -> List[SMCAnnotation]:
        """Detect Order Block patterns"""
        annotations = []
        
        try:
            # Simple order block detection based on significant moves
            if len(price_data) < 20:
                return annotations
            
            # Calculate price movements
            price_data['price_change'] = price_data['close'].pct_change()
            
            # Look for significant moves followed by consolidation
            significant_moves = price_data[abs(price_data['price_change']) > 0.005]  # 0.5% moves
            
            for idx in significant_moves.index:
                move_idx = price_data.index.get_loc(idx)
                if move_idx >= 10 and move_idx < len(price_data) - 10:
                    # Check for consolidation after move
                    post_move_data = price_data.iloc[move_idx:move_idx+10]
                    consolidation_range = post_move_data['high'].max() - post_move_data['low'].min()
                    
                    if consolidation_range / price_data['close'].iloc[move_idx] < 0.002:  # Tight consolidation
                        ob_annotation = SMCAnnotation(
                            annotation_type=ChartAnnotation.ORDER_BLOCK,
                            symbol=price_data.attrs.get('symbol', 'Unknown'),
                            timeframe=price_data.attrs.get('timeframe', 'Unknown'),
                            timestamp=idx,
                            price_levels=[post_move_data['low'].min(), post_move_data['high'].max()],
                            confidence=0.7,
                            metadata={
                                'x_position': move_idx,
                                'width': 8,
                                'move_direction': 'bullish' if price_data['price_change'].iloc[move_idx] > 0 else 'bearish'
                            },
                            visual_style={}
                        )
                        annotations.append(ob_annotation)
            
        except Exception as e:
            logging.error(f"Error detecting order blocks: {e}")
        
        return annotations
    
    def detect_fair_value_gaps(self, price_data: pd.DataFrame) -> List[SMCAnnotation]:
        """Detect Fair Value Gap patterns"""
        annotations = []
        
        try:
            if len(price_data) < 3:
                return annotations
            
            # Look for three-bar patterns with gaps
            for i in range(1, len(price_data) - 1):
                prev_low = price_data['low'].iloc[i-1]
                curr_high = price_data['high'].iloc[i]
                curr_low = price_data['low'].iloc[i]
                next_high = price_data['high'].iloc[i+1]
                
                # Bullish FVG: current low > previous high
                if curr_low > price_data['high'].iloc[i-1]:
                    fvg_annotation = SMCAnnotation(
                        annotation_type=ChartAnnotation.FAIR_VALUE_GAP,
                        symbol=price_data.attrs.get('symbol', 'Unknown'),
                        timeframe=price_data.attrs.get('timeframe', 'Unknown'),
                        timestamp=price_data.index[i],
                        price_levels=[price_data['high'].iloc[i-1], curr_low],
                        confidence=0.8,
                        metadata={
                            'x_position': i,
                            'width': 3,
                            'direction': 'bullish'
                        },
                        visual_style={}
                    )
                    annotations.append(fvg_annotation)
                
                # Bearish FVG: current high < previous low
                elif curr_high < price_data['low'].iloc[i-1]:
                    fvg_annotation = SMCAnnotation(
                        annotation_type=ChartAnnotation.FAIR_VALUE_GAP,
                        symbol=price_data.attrs.get('symbol', 'Unknown'),
                        timeframe=price_data.attrs.get('timeframe', 'Unknown'),
                        timestamp=price_data.index[i],
                        price_levels=[curr_high, price_data['low'].iloc[i-1]],
                        confidence=0.8,
                        metadata={
                            'x_position': i,
                            'width': 3,
                            'direction': 'bearish'
                        },
                        visual_style={}
                    )
                    annotations.append(fvg_annotation)
            
        except Exception as e:
            logging.error(f"Error detecting fair value gaps: {e}")
        
        return annotations
    
    def detect_break_of_structure(self, price_data: pd.DataFrame) -> List[SMCAnnotation]:
        """Detect Break of Structure patterns"""
        annotations = []
        
        try:
            if len(price_data) < 10:
                return annotations
            
            # Calculate swing highs and lows
            swing_highs = self.find_swing_highs(price_data)
            swing_lows = self.find_swing_lows(price_data)
            
            # Detect bullish BOS: higher high after higher low
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                latest_high = swing_highs[-1]
                previous_high = swing_highs[-2]
                latest_low = swing_lows[-1]
                previous_low = swing_lows[-2]
                
                if latest_high > previous_high and latest_low > previous_low:
                    bos_annotation = SMCAnnotation(
                        annotation_type=ChartAnnotation.BREAK_OF_STRUCTURE,
                        symbol=price_data.attrs.get('symbol', 'Unknown'),
                        timeframe=price_data.attrs.get('timeframe', 'Unknown'),
                        timestamp=price_data.index[-1],
                        price_levels=[previous_high, latest_high],
                        confidence=0.75,
                        metadata={
                            'x_position': len(price_data) - 1,
                            'direction': 'bullish'
                        },
                        visual_style={}
                    )
                    annotations.append(bos_annotation)
            
        except Exception as e:
            logging.error(f"Error detecting break of structure: {e}")
        
        return annotations
    
    def detect_fibonacci_levels(self, price_data: pd.DataFrame) -> List[SMCAnnotation]:
        """Detect Fibonacci retracement levels"""
        annotations = []
        
        try:
            if len(price_data) < 20:
                return annotations
            
            # Find significant swing high and low
            swing_high = price_data['high'].max()
            swing_low = price_data['low'].min()
            price_range = swing_high - swing_low
            
            # Calculate Fibonacci levels
            fib_levels = {
                'level_236': swing_high - 0.236 * price_range,
                'level_382': swing_high - 0.382 * price_range,
                'level_500': swing_high - 0.500 * price_range,
                'level_618': swing_high - 0.618 * price_range,
                'level_786': swing_high - 0.786 * price_range,
                'level_886': swing_high - 0.886 * price_range,
            }
            
            fib_annotation = SMCAnnotation(
                annotation_type=ChartAnnotation.FIBONACCI_LEVEL,
                symbol=price_data.attrs.get('symbol', 'Unknown'),
                timeframe=price_data.attrs.get('timeframe', 'Unknown'),
                timestamp=price_data.index[-1],
                price_levels=list(fib_levels.values()),
                confidence=0.9,
                metadata={
                    'fib_levels': fib_levels,
                    'swing_high': swing_high,
                    'swing_low': swing_low
                },
                visual_style={}
            )
            annotations.append(fib_annotation)
            
        except Exception as e:
            logging.error(f"Error detecting fibonacci levels: {e}")
        
        return annotations
    
    def detect_liquidity_zones(self, price_data: pd.DataFrame) -> List[SMCAnnotation]:
        """Detect Liquidity Zones"""
        annotations = []
        
        try:
            if len(price_data) < 50:
                return annotations
            
            # Find areas with high volume and price rejection
            high_volume_bars = price_data[price_data['volume'] > price_data['volume'].quantile(0.8)]
            
            for idx in high_volume_bars.index:
                bar_idx = price_data.index.get_loc(idx)
                if bar_idx > 0 and bar_idx < len(price_data) - 1:
                    prev_bar = price_data.iloc[bar_idx - 1]
                    curr_bar = price_data.iloc[bar_idx]
                    next_bar = price_data.iloc[bar_idx + 1]
                    
                    # Check for price rejection (long wicks)
                    upper_wick = curr_bar['high'] - max(curr_bar['open'], curr_bar['close'])
                    lower_wick = min(curr_bar['open'], curr_bar['close']) - curr_bar['low']
                    
                    if upper_wick > (curr_bar['high'] - curr_bar['low']) * 0.3:  # Significant upper wick
                        liquidity_annotation = SMCAnnotation(
                            annotation_type=ChartAnnotation.LIQUIDITY_ZONE,
                            symbol=price_data.attrs.get('symbol', 'Unknown'),
                            timeframe=price_data.attrs.get('timeframe', 'Unknown'),
                            timestamp=idx,
                            price_levels=[curr_bar['high'] - upper_wick * 0.5, curr_bar['high']],
                            confidence=0.6,
                            metadata={
                                'x_position': bar_idx,
                                'width': 5,
                                'liquidity_type': 'above'
                            },
                            visual_style={}
                        )
                        annotations.append(liquidity_annotation)
                    
                    if lower_wick > (curr_bar['high'] - curr_bar['low']) * 0.3:  # Significant lower wick
                        liquidity_annotation = SMCAnnotation(
                            annotation_type=ChartAnnotation.LIQUIDITY_ZONE,
                            symbol=price_data.attrs.get('symbol', 'Unknown'),
                            timeframe=price_data.attrs.get('timeframe', 'Unknown'),
                            timestamp=idx,
                            price_levels=[curr_bar['low'], curr_bar['low'] + lower_wick * 0.5],
                            confidence=0.6,
                            metadata={
                                'x_position': bar_idx,
                                'width': 5,
                                'liquidity_type': 'below'
                            },
                            visual_style={}
                        )
                        annotations.append(liquidity_annotation)
            
        except Exception as e:
            logging.error(f"Error detecting liquidity zones: {e}")
        
        return annotations
    
    def find_swing_highs(self, price_data: pd.DataFrame, window: int = 5) -> List[float]:
        """Find swing highs in price data"""
        highs = []
        for i in range(window, len(price_data) - window):
            if price_data['high'].iloc[i] == price_data['high'].iloc[i-window:i+window].max():
                highs.append(price_data['high'].iloc[i])
        return highs
    
    def find_swing_lows(self, price_data: pd.DataFrame, window: int = 5) -> List[float]:
        """Find swing lows in price data"""
        lows = []
        for i in range(window, len(price_data) - window):
            if price_data['low'].iloc[i] == price_data['low'].iloc[i-window:i+window].min():
                lows.append(price_data['low'].iloc[i])
        return lows
    
    def generate_chart_report(self, price_data: pd.DataFrame, 
                            annotations: List[SMCAnnotation]) -> Dict:
        """Generate comprehensive chart analysis report"""
        report = {
            'symbol': price_data.attrs.get('symbol', 'Unknown'),
            'timeframe': price_data.attrs.get('timeframe', 'Unknown'),
            'analysis_date': datetime.now().isoformat(),
            'price_statistics': self.get_price_statistics(price_data),
            'pattern_summary': self.get_pattern_summary(annotations),
            'market_structure': self.analyze_market_structure(price_data),
            'trading_bias': self.generate_trading_bias(annotations),
            'key_levels': self.extract_key_levels(annotations),
            'confidence_score': self.calculate_analysis_confidence(annotations)
        }
        
        return report
    
    def get_price_statistics(self, price_data: pd.DataFrame) -> Dict:
        """Get price statistics for report"""
        if len(price_data) == 0:
            return {}
        
        current_price = price_data['close'].iloc[-1]
        high_24h = price_data['high'].max()
        low_24h = price_data['low'].min()
        volume_24h = price_data['volume'].sum()
        
        return {
            'current_price': current_price,
            '24h_high': high_24h,
            '24h_low': low_24h,
            '24h_range': high_24h - low_24h,
            '24h_range_percent': ((high_24h - low_24h) / low_24h) * 100,
            '24h_volume': volume_24h,
            'price_change_24h': ((current_price - price_data['open'].iloc[0]) / price_data['open'].iloc[0]) * 100
        }
    
    def get_pattern_summary(self, annotations: List[SMCAnnotation]) -> Dict:
        """Get pattern detection summary"""
        pattern_counts = {}
        for annotation in annotations:
            pattern_type = annotation.annotation_type.value
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        return {
            'total_patterns': len(annotations),
            'pattern_breakdown': pattern_counts,
            'most_common_pattern': max(pattern_counts, key=pattern_counts.get) if pattern_counts else 'None'
        }
    
    def analyze_market_structure(self, price_data: pd.DataFrame) -> str:
        """Analyze current market structure"""
        if len(price_data) < 20:
            return "Insufficient Data"
        
        # Simple structure analysis
        recent_highs = price_data['high'].tail(10)
        recent_lows = price_data['low'].tail(10)
        
        if recent_highs.is_monotonic_increasing and recent_lows.is_monotonic_increasing:
            return "Bullish Trend"
        elif recent_highs.is_monotonic_decreasing and recent_lows.is_monotonic_decreasing:
            return "Bearish Trend"
        else:
            return "Ranging Market"
    
    def generate_trading_bias(self, annotations: List[SMCAnnotation]) -> str:
        """Generate trading bias based on patterns"""
        bullish_patterns = 0
        bearish_patterns = 0
        
        for annotation in annotations:
            if annotation.annotation_type in [ChartAnnotation.BREAK_OF_STRUCTURE, 
                                            ChartAnnotation.FAIR_VALUE_GAP]:
                direction = annotation.metadata.get('direction', '')
                if direction == 'bullish':
                    bullish_patterns += 1
                elif direction == 'bearish':
                    bearish_patterns += 1
        
        if bullish_patterns > bearish_patterns + 2:
            return "Bullish"
        elif bearish_patterns > bullish_patterns + 2:
            return "Bearish"
        else:
            return "Neutral"
    
    def extract_key_levels(self, annotations: List[SMCAnnotation]) -> List[Dict]:
        """Extract key support/resistance levels"""
        key_levels = []
        
        for annotation in annotations:
            if annotation.annotation_type in [ChartAnnotation.ORDER_BLOCK,
                                            ChartAnnotation.LIQUIDITY_ZONE,
                                            ChartAnnotation.FIBONACCI_LEVEL]:
                level_type = "Support" if annotation.metadata.get('direction') == 'bullish' else "Resistance"
                key_levels.append({
                    'type': level_type,
                    'price_levels': annotation.price_levels,
                    'pattern': annotation.annotation_type.value,
                    'confidence': annotation.confidence
                })
        
        # Remove duplicates and sort
        unique_levels = []
        seen_levels = set()
        
        for level in key_levels:
            level_key = tuple(level['price_levels'])
            if level_key not in seen_levels:
                unique_levels.append(level)
                seen_levels.add(level_key)
        
        return sorted(unique_levels, key=lambda x: min(x['price_levels']))
    
    def calculate_analysis_confidence(self, annotations: List[SMCAnnotation]) -> float:
        """Calculate overall analysis confidence score"""
        if not annotations:
            return 0.0
        
        total_confidence = sum(ann.confidence for ann in annotations)
        avg_confidence = total_confidence / len(annotations)
        
        # Adjust based on pattern variety
        unique_patterns = len(set(ann.annotation_type for ann in annotations))
        pattern_bonus = min(unique_patterns * 0.05, 0.2)  # Max 20% bonus
        
        return min(avg_confidence + pattern_bonus, 1.0)