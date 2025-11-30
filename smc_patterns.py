"""
Babs AI Trading System - SMC Pattern Detection
Smart Money Concepts pattern detection without TA-Lib dependency
"""
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class OrderBlock:
    symbol: str
    timestamp: datetime
    price_high: float
    price_low: float
    direction: str  # 'bullish' or 'bearish'
    strength: float  # 0-1 confidence score
    timeframe: str


@dataclass
class FairValueGap:
    symbol: str
    timestamp: datetime
    gap_high: float
    gap_low: float
    direction: str
    strength: float
    timeframe: str


@dataclass
class BreakOfStructure:
    symbol: str
    timestamp: datetime
    level: float
    direction: str  # 'up' or 'down'
    confirmed: bool
    timeframe: str


@dataclass
class LiquidityLevel:
    symbol: str
    timestamp: datetime
    price: float
    type: str  # 'high' or 'low'
    strength: float
    timeframe: str


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI without TA-Lib"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ATR without TA-Lib"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ADX without TA-Lib"""
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    adx = dx.rolling(window=period).mean()
    
    return adx


def rolling_max(series: pd.Series, period: int) -> pd.Series:
    """Calculate rolling maximum"""
    return series.rolling(window=period).max()


def rolling_min(series: pd.Series, period: int) -> pd.Series:
    """Calculate rolling minimum"""
    return series.rolling(window=period).min()


class SMCPatternDetector:
    """Smart Money Concepts pattern detector"""
    
    def __init__(self):
        self.min_ob_strength = 0.6
        self.fvg_threshold = 0.001  # 0.1% minimum gap
        
    def detect_all_patterns(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Detect all SMC patterns in price data"""
        patterns = {
            'order_blocks': self.detect_order_blocks(df, symbol, timeframe),
            'fair_value_gaps': self.detect_fair_value_gaps(df, symbol, timeframe),
            'break_of_structure': self.detect_break_of_structure(df, symbol, timeframe),
            'liquidity_levels': self.detect_liquidity_levels(df, symbol, timeframe),
            'wyckoff_phases': self.detect_wyckoff_phases(df, symbol, timeframe)
        }
        return patterns
    
    def detect_order_blocks(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[OrderBlock]:
        """Detect Order Blocks using institutional logic"""
        order_blocks = []
        
        if len(df) < 5:
            return order_blocks
        
        # Calculate price movements and volatility
        df = df.copy()
        df['price_change'] = df['close'].pct_change()
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        
        if 'volume' in df.columns:
            df['volume_spike'] = df['volume'] > df['volume'].rolling(20).mean() * 1.5
        else:
            df['volume_spike'] = False
        
        for i in range(2, len(df)-1):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            next_candle = df.iloc[i+1]
            
            # Bullish Order Block
            if (prev['close'] < prev['open'] and
                current['close'] > current['open'] and
                next_candle['close'] > next_candle['open'] and
                abs(next_candle['price_change']) > 0.002):
                
                strength = self.calculate_ob_strength(df, i, 'bullish')
                if strength >= self.min_ob_strength:
                    ob = OrderBlock(
                        symbol=symbol,
                        timestamp=current.name if hasattr(current, 'name') else datetime.now(),
                        price_high=current['high'],
                        price_low=current['low'],
                        direction='bullish',
                        strength=strength,
                        timeframe=timeframe
                    )
                    order_blocks.append(ob)
            
            # Bearish Order Block
            elif (prev['close'] > prev['open'] and
                  current['close'] < current['open'] and
                  next_candle['close'] < next_candle['open'] and
                  abs(next_candle['price_change']) > 0.002):
                
                strength = self.calculate_ob_strength(df, i, 'bearish')
                if strength >= self.min_ob_strength:
                    ob = OrderBlock(
                        symbol=symbol,
                        timestamp=current.name if hasattr(current, 'name') else datetime.now(),
                        price_high=current['high'],
                        price_low=current['low'],
                        direction='bearish',
                        strength=strength,
                        timeframe=timeframe
                    )
                    order_blocks.append(ob)
        
        return order_blocks[-10:]
    
    def calculate_ob_strength(self, df: pd.DataFrame, index: int, direction: str) -> float:
        """Calculate Order Block strength (0-1)"""
        current = df.iloc[index]
        strength = 0.0
        
        # Volume confirmation (30%)
        if current.get('volume_spike', False):
            strength += 0.3
        else:
            strength += 0.15  # Base volume score
        
        # Price range significance (30%)
        range_strength = min(current['high_low_range'] / 0.01, 1.0)
        strength += range_strength * 0.3
        
        # Subsequent move strength (40%)
        if index + 1 < len(df):
            if direction == 'bullish':
                move_strength = min((df.iloc[index+1]['close'] - current['close']) / current['close'] / 0.005, 1.0)
            else:
                move_strength = min((current['close'] - df.iloc[index+1]['close']) / current['close'] / 0.005, 1.0)
            strength += max(move_strength, 0) * 0.4
        
        return min(strength, 1.0)
    
    def detect_fair_value_gaps(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[FairValueGap]:
        """Detect Fair Value Gaps in price action"""
        fair_value_gaps = []
        
        if len(df) < 3:
            return fair_value_gaps
        
        for i in range(1, len(df)-1):
            current = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Bullish FVG
            if current['low'] > next_candle['high']:
                gap_size = (current['low'] - next_candle['high']) / current['close']
                if gap_size >= self.fvg_threshold:
                    fvg = FairValueGap(
                        symbol=symbol,
                        timestamp=current.name if hasattr(current, 'name') else datetime.now(),
                        gap_high=current['low'],
                        gap_low=next_candle['high'],
                        direction='bullish',
                        strength=min(gap_size / 0.005, 1.0),
                        timeframe=timeframe
                    )
                    fair_value_gaps.append(fvg)
            
            # Bearish FVG
            elif current['high'] < next_candle['low']:
                gap_size = (next_candle['low'] - current['high']) / current['close']
                if gap_size >= self.fvg_threshold:
                    fvg = FairValueGap(
                        symbol=symbol,
                        timestamp=current.name if hasattr(current, 'name') else datetime.now(),
                        gap_high=next_candle['low'],
                        gap_low=current['high'],
                        direction='bearish',
                        strength=min(gap_size / 0.005, 1.0),
                        timeframe=timeframe
                    )
                    fair_value_gaps.append(fvg)
        
        return fair_value_gaps[-15:]
    
    def detect_break_of_structure(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[BreakOfStructure]:
        """Detect Break of Structure and Change of Character"""
        bos_signals = []
        
        if len(df) < 15:
            return bos_signals
        
        # Calculate swing highs and lows
        df = df.copy()
        df['swing_high'] = rolling_max(df['high'], 5)
        df['swing_low'] = rolling_min(df['low'], 5)
        
        for i in range(10, len(df)-5):
            current = df.iloc[i]
            
            # Bullish BOS
            if (current['close'] > df.iloc[i-5:i]['swing_high'].max() and
                current['close'] > current['open']):
                
                bos = BreakOfStructure(
                    symbol=symbol,
                    timestamp=current.name if hasattr(current, 'name') else datetime.now(),
                    level=current['close'],
                    direction='up',
                    confirmed=self.confirm_bos(df, i, 'up'),
                    timeframe=timeframe
                )
                bos_signals.append(bos)
            
            # Bearish BOS
            elif (current['close'] < df.iloc[i-5:i]['swing_low'].min() and
                  current['close'] < current['open']):
                
                bos = BreakOfStructure(
                    symbol=symbol,
                    timestamp=current.name if hasattr(current, 'name') else datetime.now(),
                    level=current['close'],
                    direction='down',
                    confirmed=self.confirm_bos(df, i, 'down'),
                    timeframe=timeframe
                )
                bos_signals.append(bos)
        
        return bos_signals[-5:]
    
    def confirm_bos(self, df: pd.DataFrame, index: int, direction: str) -> bool:
        """Confirm Break of Structure with follow-through"""
        if direction == 'up':
            for i in range(1, 3):
                if index + i < len(df):
                    if df.iloc[index+i]['low'] < df.iloc[index]['close']:
                        return False
            return True
        else:
            for i in range(1, 3):
                if index + i < len(df):
                    if df.iloc[index+i]['high'] > df.iloc[index]['close']:
                        return False
            return True
    
    def detect_liquidity_levels(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[LiquidityLevel]:
        """Detect key liquidity levels"""
        liquidity_levels = []
        
        if len(df) < 50:
            return liquidity_levels
        
        # Recent highs (potential sell-side liquidity)
        recent_highs = df['high'].tail(50).nlargest(3)
        for level in recent_highs:
            matching_rows = df[df['high'] == level]
            if len(matching_rows) > 0:
                liquidity_levels.append(
                    LiquidityLevel(
                        symbol=symbol,
                        timestamp=matching_rows.index[-1] if hasattr(matching_rows.index[-1], 'isoformat') else datetime.now(),
                        price=level,
                        type='high',
                        strength=0.8,
                        timeframe=timeframe
                    )
                )
        
        # Recent lows (potential buy-side liquidity)
        recent_lows = df['low'].tail(50).nsmallest(3)
        for level in recent_lows:
            matching_rows = df[df['low'] == level]
            if len(matching_rows) > 0:
                liquidity_levels.append(
                    LiquidityLevel(
                        symbol=symbol,
                        timestamp=matching_rows.index[-1] if hasattr(matching_rows.index[-1], 'isoformat') else datetime.now(),
                        price=level,
                        type='low',
                        strength=0.8,
                        timeframe=timeframe
                    )
                )
        
        return liquidity_levels
    
    def detect_wyckoff_phases(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Detect Wyckoff accumulation/distribution phases"""
        phases = {
            'phase': 'unknown',
            'confidence': 0.0,
            'levels': {}
        }
        
        if len(df) < 20:
            return phases
        
        current_price = df['close'].iloc[-1]
        
        # Simple phase detection based on price position in range
        if current_price > df['close'].quantile(0.7):
            phases['phase'] = 'distribution'
            phases['confidence'] = 0.7
        elif current_price < df['close'].quantile(0.3):
            phases['phase'] = 'accumulation'
            phases['confidence'] = 0.7
        else:
            phases['phase'] = 'markup/markdown'
            phases['confidence'] = 0.5
        
        return phases


class FibonacciAnalyzer:
    """Fibonacci level analyzer"""
    
    def __init__(self):
        self.levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    def calculate_fib_levels(self, high: float, low: float) -> Dict[float, float]:
        """Calculate Fibonacci retracement levels"""
        fib_levels = {}
        price_range = high - low
        
        for level in self.levels:
            fib_price = high - (price_range * level)
            fib_levels[level] = fib_price
        
        # Add extensions
        extensions = [1.272, 1.414, 1.618]
        for ext in extensions:
            fib_price = high + (price_range * (ext - 1.0))
            fib_levels[ext] = fib_price
        
        return fib_levels
    
    def find_fib_clusters(self, patterns: Dict, current_price: float) -> List[Dict]:
        """Find Fibonacci level clusters for confluence"""
        clusters = []
        
        if len(patterns.get('order_blocks', [])) > 0:
            latest_ob = patterns['order_blocks'][-1]
            ob_mid = (latest_ob.price_high + latest_ob.price_low) / 2
            
            price_diff = abs(current_price - ob_mid) / current_price
            if price_diff < 0.002:
                clusters.append({
                    'type': 'ob_fib_confluence',
                    'level': ob_mid,
                    'strength': 0.8,
                    'description': 'Price at Order Block + Fib level'
                })
        
        return clusters
