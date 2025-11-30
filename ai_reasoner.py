import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import sqlite3
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import talib

@dataclass
class TradeSignal:
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    take_profit: List[float]
    confidence: float
    reasoning: List[str]
    timestamp: datetime
    timeframe: str
    r_r_ratio: float

@dataclass
class MarketRegime:
    regime: str  # 'TRENDING', 'RANGING', 'VOLATILE', 'CHOppy'
    confidence: float
    indicators: Dict
    timestamp: datetime

@dataclass
class LearningMemory:
    trade_outcome: str  # 'WIN', 'LOSS', 'BREAKEVEN'
    setup_conditions: Dict
    market_regime: str
    performance_metrics: Dict
    timestamp: datetime

class AIReasoningEngine:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.learning_memory = []
        self.market_regimes = {}
        self.init_models()
        
    def init_models(self):
        """Initialize machine learning models"""
        # Random Forest for pattern recognition
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Gradient Boosting for confidence scoring
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        
        # LSTM for sequence pattern recognition
        self.lstm_model = self.build_lstm_model()
        
        self.is_trained = False
    
    def build_lstm_model(self):
        """Build LSTM model for time series pattern recognition"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(20, 15)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def extract_features(self, df: pd.DataFrame, patterns: Dict) -> np.ndarray:
        """Extract comprehensive features for AI reasoning"""
        features = []
        
        # Price-based features
        features.extend([
            df['close'].pct_change().iloc[-1],  # Latest return
            df['close'].pct_change().rolling(5).mean().iloc[-1],  # 5-period avg return
            (df['high'] - df['low']).mean() / df['close'].mean(),  # Normalized volatility
            talib.RSI(df['close'], timeperiod=14).iloc[-1] / 100,  # RSI normalized
            talib.ADX(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1] / 100,  # ADX normalized
        ])
        
        # Volume features
        if 'volume' in df.columns:
            features.extend([
                df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1],  # Volume ratio
                (df['volume'] * df['close']).rolling(5).mean().iloc[-1],  # Dollar volume
            ])
        
        # SMC Pattern features
        ob_features = self.extract_order_block_features(patterns.get('order_blocks', []))
        fvg_features = self.extract_fvg_features(patterns.get('fair_value_gaps', []))
        bos_features = self.extract_bos_features(patterns.get('break_of_structure', []))
        
        features.extend(ob_features)
        features.extend(fvg_features)
        features.extend(bos_features)
        
        # Market regime features
        regime_features = self.extract_regime_features(df)
        features.extend(regime_features)
        
        # Fibonacci confluence features
        fib_features = self.extract_fibonacci_features(patterns.get('fibonacci', {}))
        features.extend(fib_features)
        
        return np.array(features).reshape(1, -1)
    
    def extract_order_block_features(self, order_blocks: List) -> List[float]:
        """Extract features from Order Blocks"""
        if not order_blocks:
            return [0.0, 0.0, 0.0]
        
        recent_obs = order_blocks[-5:]  # Last 5 order blocks
        
        bullish_obs = [ob for ob in recent_obs if ob.direction == 'bullish']
        bearish_obs = [ob for ob in recent_obs if ob.direction == 'bearish']
        
        return [
            len(bullish_obs) / len(recent_obs) if recent_obs else 0.0,
            len(bearish_obs) / len(recent_obs) if recent_obs else 0.0,
            np.mean([ob.strength for ob in recent_obs]) if recent_obs else 0.0
        ]
    
    def extract_fvg_features(self, fvgs: List) -> List[float]:
        """Extract features from Fair Value Gaps"""
        if not fvgs:
            return [0.0, 0.0, 0.0]
        
        recent_fvgs = fvgs[-5:]
        
        bullish_fvgs = [fvg for fvg in recent_fvgs if fvg.direction == 'bullish']
        bearish_fvgs = [fvg for fvg in recent_fvgs if fvg.direction == 'bearish']
        
        return [
            len(bullish_fvgs) / len(recent_fvgs) if recent_fvgs else 0.0,
            len(bearish_fvgs) / len(recent_fvgs) if recent_fvgs else 0.0,
            np.mean([fvg.strength for fvg in recent_fvgs]) if recent_fvgs else 0.0
        ]
    
    def extract_bos_features(self, bos_signals: List) -> List[float]:
        """Extract features from Break of Structure"""
        if not bos_signals:
            return [0.0, 0.0, 0.0]
        
        recent_bos = bos_signals[-3:]
        
        bullish_bos = [bos for bos in recent_bos if bos.direction == 'up']
        bearish_bos = [bos for bos in recent_bos if bos.direction == 'down']
        confirmed_bos = [bos for bos in recent_bos if bos.confirmed]
        
        return [
            len(bullish_bos) / len(recent_bos) if recent_bos else 0.0,
            len(bearish_bos) / len(recent_bos) if recent_bos else 0.0,
            len(confirmed_bos) / len(recent_bos) if recent_bos else 0.0
        ]
    
    def extract_regime_features(self, df: pd.DataFrame) -> List[float]:
        """Extract market regime features"""
        # Volatility regime
        volatility = df['close'].pct_change().std()
        avg_volatility = df['close'].pct_change().rolling(20).std().mean()
        
        # Trend regime
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        trend_strength = adx / 100  # Normalize
        
        # Range regime
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        price_range = (df['close'] - low_20) / (high_20 - low_20)
        range_position = price_range.iloc[-1]
        
        return [volatility / avg_volatility, trend_strength, range_position]
    
    def extract_fibonacci_features(self, fib_data: Dict) -> List[float]:
        """Extract Fibonacci confluence features"""
        if not fib_data.get('clusters'):
            return [0.0, 0.0]
        
        clusters = fib_data['clusters']
        return [
            len(clusters),
            np.mean([cluster.get('strength', 0) for cluster in clusters])
        ]
    
    def analyze_setup(self, symbol: str, df: pd.DataFrame, patterns: Dict, 
                     multi_timeframe_analysis: Dict) -> TradeSignal:
        """Comprehensive AI analysis of trading setup"""
        
        # Extract features for AI reasoning
        features = self.extract_features(df, patterns)
        
        # Get AI confidence score
        confidence = self.calculate_confidence(features, patterns, multi_timeframe_analysis)
        
        # Determine trade direction
        direction = self.determine_direction(patterns, multi_timeframe_analysis)
        
        # Calculate entry, SL, TP levels
        entry_price = df['close'].iloc[-1]
        stop_loss = self.calculate_stop_loss(df, patterns, direction)
        take_profit = self.calculate_take_profit(entry_price, stop_loss, direction, df)
        
        # Risk-reward ratio
        r_r_ratio = self.calculate_risk_reward(entry_price, stop_loss, take_profit)
        
        # Generate reasoning
        reasoning = self.generate_reasoning(patterns, multi_timeframe_analysis, confidence)
        
        # Apply reinforcement learning adjustments
        confidence = self.apply_learning_adjustments(confidence, symbol, patterns)
        
        return TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now(),
            timeframe='1h',
            r_r_ratio=r_r_ratio
        )
    
    def calculate_confidence(self, features: np.ndarray, patterns: Dict, 
                           multi_tf_analysis: Dict) -> float:
        """Calculate AI confidence score (0-1)"""
        base_confidence = 0.5
        
        # Pattern confluence scoring
        pattern_score = self.score_pattern_confluence(patterns)
        base_confidence += pattern_score * 0.3
        
        # Multi-timeframe alignment scoring
        tf_score = self.score_timeframe_alignment(multi_tf_analysis)
        base_confidence += tf_score * 0.3
        
        # Market regime scoring
        regime_score = self.score_market_regime(features)
        base_confidence += regime_score * 0.2
        
        # Volume confirmation (if available)
        volume_score = self.score_volume_confirmation(patterns)
        base_confidence += volume_score * 0.1
        
        # Fibonacci confluence scoring
        fib_score = self.score_fibonacci_confluence(patterns.get('fibonacci', {}))
        base_confidence += fib_score * 0.1
        
        return max(0.0, min(1.0, base_confidence))
    
    def score_pattern_confluence(self, patterns: Dict) -> float:
        """Score pattern confluence"""
        score = 0.0
        
        # Order Block strength
        if patterns.get('order_blocks'):
            ob_strengths = [ob.strength for ob in patterns['order_blocks'][-3:]]
            score += np.mean(ob_strengths) * 0.4
        
        # FVG strength
        if patterns.get('fair_value_gaps'):
            fvg_strengths = [fvg.strength for fvg in patterns['fair_value_gaps'][-3:]]
            score += np.mean(fvg_strengths) * 0.3
        
        # BOS confirmation
        if patterns.get('break_of_structure'):
            confirmed_bos = [bos for bos in patterns['break_of_structure'] if bos.confirmed]
            if confirmed_bos:
                score += 0.3
        
        return score
    
    def score_timeframe_alignment(self, multi_tf_analysis: Dict) -> float:
        """Score multi-timeframe alignment"""
        tf_data = multi_tf_analysis.get('multi_timeframe_analysis', {})
        
        if not tf_data:
            return 0.0
        
        # Count aligned timeframes
        biases = [data.get('bias') for data in tf_data.values() if 'bias' in data]
        if not biases:
            return 0.0
        
        primary_bias = multi_tf_analysis.get('composite_bias', 'neutral')
        aligned_count = sum(1 for bias in biases if bias == primary_bias)
        
        return aligned_count / len(biases)
    
    def score_market_regime(self, features: np.ndarray) -> float:
        """Score suitability for current market regime"""
        # Simplified regime scoring
        volatility = features[0, 2]  # Volatility feature
        trend_strength = features[0, 3]  # Trend strength feature
        
        # Prefer trending markets with moderate volatility
        if trend_strength > 0.3 and 0.5 < volatility < 2.0:
            return 0.8
        elif trend_strength > 0.5:
            return 0.9
        else:
            return 0.3
    
    def score_volume_confirmation(self, patterns: Dict) -> float:
        """Score volume confirmation for patterns"""
        # Placeholder - in production, integrate actual volume analysis
        return 0.5
    
    def score_fibonacci_confluence(self, fib_data: Dict) -> float:
        """Score Fibonacci level confluence"""
        clusters = fib_data.get('clusters', [])
        if not clusters:
            return 0.0
        
        cluster_strengths = [cluster.get('strength', 0) for cluster in clusters]
        return np.mean(cluster_strengths)
    
    def determine_direction(self, patterns: Dict, multi_tf_analysis: Dict) -> str:
        """Determine trade direction based on analysis"""
        composite_bias = multi_tf_analysis.get('composite_bias', 'neutral')
        
        # Use composite bias as primary direction
        if composite_bias == 'bullish':
            return 'BUY'
        elif composite_bias == 'bearish':
            return 'SELL'
        
        # Fallback to pattern-based direction
        recent_obs = patterns.get('order_blocks', [])[-3:]
        if recent_obs:
            bullish_obs = sum(1 for ob in recent_obs if ob.direction == 'bullish')
            bearish_obs = sum(1 for ob in recent_obs if ob.direction == 'bearish')
            
            if bullish_obs > bearish_obs:
                return 'BUY'
            elif bearish_obs > bullish_obs:
                return 'SELL'
        
        return 'BUY'  # Default fallback
    
    def calculate_stop_loss(self, df: pd.DataFrame, patterns: Dict, direction: str) -> float:
        """Calculate intelligent stop loss level"""
        current_price = df['close'].iloc[-1]
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        
        if direction == 'BUY':
            # Look for recent bearish order blocks or liquidity levels
            bearish_obs = [ob for ob in patterns.get('order_blocks', []) 
                          if ob.direction == 'bearish']
            if bearish_obs:
                recent_ob = bearish_obs[-1]
                sl_candidate = recent_ob.price_low - (atr * 0.5)
            else:
                sl_candidate = current_price - (atr * 1.5)
            
            # Find nearest liquidity level below
            liquidity_lows = [ll for ll in patterns.get('liquidity_levels', [])
                             if ll.type == 'low' and ll.price < current_price]
            if liquidity_lows:
                nearest_low = max(ll.price for ll in liquidity_lows)
                sl_candidate = min(sl_candidate, nearest_low - (atr * 0.2))
        
        else:  # SELL
            # Look for recent bullish order blocks or liquidity levels
            bullish_obs = [ob for ob in patterns.get('order_blocks', []) 
                          if ob.direction == 'bullish']
            if bullish_obs:
                recent_ob = bullish_obs[-1]
                sl_candidate = recent_ob.price_high + (atr * 0.5)
            else:
                sl_candidate = current_price + (atr * 1.5)
            
            # Find nearest liquidity level above
            liquidity_highs = [ll for ll in patterns.get('liquidity_levels', [])
                              if ll.type == 'high' and ll.price > current_price]
            if liquidity_highs:
                nearest_high = min(ll.price for ll in liquidity_highs)
                sl_candidate = max(sl_candidate, nearest_high + (atr * 0.2))
        
        return round(sl_candidate, 5)
    
    def calculate_take_profit(self, entry: float, sl: float, direction: str, 
                            df: pd.DataFrame) -> List[float]:
        """Calculate multiple take profit levels"""
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        risk = abs(entry - sl)
        
        if direction == 'BUY':
            tp1 = entry + risk * 1.0
            tp2 = entry + risk * 1.5
            tp3 = entry + risk * 2.0
        else:  # SELL
            tp1 = entry - risk * 1.0
            tp2 = entry - risk * 1.5
            tp3 = entry - risk * 2.0
        
        # Adjust based on nearby resistance/support levels
        # This would integrate with SMC levels in production
        
        return [round(tp, 5) for tp in [tp1, tp2, tp3]]
    
    def calculate_risk_reward(self, entry: float, sl: float, tp_levels: List[float]) -> float:
        """Calculate risk-reward ratio"""
        risk = abs(entry - sl)
        avg_reward = np.mean([abs(tp - entry) for tp in tp_levels])
        
        return avg_reward / risk if risk > 0 else 0.0
    
    def generate_reasoning(self, patterns: Dict, multi_tf_analysis: Dict, 
                          confidence: float) -> List[str]:
        """Generate human-readable reasoning for the trade"""
        reasoning = []
        
        # Multi-timeframe reasoning
        composite_bias = multi_tf_analysis.get('composite_bias', 'neutral')
        reasoning.append(f"Composite {composite_bias.upper()} bias across timeframes")
        
        # Pattern reasoning
        if patterns.get('order_blocks'):
            recent_ob = patterns['order_blocks'][-1]
            reasoning.append(f"Recent {recent_ob.direction} Order Block (Strength: {recent_ob.strength:.0%})")
        
        if patterns.get('break_of_structure'):
            recent_bos = patterns['break_of_structure'][-1]
            if recent_bos.confirmed:
                reasoning.append(f"Confirmed {recent_bos.direction.upper()} Break of Structure")
        
        # Confidence-based reasoning
        if confidence > 0.8:
            reasoning.append("High confidence setup with strong pattern confluence")
        elif confidence > 0.6:
            reasoning.append("Moderate confidence with good pattern alignment")
        else:
            reasoning.append("Lower confidence - awaiting better confirmation")
        
        return reasoning
    
    def apply_learning_adjustments(self, confidence: float, symbol: str, 
                                 patterns: Dict) -> float:
        """Apply reinforcement learning adjustments to confidence"""
        # Load historical performance for this symbol/setup
        historical_performance = self.load_historical_performance(symbol, patterns)
        
        if historical_performance:
            win_rate = historical_performance.get('win_rate', 0.5)
            # Adjust confidence based on historical performance
            confidence *= (0.5 + win_rate)  # Scale by historical success
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def load_historical_performance(self, symbol: str, patterns: Dict) -> Dict:
        """Load historical performance for similar setups"""
        # This would query the database for similar historical setups
        # Simplified implementation for demo
        return {
            'win_rate': 0.65,
            'avg_rr': 1.8,
            'total_trades': 24
        }
    
    def learn_from_trade(self, trade_signal: TradeSignal, outcome: str, 
                        performance: Dict):
        """Learn from trade outcome for future improvements"""
        memory = LearningMemory(
            trade_outcome=outcome,
            setup_conditions=self.extract_setup_conditions(trade_signal),
            market_regime=self.detect_current_regime(),
            performance_metrics=performance,
            timestamp=datetime.now()
        )
        
        self.learning_memory.append(memory)
        self.update_models()
    
    def extract_setup_conditions(self, trade_signal: TradeSignal) -> Dict:
        """Extract setup conditions for learning"""
        return {
            'symbol': trade_signal.symbol,
            'direction': trade_signal.direction,
            'confidence': trade_signal.confidence,
            'r_r_ratio': trade_signal.r_r_ratio,
            'timeframe': trade_signal.timeframe
        }
    
    def detect_current_regime(self) -> str:
        """Detect current market regime"""
        # Simplified implementation
        return "TRENDING"
    
    def update_models(self):
        """Update ML models with new learning data"""
        if len(self.learning_memory) < 10:  # Minimum samples needed
            return
        
        # Prepare training data from learning memory
        X = []
        y = []
        
        for memory in self.learning_memory[-100:]:  # Use last 100 samples
            features = self.memory_to_features(memory)
            X.append(features)
            y.append(1 if memory.trade_outcome == 'WIN' else 0)
        
        if len(X) > 10:
            X = np.array(X)
            y = np.array(y)
            
            # Update Random Forest model
            self.rf_model.fit(X, y)
            
            print(f"AI Models updated with {len(X)} new samples")
    
    def memory_to_features(self, memory: LearningMemory) -> List[float]:
        """Convert learning memory to feature vector"""
        features = []
        
        # Setup conditions
        setup = memory.setup_conditions
        features.extend([
            1.0 if setup['direction'] == 'BUY' else 0.0,
            setup['confidence'],
            setup['r_r_ratio']
        ])
        
        # Market regime
        regime = memory.market_regime
        regime_features = {
            'TRENDING': [1, 0, 0],
            'RANGING': [0, 1, 0], 
            'VOLATILE': [0, 0, 1]
        }
        features.extend(regime_features.get(regime, [0, 0, 0]))
        
        # Performance metrics
        performance = memory.performance_metrics
        features.extend([
            performance.get('duration_minutes', 0) / 1440,  # Normalized
            performance.get('max_favor', 0),
            performance.get('max_adverse', 0)
        ])
        
        return features

class ReinforcementLearner:
    """Advanced Reinforcement Learning for continuous improvement"""
    
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1
        
    def get_state_key(self, features: np.ndarray) -> str:
        """Convert features to state key for Q-learning"""
        # Discretize features for Q-table
        discretized = [int(f * 10) for f in features.flatten()]  # Simple discretization
        return str(discretized)
    
    def choose_action(self, state_key: str, possible_actions: List[str]) -> str:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.exploration_rate:
            return np.random.choice(possible_actions)
        
        # Choose best action from Q-table
        if state_key in self.q_table:
            q_values = self.q_table[state_key]
            return max(q_values, key=q_values.get)
        else:
            return np.random.choice(possible_actions)
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value using Bellman equation"""
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in ['BUY', 'SELL', 'HOLD']}
        
        current_q = self.q_table[state].get(action, 0)
        max_next_q = max(self.q_table.get(next_state, {'HOLD': 0}).values())
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def calculate_reward(self, trade_outcome: str, performance: Dict) -> float:
        """Calculate reward based on trade outcome"""
        if trade_outcome == 'WIN':
            base_reward = 10.0
            # Bonus for good risk-reward
            rr_ratio = performance.get('risk_reward_ratio', 1.0)
            return base_reward * min(rr_ratio, 3.0)
        elif trade_outcome == 'LOSS':
            return -8.0
        else:  # BREAKEVEN
            return 0.0