import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import sqlite3
from enum import Enum
import json

class SignalPriority(Enum):
    HIGH = 3
    MEDIUM = 2
    LOW = 1

class DeliveryStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    CONFIRMED = "confirmed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

@dataclass
class RankedSignal:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: List[float]
    confidence: float
    priority: SignalPriority
    ranking_score: float
    reasoning: List[str]
    timestamp: datetime
    expiry: datetime
    delivery_status: DeliveryStatus
    delivery_channels: List[str]
    risk_reward_ratio: float
    setup_quality: float

@dataclass
class SignalFilter:
    min_confidence: float = 0.6
    min_rr_ratio: float = 1.5
    max_daily_signals: int = 3
    allowed_instruments: List[str] = None
    exclude_high_impact_news: bool = True
    require_multi_tf_alignment: bool = True

class SignalRankingEngine:
    def __init__(self):
        self.weights = {
            'confidence': 0.25,
            'risk_reward': 0.20,
            'setup_quality': 0.15,
            'multi_tf_alignment': 0.15,
            'sentiment_alignment': 0.10,
            'volume_confirmation': 0.10,
            'pattern_confluence': 0.05
        }
        self.sent_today = 0
        self.last_reset = datetime.now().date()
        self.init_database()
    
    def init_database(self):
        """Initialize signal tracking database"""
        conn = sqlite3.connect('trading_signals.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit TEXT NOT NULL,
                confidence REAL NOT NULL,
                ranking_score REAL NOT NULL,
                priority TEXT NOT NULL,
                reasoning TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                expiry DATETIME NOT NULL,
                delivery_status TEXT NOT NULL,
                delivery_channels TEXT NOT NULL,
                risk_reward_ratio REAL NOT NULL,
                setup_quality REAL NOT NULL,
                sent_count INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_deliveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                channel TEXT NOT NULL,
                status TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                message_id TEXT,
                FOREIGN KEY (signal_id) REFERENCES signals (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def rank_signals(self, raw_signals: List, filter_config: SignalFilter = None) -> List[RankedSignal]:
        """Rank and filter trading signals"""
        if filter_config is None:
            filter_config = SignalFilter()
        
        # Reset daily counter if new day
        self.reset_daily_counter_if_needed()
        
        ranked_signals = []
        
        for signal in raw_signals:
            # Calculate ranking score
            ranking_score = self.calculate_ranking_score(signal)
            
            # Apply filters
            if not self.passes_filters(signal, ranking_score, filter_config):
                continue
            
            # Determine priority
            priority = self.determine_priority(ranking_score)
            
            # Create ranked signal
            ranked_signal = RankedSignal(
                symbol=signal.get('symbol'),
                direction=signal.get('direction'),
                entry_price=signal.get('entry_price'),
                stop_loss=signal.get('stop_loss'),
                take_profit=signal.get('take_profit', []),
                confidence=signal.get('confidence', 0),
                priority=priority,
                ranking_score=ranking_score,
                reasoning=signal.get('reasoning', []),
                timestamp=datetime.now(),
                expiry=datetime.now() + timedelta(hours=4),  # 4-hour expiry
                delivery_status=DeliveryStatus.PENDING,
                delivery_channels=self.get_delivery_channels(priority),
                risk_reward_ratio=signal.get('risk_reward_ratio', 1.0),
                setup_quality=self.calculate_setup_quality(signal)
            )
            
            ranked_signals.append(ranked_signal)
        
        # Sort by ranking score and return top N
        ranked_signals.sort(key=lambda x: x.ranking_score, reverse=True)
        return ranked_signals[:filter_config.max_daily_signals]
    
    def calculate_ranking_score(self, signal: Dict) -> float:
        """Calculate comprehensive ranking score (0-100)"""
        score = 0.0
        
        # Confidence score (0-25)
        confidence_score = signal.get('confidence', 0) * self.weights['confidence'] * 100
        score += confidence_score
        
        # Risk-reward score (0-20)
        rr_ratio = signal.get('risk_reward_ratio', 1.0)
        rr_score = min(rr_ratio / 3.0, 1.0) * self.weights['risk_reward'] * 100
        score += rr_score
        
        # Setup quality score (0-15)
        setup_quality = self.calculate_setup_quality(signal)
        score += setup_quality * self.weights['setup_quality'] * 100
        
        # Multi-timeframe alignment (0-15)
        mtf_score = self.calculate_multi_tf_score(signal)
        score += mtf_score * self.weights['multi_tf_alignment'] * 100
        
        # Sentiment alignment (0-10)
        sentiment_score = self.calculate_sentiment_score(signal)
        score += sentiment_score * self.weights['sentiment_alignment'] * 100
        
        # Volume confirmation (0-10)
        volume_score = self.calculate_volume_score(signal)
        score += volume_score * self.weights['volume_confirmation'] * 100
        
        # Pattern confluence (0-5)
        pattern_score = self.calculate_pattern_score(signal)
        score += pattern_score * self.weights['pattern_confluence'] * 100
        
        return min(score, 100.0)
    
    def calculate_setup_quality(self, signal: Dict) -> float:
        """Calculate setup quality score (0-1)"""
        quality_factors = []
        
        # SMC pattern strength
        patterns = signal.get('patterns', {})
        if patterns.get('order_blocks'):
            ob_strength = np.mean([ob.get('strength', 0) for ob in patterns['order_blocks'][-3:]])
            quality_factors.append(ob_strength * 0.3)
        
        if patterns.get('break_of_structure'):
            confirmed_bos = [bos for bos in patterns['break_of_structure'] if bos.get('confirmed', False)]
            if confirmed_bos:
                quality_factors.append(0.3)
        
        # Fibonacci confluence
        fib_clusters = patterns.get('fibonacci', {}).get('clusters', [])
        if fib_clusters:
            fib_strength = np.mean([cluster.get('strength', 0) for cluster in fib_clusters])
            quality_factors.append(fib_strength * 0.2)
        
        # Wyckoff phase confirmation
        wyckoff = patterns.get('wyckoff_phases', {})
        if wyckoff.get('phase') in ['accumulation', 'markup']:
            quality_factors.append(wyckoff.get('confidence', 0) * 0.2)
        
        return np.mean(quality_factors) if quality_factors else 0.5
    
    def calculate_multi_tf_score(self, signal: Dict) -> float:
        """Calculate multi-timeframe alignment score (0-1)"""
        mtf_analysis = signal.get('multi_timeframe_analysis', {})
        composite_bias = mtf_analysis.get('composite_bias', 'neutral')
        signal_direction = signal.get('direction', '').lower()
        
        if composite_bias == 'bullish' and signal_direction == 'buy':
            return 1.0
        elif composite_bias == 'bearish' and signal_direction == 'sell':
            return 1.0
        elif composite_bias == 'neutral':
            return 0.5
        else:
            return 0.2
    
    def calculate_sentiment_score(self, signal: Dict) -> float:
        """Calculate sentiment alignment score (0-1)"""
        sentiment = signal.get('sentiment_analysis', {})
        signal_direction = signal.get('direction', '').lower()
        market_sentiment = sentiment.get('overall_sentiment', 'neutral').lower()
        
        if market_sentiment == 'bullish' and signal_direction == 'buy':
            return 1.0
        elif market_sentiment == 'bearish' and signal_direction == 'sell':
            return 1.0
        elif market_sentiment == 'neutral':
            return 0.7
        else:
            return 0.3
    
    def calculate_volume_score(self, signal: Dict) -> float:
        """Calculate volume confirmation score (0-1)"""
        # Placeholder - in production, integrate actual volume analysis
        patterns = signal.get('patterns', {})
        
        # Check if patterns have volume confirmation
        if patterns.get('order_blocks'):
            recent_ob = patterns['order_blocks'][-1] if patterns['order_blocks'] else None
            if recent_ob and recent_ob.get('volume_confirmation', False):
                return 0.8
        
        return 0.5  # Default medium score
    
    def calculate_pattern_score(self, signal: Dict) -> float:
        """Calculate pattern confluence score (0-1)"""
        patterns = signal.get('patterns', {})
        pattern_count = 0
        total_strength = 0
        
        # Count significant patterns
        if patterns.get('order_blocks'):
            strong_obs = [ob for ob in patterns['order_blocks'] if ob.get('strength', 0) > 0.7]
            pattern_count += len(strong_obs)
            total_strength += sum(ob.get('strength', 0) for ob in strong_obs)
        
        if patterns.get('fair_value_gaps'):
            strong_fvgs = [fvg for fvg in patterns['fair_value_gaps'] if fvg.get('strength', 0) > 0.7]
            pattern_count += len(strong_fvgs)
            total_strength += sum(fvg.get('strength', 0) for fvg in strong_fvgs)
        
        if patterns.get('break_of_structure'):
            confirmed_bos = [bos for bos in patterns['break_of_structure'] if bos.get('confirmed', False)]
            pattern_count += len(confirmed_bos)
            total_strength += 0.8 * len(confirmed_bos)  # BOS has high weight
        
        if pattern_count > 0:
            return min(total_strength / pattern_count, 1.0)
        return 0.3
    
    def passes_filters(self, signal: Dict, ranking_score: float, filter_config: SignalFilter) -> bool:
        """Check if signal passes all filters"""
        # Confidence filter
        if signal.get('confidence', 0) < filter_config.min_confidence:
            return False
        
        # Risk-reward filter
        if signal.get('risk_reward_ratio', 1.0) < filter_config.min_rr_ratio:
            return False
        
        # Daily signal limit
        if self.sent_today >= filter_config.max_daily_signals:
            return False
        
        # Instrument filter
        if (filter_config.allowed_instruments and 
            signal.get('symbol') not in filter_config.allowed_instruments):
            return False
        
        # High impact news filter
        if (filter_config.exclude_high_impact_news and 
            self.has_high_impact_news(signal)):
            return False
        
        # Multi-timeframe alignment filter
        if (filter_config.require_multi_tf_alignment and 
            not self.has_multi_tf_alignment(signal)):
            return False
        
        return True
    
    def has_high_impact_news(self, signal: Dict) -> bool:
        """Check if signal has high impact news conflict"""
        sentiment = signal.get('sentiment_analysis', {})
        news_impact = sentiment.get('news_impact', {})
        
        return news_impact.get('high_impact_articles', 0) > 0
    
    def has_multi_tf_alignment(self, signal: Dict) -> bool:
        """Check if signal has multi-timeframe alignment"""
        mtf_analysis = signal.get('multi_timeframe_analysis', {})
        composite_bias = mtf_analysis.get('composite_bias', 'neutral')
        signal_direction = signal.get('direction', '').lower()
        
        return (composite_bias == 'bullish' and signal_direction == 'buy') or \
               (composite_bias == 'bearish' and signal_direction == 'sell')
    
    def determine_priority(self, ranking_score: float) -> SignalPriority:
        """Determine signal priority based on ranking score"""
        if ranking_score >= 80:
            return SignalPriority.HIGH
        elif ranking_score >= 65:
            return SignalPriority.MEDIUM
        else:
            return SignalPriority.LOW
    
    def get_delivery_channels(self, priority: SignalPriority) -> List[str]:
        """Get delivery channels based on priority"""
        base_channels = ['in_app']
        
        if priority == SignalPriority.HIGH:
            return base_channels + ['telegram', 'email']
        elif priority == SignalPriority.MEDIUM:
            return base_channels + ['telegram']
        else:
            return base_channels
    
    def reset_daily_counter_if_needed(self):
        """Reset daily signal counter if it's a new day"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.sent_today = 0
            self.last_reset = today
    
    def save_signal(self, signal: RankedSignal) -> int:
        """Save signal to database and return signal ID"""
        conn = sqlite3.connect('trading_signals.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals (
                symbol, direction, entry_price, stop_loss, take_profit, confidence,
                ranking_score, priority, reasoning, timestamp, expiry, delivery_status,
                delivery_channels, risk_reward_ratio, setup_quality
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.symbol,
            signal.direction,
            signal.entry_price,
            signal.stop_loss,
            json.dumps(signal.take_profit),
            signal.confidence,
            signal.ranking_score,
            signal.priority.value,
            json.dumps(signal.reasoning),
            signal.timestamp,
            signal.expiry,
            signal.delivery_status.value,
            json.dumps(signal.delivery_channels),
            signal.risk_reward_ratio,
            signal.setup_quality
        ))
        
        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return signal_id
    
    def update_delivery_status(self, signal_id: int, status: DeliveryStatus):
        """Update signal delivery status"""
        conn = sqlite3.connect('trading_signals.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE signals 
            SET delivery_status = ?
            WHERE id = ?
        ''', (status.value, signal_id))
        
        conn.commit()
        conn.close()
    
    def get_todays_signals(self) -> List[RankedSignal]:
        """Get today's delivered signals"""
        conn = sqlite3.connect('trading_signals.db')
        cursor = conn.cursor()
        
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        
        cursor.execute('''
            SELECT * FROM signals 
            WHERE timestamp >= ? AND timestamp < ?
            ORDER BY ranking_score DESC
        ''', (today, tomorrow))
        
        rows = cursor.fetchall()
        conn.close()
        
        signals = []
        for row in rows:
            signal = RankedSignal(
                symbol=row[1],
                direction=row[2],
                entry_price=row[3],
                stop_loss=row[4],
                take_profit=json.loads(row[5]),
                confidence=row[6],
                ranking_score=row[7],
                priority=SignalPriority(row[8]),
                reasoning=json.loads(row[9]),
                timestamp=datetime.fromisoformat(row[10]),
                expiry=datetime.fromisoformat(row[11]),
                delivery_status=DeliveryStatus(row[12]),
                delivery_channels=json.loads(row[13]),
                risk_reward_ratio=row[14],
                setup_quality=row[15]
            )
            signals.append(signal)
        
        return signals

class DeliveryEngine:
    """Multi-channel signal delivery engine"""
    
    def __init__(self):
        self.telegram_bot = None
        self.email_service = None
        self.init_delivery_services()
    
    def init_delivery_services(self):
        """Initialize delivery services"""
        # Telegram Bot (you need to set up your bot token)
        try:
            # import telegram
            # self.telegram_bot = telegram.Bot(token='YOUR_TELEGRAM_BOT_TOKEN')
            pass
        except ImportError:
            print("Telegram bot not configured")
        
        # Email service (SMTP configuration)
        # self.email_service = self.setup_smtp()
    
    def deliver_signal(self, signal: RankedSignal) -> Dict[str, str]:
        """Deliver signal through configured channels"""
        delivery_results = {}
        
        for channel in signal.delivery_channels:
            try:
                if channel == 'telegram':
                    result = self.deliver_telegram(signal)
                    delivery_results['telegram'] = result
                elif channel == 'email':
                    result = self.deliver_email(signal)
                    delivery_results['email'] = result
                elif channel == 'in_app':
                    result = self.deliver_in_app(signal)
                    delivery_results['in_app'] = result
                
                # Record delivery attempt
                self.record_delivery_attempt(signal, channel, 'sent')
                
            except Exception as e:
                print(f"Failed to deliver via {channel}: {e}")
                self.record_delivery_attempt(signal, channel, 'failed')
                delivery_results[channel] = f'failed: {str(e)}'
        
        return delivery_results
    
    def deliver_telegram(self, signal: RankedSignal) -> str:
        """Deliver signal via Telegram"""
        message = self.format_signal_message(signal, 'telegram')
        
        # In production, uncomment and configure:
        # if self.telegram_bot:
        #     chat_id = 'YOUR_CHAT_ID'
        #     sent_message = self.telegram_bot.send_message(
        #         chat_id=chat_id,
        #         text=message,
        #         parse_mode='HTML'
        #     )
        #     return f'sent: {sent_message.message_id}'
        
        # For demo, just return success
        return 'sent: demo_mode'
    
    def deliver_email(self, signal: RankedSignal) -> str:
        """Deliver signal via Email"""
        subject = f"Trading Signal: {signal.symbol} {signal.direction}"
        body = self.format_signal_message(signal, 'email')
        
        # In production, implement SMTP sending
        # self.send_email(subject, body)
        
        return 'sent: demo_mode'
    
    def deliver_in_app(self, signal: RankedSignal) -> str:
        """Deliver signal via in-app notification"""
        # This will be handled by the frontend polling
        return 'queued'
    
    def format_signal_message(self, signal: RankedSignal, channel: str) -> str:
        """Format signal message for different channels"""
        if channel == 'telegram':
            return self.format_telegram_message(signal)
        else:
            return self.format_generic_message(signal)
    
    def format_telegram_message(self, signal: RankedSignal) -> str:
        """Format signal message for Telegram"""
        direction_emoji = "🟢" if signal.direction == 'BUY' else "🔴"
        priority_emoji = "🚀" if signal.priority == SignalPriority.HIGH else "⚡" if signal.priority == SignalPriority.MEDIUM else "📊"
        
        message = f"""
{direction_emoji} <b>TRADING SIGNAL</b> {priority_emoji}

<b>Symbol:</b> {signal.symbol}
<b>Direction:</b> {signal.direction}
<b>Priority:</b> {signal.priority.name}

<b>Entry:</b> {signal.entry_price}
<b>Stop Loss:</b> {signal.stop_loss}
<b>Take Profit:</b> {', '.join(map(str, signal.take_profit))}

<b>Confidence:</b> {signal.confidence:.0%}
<b>Ranking Score:</b> {signal.ranking_score:.1f}/100
<b>Risk-Reward:</b> {signal.risk_reward_ratio:.2f}

<b>Key Reasons:</b>
""" + "\n".join([f"• {reason}" for reason in signal.reasoning[:3]]) + f"""

⏰ <i>Expires: {signal.expiry.strftime('%H:%M UTC')}</i>
        """
        
        return message
    
    def format_generic_message(self, signal: RankedSignal) -> str:
        """Format signal message for other channels"""
        return f"""
TRADING SIGNAL - {signal.symbol} {signal.direction}

Entry: {signal.entry_price}
Stop Loss: {signal.stop_loss}
Take Profit: {', '.join(map(str, signal.take_profit))}

Confidence: {signal.confidence:.0%}
Ranking Score: {signal.ranking_score:.1f}/100
Risk-Reward: {signal.risk_reward_ratio:.2f}

Key Reasons:
""" + "\n".join([f"- {reason}" for reason in signal.reasoning[:3]]) + f"""

Expires: {signal.expiry.strftime('%Y-%m-%d %H:%M UTC')}
        """
    
    def record_delivery_attempt(self, signal: RankedSignal, channel: str, status: str):
        """Record delivery attempt in database"""
        conn = sqlite3.connect('trading_signals.db')
        cursor = conn.cursor()
        
        # Get signal ID (you might need to adjust this)
        cursor.execute('SELECT id FROM signals WHERE symbol = ? AND timestamp = ?', 
                      (signal.symbol, signal.timestamp))
        result = cursor.fetchone()
        
        if result:
            signal_id = result[0]
            cursor.execute('''
                INSERT INTO signal_deliveries (signal_id, channel, status, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (signal_id, channel, status, datetime.now()))
        
        conn.commit()
        conn.close()

class ReversalAlertSystem:
    """System for detecting and alerting on potential reversals"""
    
    def __init__(self):
        self.monitored_signals = {}
        self.alert_threshold = 0.02  # 2% move towards SL before hitting it
    
    def monitor_signal(self, signal: RankedSignal):
        """Start monitoring a signal for potential reversal"""
        self.monitored_signals[signal.symbol] = {
            'signal': signal,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'direction': signal.direction,
            'last_price': signal.entry_price,
            'alert_sent': False
        }
    
    def check_reversals(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Check for potential reversals in monitored signals"""
        alerts = []
        
        for symbol, data in self.monitored_signals.items():
            if symbol in current_prices and not data['alert_sent']:
                current_price = current_prices[symbol]
                signal = data['signal']
                
                # Calculate distance to stop loss
                if signal.direction == 'BUY':
                    distance_to_sl = (current_price - signal.stop_loss) / signal.entry_price
                    reversal_risk = (signal.entry_price - current_price) / (signal.entry_price - signal.stop_loss)
                else:  # SELL
                    distance_to_sl = (signal.stop_loss - current_price) / signal.entry_price
                    reversal_risk = (current_price - signal.entry_price) / (signal.stop_loss - signal.entry_price)
                
                # Check if reversal risk is high
                if reversal_risk > self.alert_threshold:
                    alert = self.generate_reversal_alert(signal, current_price, reversal_risk)
                    alerts.append(alert)
                    data['alert_sent'] = True
        
        return alerts
    
    def generate_reversal_alert(self, signal: RankedSignal, current_price: float, risk: float) -> Dict:
        """Generate reversal alert message"""
        return {
            'type': 'REVERSAL_ALERT',
            'symbol': signal.symbol,
            'direction': signal.direction,
            'current_price': current_price,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'risk_level': 'HIGH' if risk > 0.05 else 'MEDIUM',
            'message': f"⚠️ Reversal risk for {signal.symbol}: Price moving towards SL ({risk:.1%} risk)",
            'timestamp': datetime.now(),
            'recommendation': 'Consider early exit or adjust stop loss'
        }