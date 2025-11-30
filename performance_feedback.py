import numpy as np
import pandas as pd
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

class FeedbackType(Enum):
    TRADE_COMPLETE = "trade_complete"
    PARTIAL_RESULT = "partial_result" 
    MISSED_OPPORTUNITY = "missed_opportunity"
    FALSE_SIGNAL = "false_signal"
    MARKET_REGIME_CHANGE = "market_regime_change"
    PERFORMANCE_ANOMALY = "performance_anomaly"

@dataclass
class TradeFeedback:
    feedback_id: str
    timestamp: datetime
    symbol: str
    feedback_type: FeedbackType
    trade_outcome: str  # 'WIN', 'LOSS', 'BREAKEVEN', 'PARTIAL'
    setup_conditions: Dict
    market_conditions: Dict 
    performance_metrics: Dict
    learning_priority: float  # 0.0 to 1.0
    immediate_actions: List[str]
    model_adjustments: Dict

@dataclass
class PartialResult:
    trade_id: str
    timestamp: datetime
    symbol: str
    current_pnl: float
    progress_percentage: float
    risk_adjustment_needed: bool
    exit_recommendation: Optional[str]
    confidence_change: float

@dataclass
class MarketAdaptation:
    symbol: str
    timestamp: datetime
    regime_shift: bool
    volatility_change: float
    liquidity_conditions: str
    adaptation_actions: List[str]
    parameter_adjustments: Dict

class RealTimeFeedbackEngine:
    def __init__(self, learning_engine):
        self.learning_engine = learning_engine
        self.active_feedbacks = []
        self.partial_results = []
        self.market_adaptations = []
        self.feedback_queue = asyncio.Queue()
        self.adaptation_lock = threading.Lock()
        self.performance_alerts = []
        self.init_feedback_system()
    
    def init_feedback_system(self):
        """Initialize real-time feedback system"""
        self.setup_feedback_database()
        self.start_feedback_processor()
        self.start_market_monitoring()
        self.start_performance_alerts()
        
        logging.info("Real-time Feedback Engine initialized")
    
    def setup_feedback_database(self):
        """Setup feedback database tables"""
        conn = sqlite3.connect('feedback_system.db')
        cursor = conn.cursor()
        
        # Trade feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_feedback (
                feedback_id TEXT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                trade_outcome TEXT NOT NULL,
                setup_conditions TEXT NOT NULL,
                market_conditions TEXT NOT NULL,
                performance_metrics TEXT NOT NULL,
                learning_priority REAL NOT NULL,
                immediate_actions TEXT NOT NULL,
                model_adjustments TEXT NOT NULL
            )
        ''')
        
        # Partial results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS partial_results (
                trade_id TEXT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                current_pnl REAL NOT NULL,
                progress_percentage REAL NOT NULL,
                risk_adjustment_needed BOOLEAN NOT NULL,
                exit_recommendation TEXT,
                confidence_change REAL NOT NULL
            )
        ''')
        
        # Market adaptations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_adaptations (
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                regime_shift BOOLEAN NOT NULL,
                volatility_change REAL NOT NULL,
                liquidity_conditions TEXT NOT NULL,
                adaptation_actions TEXT NOT NULL,
                parameter_adjustments TEXT NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            )
        ''')
        
        # Performance alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_alerts (
                alert_id TEXT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                symbol TEXT,
                message TEXT NOT NULL,
                actions_taken TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_feedback_processor(self):
        """Start background feedback processing"""
        def process_feedback_queue():
            while True:
                try:
                    # Process feedback items from queue
                    if not self.feedback_queue.empty():
                        feedback_item = self.feedback_queue.get_nowait()
                        self.process_feedback_item(feedback_item)
                    
                    # Process partial results
                    self.process_partial_results()
                    
                    # Check for market adaptations
                    self.check_market_adaptations()
                    
                    threading.Event().wait(1)  # Process every second
                    
                except Exception as e:
                    logging.error(f"Error in feedback processor: {e}")
                    threading.Event().wait(5)  # Wait 5 seconds on error
        
        processor_thread = threading.Thread(target=process_feedback_queue, daemon=True)
        processor_thread.start()
    
    def start_market_monitoring(self):
        """Start real-time market condition monitoring"""
        def monitor_market_conditions():
            while True:
                try:
                    # Monitor for regime changes
                    self.detect_market_regime_changes()
                    
                    # Monitor volatility shifts
                    self.monitor_volatility_changes()
                    
                    # Monitor liquidity conditions
                    self.monitor_liquidity_conditions()
                    
                    threading.Event().wait(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logging.error(f"Error in market monitoring: {e}")
                    threading.Event().wait(60)  # Wait 1 minute on error
        
        market_thread = threading.Thread(target=monitor_market_conditions, daemon=True)
        market_thread.start()
    
    def start_performance_alerts(self):
        """Start performance anomaly detection"""
        def monitor_performance():
            while True:
                try:
                    # Check for performance anomalies
                    self.detect_performance_anomalies()
                    
                    # Check for strategy degradation
                    self.detect_strategy_degradation()
                    
                    # Check for risk limit breaches
                    self.check_risk_limits()
                    
                    threading.Event().wait(60)  # Check every minute
                    
                except Exception as e:
                    logging.error(f"Error in performance monitoring: {e}")
                    threading.Event().wait(300)  # Wait 5 minutes on error
        
        performance_thread = threading.Thread(target=monitor_performance, daemon=True)
        performance_thread.start()
    
    async def submit_trade_feedback(self, symbol: str, feedback_type: FeedbackType, 
                                  trade_outcome: str, setup_conditions: Dict,
                                  market_conditions: Dict, performance_metrics: Dict):
        """Submit trade feedback for immediate processing"""
        try:
            feedback_id = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Calculate learning priority
            learning_priority = self.calculate_learning_priority(
                feedback_type, trade_outcome, performance_metrics
            )
            
            # Determine immediate actions
            immediate_actions = self.determine_immediate_actions(
                feedback_type, trade_outcome, performance_metrics
            )
            
            # Generate model adjustments
            model_adjustments = self.generate_model_adjustments(
                feedback_type, trade_outcome, setup_conditions, performance_metrics
            )
            
            feedback = TradeFeedback(
                feedback_id=feedback_id,
                timestamp=datetime.now(),
                symbol=symbol,
                feedback_type=feedback_type,
                trade_outcome=trade_outcome,
                setup_conditions=setup_conditions,
                market_conditions=market_conditions,
                performance_metrics=performance_metrics,
                learning_priority=learning_priority,
                immediate_actions=immediate_actions,
                model_adjustments=model_adjustments
            )
            
            # Add to processing queue
            await self.feedback_queue.put(feedback)
            
            logging.info(f"Submitted trade feedback: {feedback_id}")
            
            return feedback
            
        except Exception as e:
            logging.error(f"Error submitting trade feedback: {e}")
            return None
    
    def calculate_learning_priority(self, feedback_type: FeedbackType, trade_outcome: str,
                                 performance_metrics: Dict) -> float:
        """Calculate learning priority for feedback"""
        priority = 0.5  # Base priority
        
        # Adjust based on feedback type
        if feedback_type == FeedbackType.TRADE_COMPLETE:
            priority += 0.3
        elif feedback_type == FeedbackType.MISSED_OPPORTUNITY:
            priority += 0.2
        elif feedback_type == FeedbackType.FALSE_SIGNAL:
            priority += 0.4
        
        # Adjust based on outcome
        if trade_outcome == 'LOSS':
            priority += 0.3
        elif trade_outcome == 'WIN':
            priority -= 0.1
        
        # Adjust based on performance impact
        pnl = performance_metrics.get('pnl', 0)
        if abs(pnl) > performance_metrics.get('avg_pnl', 0) * 2:
            priority += 0.2
        
        return max(0.0, min(1.0, priority))
    
    def determine_immediate_actions(self, feedback_type: FeedbackType, trade_outcome: str,
                                 performance_metrics: Dict) -> List[str]:
        """Determine immediate actions from feedback"""
        actions = []
        
        if feedback_type == FeedbackType.TRADE_COMPLETE:
            if trade_outcome == 'LOSS':
                actions.extend([
                    "Review stop-loss placement",
                    "Check setup validation criteria",
                    "Verify market condition alignment"
                ])
            elif trade_outcome == 'WIN':
                actions.extend([
                    "Reinforce successful pattern recognition",
                    "Update win probability estimates",
                    "Optimize position sizing for similar setups"
                ])
        
        elif feedback_type == FeedbackType.PARTIAL_RESULT:
            actions.extend([
                "Adjust risk management for open position",
                "Update trailing stop logic",
                "Re-evaluate profit targets"
            ])
        
        elif feedback_type == FeedbackType.MISSED_OPPORTUNITY:
            actions.extend([
                "Review signal filtering criteria",
                "Adjust confidence thresholds",
                "Update pattern recognition sensitivity"
            ])
        
        elif feedback_type == FeedbackType.FALSE_SIGNAL:
            actions.extend([
                "Increase validation requirements",
                "Add additional confirmation criteria",
                "Reduce position size for similar signals"
            ])
        
        # Add risk management actions for large moves
        if performance_metrics.get('pnl_impact', 0) > 0.1:  # 10% impact
            actions.append("Immediate risk review required")
        
        return actions
    
    def generate_model_adjustments(self, feedback_type: FeedbackType, trade_outcome: str,
                                setup_conditions: Dict, performance_metrics: Dict) -> Dict:
        """Generate immediate model adjustments from feedback"""
        adjustments = {}
        
        confidence = setup_conditions.get('confidence', 0.5)
        
        if feedback_type == FeedbackType.TRADE_COMPLETE:
            if trade_outcome == 'LOSS' and confidence > 0.7:
                adjustments['confidence_threshold'] = -0.05
                adjustments['validation_requirements'] = 'increase'
            elif trade_outcome == 'WIN' and confidence < 0.6:
                adjustments['confidence_threshold'] = 0.02
                adjustments['validation_requirements'] = 'decrease'
        
        elif feedback_type == FeedbackType.MISSED_OPPORTUNITY:
            adjustments['signal_sensitivity'] = 0.05
            adjustments['pattern_recognition_threshold'] = -0.03
        
        elif feedback_type == FeedbackType.FALSE_SIGNAL:
            adjustments['signal_sensitivity'] = -0.08
            adjustments['validation_requirements'] = 'increase'
            adjustments['pattern_recognition_threshold'] = 0.05
        
        # Adjust risk parameters based on performance
        pnl = performance_metrics.get('pnl', 0)
        if abs(pnl) > performance_metrics.get('avg_pnl', 0) * 3:
            adjustments['position_size_multiplier'] = -0.1
        
        return adjustments
    
    def process_feedback_item(self, feedback: TradeFeedback):
        """Process individual feedback item"""
        try:
            # Apply immediate model adjustments
            self.apply_model_adjustments(feedback.model_adjustments)
            
            # Trigger learning episode if high priority
            if feedback.learning_priority > 0.7:
                self.trigger_immediate_learning(feedback)
            
            # Send alerts if critical
            if feedback.learning_priority > 0.9:
                self.send_critical_alert(feedback)
            
            # Save feedback to database
            self.save_trade_feedback(feedback)
            
            logging.info(f"Processed feedback: {feedback.feedback_id}")
            
        except Exception as e:
            logging.error(f"Error processing feedback {feedback.feedback_id}: {e}")
    
    def apply_model_adjustments(self, adjustments: Dict):
        """Apply immediate model adjustments"""
        with self.adaptation_lock:
            # Update adaptive parameters in learning engine
            for param_name, adjustment in adjustments.items():
                if param_name in self.learning_engine.adaptive_parameters:
                    param = self.learning_engine.adaptive_parameters[param_name]
                    
                    # Apply adjustment
                    if isinstance(adjustment, (int, float)):
                        new_value = param.current_value + adjustment
                        optimal_min, optimal_max = param.optimal_range
                        new_value = max(optimal_min, min(optimal_max, new_value))
                        
                        adjustment_record = {
                            'timestamp': datetime.now().isoformat(),
                            'old_value': param.current_value,
                            'new_value': new_value,
                            'adjustment': adjustment,
                            'reason': 'real_time_feedback'
                        }
                        
                        param.current_value = new_value
                        param.adjustment_history.append(adjustment_record)
                        param.last_updated = datetime.now()
            
            # Save updated parameters
            self.learning_engine.save_adaptive_parameters()
    
    def trigger_immediate_learning(self, feedback: TradeFeedback):
        """Trigger immediate learning from high-priority feedback"""
        try:
            # Create learning episode from feedback
            episode = self.learning_engine.record_learning_episode(
                trade_outcome=feedback.trade_outcome,
                setup_conditions=feedback.setup_conditions,
                market_conditions=feedback.market_conditions,
                performance_metrics=feedback.performance_metrics
            )
            
            # Force immediate retraining if critical
            if feedback.learning_priority > 0.95:
                logging.warning("Critical feedback detected - forcing immediate retraining")
                self.learning_engine.retrain_models()
            
            return episode
            
        except Exception as e:
            logging.error(f"Error triggering immediate learning: {e}")
    
    def send_critical_alert(self, feedback: TradeFeedback):
        """Send critical performance alert"""
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        alert = {
            'alert_id': alert_id,
            'timestamp': datetime.now(),
            'alert_type': 'critical_feedback',
            'severity': 'high',
            'symbol': feedback.symbol,
            'message': f"Critical feedback received: {feedback.feedback_type.value}",
            'actions_taken': feedback.immediate_actions,
            'resolved': False
        }
        
        self.performance_alerts.append(alert)
        self.save_performance_alert(alert)
        
        # Could also send email/notification here
        logging.warning(f"CRITICAL ALERT: {alert['message']}")
    
    def submit_partial_result(self, trade_id: str, symbol: str, current_pnl: float,
                            progress_percentage: float, market_conditions: Dict):
        """Submit partial trade result for real-time adjustment"""
        try:
            # Calculate risk adjustment need
            risk_adjustment = self.assess_risk_adjustment(
                current_pnl, progress_percentage, market_conditions
            )
            
            # Generate exit recommendation
            exit_recommendation = self.generate_exit_recommendation(
                current_pnl, progress_percentage, market_conditions
            )
            
            # Calculate confidence change
            confidence_change = self.calculate_confidence_change(
                current_pnl, progress_percentage
            )
            
            partial_result = PartialResult(
                trade_id=trade_id,
                timestamp=datetime.now(),
                symbol=symbol,
                current_pnl=current_pnl,
                progress_percentage=progress_percentage,
                risk_adjustment_needed=risk_adjustment,
                exit_recommendation=exit_recommendation,
                confidence_change=confidence_change
            )
            
            self.partial_results.append(partial_result)
            self.save_partial_result(partial_result)
            
            # Apply immediate adjustments if needed
            if risk_adjustment:
                self.apply_risk_adjustments(symbol, current_pnl, progress_percentage)
            
            return partial_result
            
        except Exception as e:
            logging.error(f"Error submitting partial result: {e}")
            return None
    
    def assess_risk_adjustment(self, current_pnl: float, progress: float, 
                             market_conditions: Dict) -> bool:
        """Assess if risk adjustment is needed"""
        # Check if PnL is outside expected range
        if abs(current_pnl) > 0.05:  # 5% move
            return True
        
        # Check if progress is stalled
        if progress > 0.5 and abs(current_pnl) < 0.01:  # Halfway but little movement
            return True
        
        # Check market condition changes
        volatility = market_conditions.get('volatility', 'medium')
        if volatility == 'high' and abs(current_pnl) > 0.03:
            return True
        
        return False
    
    def generate_exit_recommendation(self, current_pnl: float, progress: float,
                                  market_conditions: Dict) -> Optional[str]:
        """Generate exit recommendation for partial result"""
        if current_pnl >= 0.02 and progress > 0.7:  # 2% profit, 70% progress
            return "Consider taking partial profits"
        
        if current_pnl <= -0.03:  # 3% loss
            return "Review stop-loss placement"
        
        if progress > 0.9 and abs(current_pnl) < 0.01:  # Near end, little movement
            return "Consider early exit - low momentum"
        
        return None
    
    def calculate_confidence_change(self, current_pnl: float, progress: float) -> float:
        """Calculate confidence change based on partial results"""
        confidence_change = 0.0
        
        # Positive PnL increases confidence
        if current_pnl > 0:
            confidence_change += min(current_pnl * 10, 0.1)  # Max 10% increase
        
        # Negative PnL decreases confidence
        if current_pnl < 0:
            confidence_change -= min(abs(current_pnl) * 15, 0.15)  # Max 15% decrease
        
        # Good progress increases confidence
        if progress > 0.7 and current_pnl >= 0:
            confidence_change += 0.05
        
        return confidence_change
    
    def apply_risk_adjustments(self, symbol: str, current_pnl: float, progress: float):
        """Apply real-time risk adjustments"""
        adjustments = {}
        
        if current_pnl < -0.03:  # 3% loss
            adjustments['position_size_multiplier'] = -0.15
            adjustments['stop_loss_adjustment'] = 0.05  # Tighten stops
        
        elif current_pnl > 0.04:  # 4% profit
            adjustments['trailing_stop_activation'] = 0.02  # Activate trailing stops
        
        # Apply adjustments to learning engine parameters
        for param_name, adjustment in adjustments.items():
            if param_name in self.learning_engine.adaptive_parameters:
                param = self.learning_engine.adaptive_parameters[param_name]
                new_value = param.current_value + adjustment
                optimal_min, optimal_max = param.optimal_range
                new_value = max(optimal_min, min(optimal_max, new_value))
                
                param.current_value = new_value
                param.last_updated = datetime.now()
        
        self.learning_engine.save_adaptive_parameters()
    
    def process_partial_results(self):
        """Process accumulated partial results"""
        try:
            recent_results = [r for r in self.partial_results 
                            if datetime.now() - r.timestamp < timedelta(hours=1)]
            
            for result in recent_results:
                # Update confidence scores for similar setups
                if abs(result.confidence_change) > 0.05:
                    self.update_confidence_scores(result)
                
                # Trigger learning if significant deviation
                if abs(result.current_pnl) > 0.04:
                    self.trigger_partial_learning(result)
            
            # Clear old results
            self.partial_results = [r for r in self.partial_results 
                                  if datetime.now() - r.timestamp < timedelta(hours=24)]
            
        except Exception as e:
            logging.error(f"Error processing partial results: {e}")
    
    def update_confidence_scores(self, result: PartialResult):
        """Update confidence scores based on partial results"""
        # This would update the AI model's confidence scoring
        # based on how trades are progressing in real-time
        logging.info(f"Updating confidence scores for {result.symbol}: {result.confidence_change}")
    
    def trigger_partial_learning(self, result: PartialResult):
        """Trigger learning from significant partial results"""
        # Create partial learning episode
        partial_episode = {
            'trade_id': result.trade_id,
            'symbol': result.symbol,
            'partial_pnl': result.current_pnl,
            'progress': result.progress_percentage,
            'confidence_change': result.confidence_change,
            'timestamp': datetime.now()
        }
        
        logging.info(f"Partial learning triggered: {partial_episode}")
    
    def detect_market_regime_changes(self):
        """Detect changes in market regimes"""
        # This would analyze market data for regime shifts
        # For now, we'll simulate detection
        symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'BTC/USDT']
        
        for symbol in symbols:
            # Simulate regime detection (in production, use actual market analysis)
            regime_shift = np.random.random() < 0.05  # 5% chance of regime shift
            
            if regime_shift:
                adaptation = MarketAdaptation(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    regime_shift=True,
                    volatility_change=np.random.uniform(-0.2, 0.2),
                    liquidity_conditions='changing',
                    adaptation_actions=[
                        "Adjust volatility parameters",
                        "Update risk management rules",
                        "Review position sizing"
                    ],
                    parameter_adjustments={
                        'volatility_multiplier': np.random.uniform(0.8, 1.2),
                        'position_size_limit': np.random.uniform(0.5, 1.0)
                    }
                )
                
                self.market_adaptations.append(adaptation)
                self.save_market_adaptation(adaptation)
                self.apply_market_adaptation(adaptation)
    
    def monitor_volatility_changes(self):
        """Monitor for significant volatility changes"""
        # This would track volatility indicators
        # For now, we'll simulate monitoring
        pass
    
    def monitor_liquidity_conditions(self):
        """Monitor market liquidity conditions"""
        # This would track liquidity metrics
        # For now, we'll simulate monitoring
        pass
    
    def apply_market_adaptation(self, adaptation: MarketAdaptation):
        """Apply market regime adaptations"""
        with self.adaptation_lock:
            # Update trading parameters for the symbol
            logging.info(f"Applying market adaptation for {adaptation.symbol}")
            
            # Adjust volatility-based parameters
            if 'volatility_multiplier' in adaptation.parameter_adjustments:
                # This would update the AI model's volatility handling
                pass
            
            # Apply to learning engine
            self.learning_engine.adaptive_parameters['position_size_multiplier'].current_value *= \
                adaptation.parameter_adjustments.get('position_size_limit', 1.0)
    
    def detect_performance_anomalies(self):
        """Detect performance anomalies and degradation"""
        try:
            # Analyze recent performance
            recent_feedbacks = [f for f in self.active_feedbacks 
                              if datetime.now() - f.timestamp < timedelta(hours=24)]
            
            if len(recent_feedbacks) < 10:
                return
            
            # Calculate win rate
            win_rate = sum(1 for f in recent_feedbacks if f.trade_outcome == 'WIN') / len(recent_feedbacks)
            
            # Check for performance degradation
            if win_rate < 0.4:  # 40% win rate threshold
                self.create_performance_alert(
                    alert_type='performance_degradation',
                    severity='medium',
                    message=f"Performance degradation detected: {win_rate:.1%} win rate",
                    actions_taken=['Increase validation criteria', 'Reduce position sizes']
                )
            
            # Check for anomaly in PnL distribution
            pnls = [f.performance_metrics.get('pnl', 0) for f in recent_feedbacks]
            if len(pnls) >= 5:
                pnl_std = np.std(pnls)
                if pnl_std > 0.1:  # High PnL volatility
                    self.create_performance_alert(
                        alert_type='high_pnl_volatility',
                        severity='low',
                        message="High PnL volatility detected",
                        actions_taken=['Review risk management', 'Check position sizing']
                    )
                    
        except Exception as e:
            logging.error(f"Error detecting performance anomalies: {e}")
    
    def detect_strategy_degradation(self):
        """Detect strategy performance degradation"""
        # This would analyze strategy-specific metrics
        # For now, we'll simulate detection
        pass
    
    def check_risk_limits(self):
        """Check for risk limit breaches"""
        # This would monitor risk metrics against limits
        # For now, we'll simulate checking
        pass
    
    def create_performance_alert(self, alert_type: str, severity: str, 
                               message: str, actions_taken: List[str]):
        """Create performance alert"""
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        alert = {
            'alert_id': alert_id,
            'timestamp': datetime.now(),
            'alert_type': alert_type,
            'severity': severity,
            'symbol': None,
            'message': message,
            'actions_taken': actions_taken,
            'resolved': False
        }
        
        self.performance_alerts.append(alert)
        self.save_performance_alert(alert)
        
        logging.warning(f"PERFORMANCE ALERT ({severity}): {message}")
    
    # Database operations
    def save_trade_feedback(self, feedback: TradeFeedback):
        """Save trade feedback to database"""
        try:
            conn = sqlite3.connect('feedback_system.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO trade_feedback 
                (feedback_id, timestamp, symbol, feedback_type, trade_outcome,
                 setup_conditions, market_conditions, performance_metrics,
                 learning_priority, immediate_actions, model_adjustments)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.feedback_id,
                feedback.timestamp.isoformat(),
                feedback.symbol,
                feedback.feedback_type.value,
                feedback.trade_outcome,
                json.dumps(feedback.setup_conditions),
                json.dumps(feedback.market_conditions),
                json.dumps(feedback.performance_metrics),
                feedback.learning_priority,
                json.dumps(feedback.immediate_actions),
                json.dumps(feedback.model_adjustments)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error saving trade feedback: {e}")
    
    def save_partial_result(self, result: PartialResult):
        """Save partial result to database"""
        try:
            conn = sqlite3.connect('feedback_system.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO partial_results 
                (trade_id, timestamp, symbol, current_pnl, progress_percentage,
                 risk_adjustment_needed, exit_recommendation, confidence_change)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.trade_id,
                result.timestamp.isoformat(),
                result.symbol,
                result.current_pnl,
                result.progress_percentage,
                result.risk_adjustment_needed,
                result.exit_recommendation,
                result.confidence_change
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error saving partial result: {e}")
    
    def save_market_adaptation(self, adaptation: MarketAdaptation):
        """Save market adaptation to database"""
        try:
            conn = sqlite3.connect('feedback_system.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO market_adaptations 
                (symbol, timestamp, regime_shift, volatility_change,
                 liquidity_conditions, adaptation_actions, parameter_adjustments)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                adaptation.symbol,
                adaptation.timestamp.isoformat(),
                adaptation.regime_shift,
                adaptation.volatility_change,
                adaptation.liquidity_conditions,
                json.dumps(adaptation.adaptation_actions),
                json.dumps(adaptation.parameter_adjustments)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error saving market adaptation: {e}")
    
    def save_performance_alert(self, alert: Dict):
        """Save performance alert to database"""
        try:
            conn = sqlite3.connect('feedback_system.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO performance_alerts 
                (alert_id, timestamp, alert_type, severity, symbol, message, actions_taken, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert['alert_id'],
                alert['timestamp'].isoformat(),
                alert['alert_type'],
                alert['severity'],
                alert['symbol'],
                alert['message'],
                json.dumps(alert['actions_taken']),
                alert['resolved']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error saving performance alert: {e}")
    
    def get_feedback_summary(self) -> Dict:
        """Get real-time feedback system summary"""
        return {
            'active_feedbacks': len(self.active_feedbacks),
            'partial_results': len(self.partial_results),
            'market_adaptations': len(self.market_adaptations),
            'performance_alerts': len([a for a in self.performance_alerts if not a['resolved']]),
            'queue_size': self.feedback_queue.qsize(),
            'learning_priority_avg': np.mean([f.learning_priority for f in self.active_feedbacks]) if self.active_feedbacks else 0,
            'recent_actions': self.get_recent_actions()
        }
    
    def get_recent_actions(self) -> List[Dict]:
        """Get recent system actions and adjustments"""
        recent_actions = []
        
        # Get recent parameter adjustments
        for param_name, param in self.learning_engine.adaptive_parameters.items():
            if param.adjustment_history:
                latest = param.adjustment_history[-1]
                if datetime.now() - datetime.fromisoformat(latest['timestamp']) < timedelta(hours=1):
                    recent_actions.append({
                        'type': 'parameter_adjustment',
                        'parameter': param_name,
                        'adjustment': latest['adjustment'],
                        'timestamp': latest['timestamp']
                    })
        
        # Get recent alerts
        recent_alerts = [a for a in self.performance_alerts 
                        if datetime.now() - a['timestamp'] < timedelta(hours=1)]
        for alert in recent_alerts:
            recent_actions.append({
                'type': 'performance_alert',
                'alert_type': alert['alert_type'],
                'severity': alert['severity'],
                'timestamp': alert['timestamp'].isoformat()
            })
        
        return sorted(recent_actions, key=lambda x: x['timestamp'], reverse=True)[:10]