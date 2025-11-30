import numpy as np
import pandas as pd
import sqlite3
import json
import pickle
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

@dataclass
class LearningEpisode:
    episode_id: str
    timestamp: datetime
    trade_outcome: str  # 'WIN', 'LOSS', 'BREAKEVEN'
    setup_conditions: Dict
    market_conditions: Dict
    performance_metrics: Dict
    learning_insights: Dict
    model_updates: Dict

@dataclass
class ModelPerformance:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_date: datetime
    sample_size: int
    improvement: float

@dataclass
class AdaptiveParameter:
    parameter_name: str
    current_value: float
    optimal_range: Tuple[float, float]
    adjustment_history: List[Dict]
    last_updated: datetime

class ContinuousLearningEngine:
    def __init__(self):
        self.learning_episodes = []
        self.model_performance = {}
        self.adaptive_parameters = {}
        self.performance_threshold = 0.65  # Minimum acceptable performance
        self.learning_rate = 0.1  # How quickly to adapt
        self.init_learning_system()
    
    def init_learning_system(self):
        """Initialize continuous learning system"""
        self.setup_learning_database()
        self.load_historical_episodes()
        self.initialize_adaptive_parameters()
        self.setup_performance_tracking()
        
        logging.info("Continuous Learning Engine initialized")
    
    def setup_learning_database(self):
        """Setup learning database"""
        conn = sqlite3.connect('learning_system.db')
        cursor = conn.cursor()
        
        # Learning episodes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_episodes (
                episode_id TEXT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                trade_outcome TEXT NOT NULL,
                setup_conditions TEXT NOT NULL,
                market_conditions TEXT NOT NULL,
                performance_metrics TEXT NOT NULL,
                learning_insights TEXT NOT NULL,
                model_updates TEXT NOT NULL
            )
        ''')
        
        # Model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                model_name TEXT PRIMARY KEY,
                accuracy REAL NOT NULL,
                precision REAL NOT NULL,
                recall REAL NOT NULL,
                f1_score REAL NOT NULL,
                training_date DATETIME NOT NULL,
                sample_size INTEGER NOT NULL,
                improvement REAL DEFAULT 0.0
            )
        ''')
        
        # Adaptive parameters table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adaptive_parameters (
                parameter_name TEXT PRIMARY KEY,
                current_value REAL NOT NULL,
                optimal_min REAL NOT NULL,
                optimal_max REAL NOT NULL,
                adjustment_history TEXT NOT NULL,
                last_updated DATETIME NOT NULL
            )
        ''')
        
        # Performance trends table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_trends (
                date DATE PRIMARY KEY,
                win_rate REAL NOT NULL,
                avg_rr_ratio REAL NOT NULL,
                total_trades INTEGER NOT NULL,
                learning_effectiveness REAL NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_historical_episodes(self):
        """Load historical learning episodes"""
        try:
            conn = sqlite3.connect('learning_system.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM learning_episodes ORDER BY timestamp DESC LIMIT 1000')
            episodes_data = cursor.fetchall()
            
            for episode_data in episodes_data:
                episode = LearningEpisode(
                    episode_id=episode_data[0],
                    timestamp=datetime.fromisoformat(episode_data[1]),
                    trade_outcome=episode_data[2],
                    setup_conditions=json.loads(episode_data[3]),
                    market_conditions=json.loads(episode_data[4]),
                    performance_metrics=json.loads(episode_data[5]),
                    learning_insights=json.loads(episode_data[6]),
                    model_updates=json.loads(episode_data[7])
                )
                self.learning_episodes.append(episode)
            
            conn.close()
            logging.info(f"Loaded {len(self.learning_episodes)} historical episodes")
            
        except Exception as e:
            logging.error(f"Error loading historical episodes: {e}")
    
    def initialize_adaptive_parameters(self):
        """Initialize adaptive parameters with optimal ranges"""
        self.adaptive_parameters = {
            'confidence_threshold': AdaptiveParameter(
                parameter_name='confidence_threshold',
                current_value=0.65,
                optimal_range=(0.60, 0.75),
                adjustment_history=[],
                last_updated=datetime.now()
            ),
            'risk_reward_minimum': AdaptiveParameter(
                parameter_name='risk_reward_minimum',
                current_value=1.5,
                optimal_range=(1.3, 2.0),
                adjustment_history=[],
                last_updated=datetime.now()
            ),
            'position_size_multiplier': AdaptiveParameter(
                parameter_name='position_size_multiplier',
                current_value=1.0,
                optimal_range=(0.5, 2.0),
                adjustment_history=[],
                last_updated=datetime.now()
            ),
            'stop_loss_adjustment': AdaptiveParameter(
                parameter_name='stop_loss_adjustment',
                current_value=1.0,
                optimal_range=(0.8, 1.5),
                adjustment_history=[],
                last_updated=datetime.now()
            ),
            'sentiment_weight': AdaptiveParameter(
                parameter_name='sentiment_weight',
                current_value=0.1,
                optimal_range=(0.05, 0.2),
                adjustment_history=[],
                last_updated=datetime.now()
            )
        }
        
        # Load from database if exists
        self.load_adaptive_parameters()
    
    def load_adaptive_parameters(self):
        """Load adaptive parameters from database"""
        try:
            conn = sqlite3.connect('learning_system.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM adaptive_parameters')
            parameters_data = cursor.fetchall()
            
            for param_data in parameters_data:
                param_name = param_data[0]
                if param_name in self.adaptive_parameters:
                    self.adaptive_parameters[param_name].current_value = param_data[1]
                    self.adaptive_parameters[param_name].optimal_range = (param_data[2], param_data[3])
                    self.adaptive_parameters[param_name].adjustment_history = json.loads(param_data[4])
                    self.adaptive_parameters[param_name].last_updated = datetime.fromisoformat(param_data[5])
            
            conn.close()
            
        except Exception as e:
            logging.error(f"Error loading adaptive parameters: {e}")
    
    def setup_performance_tracking(self):
        """Setup performance tracking system"""
        # Initialize model performance tracking
        self.model_performance = {
            'random_forest': ModelPerformance(
                model_name='random_forest',
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                training_date=datetime.now(),
                sample_size=0,
                improvement=0.0
            ),
            'lstm_network': ModelPerformance(
                model_name='lstm_network',
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                training_date=datetime.now(),
                sample_size=0,
                improvement=0.0
            ),
            'gradient_boosting': ModelPerformance(
                model_name='gradient_boosting',
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                training_date=datetime.now(),
                sample_size=0,
                improvement=0.0
            )
        }
        
        self.load_model_performance()
    
    def load_model_performance(self):
        """Load model performance from database"""
        try:
            conn = sqlite3.connect('learning_system.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM model_performance')
            performance_data = cursor.fetchall()
            
            for perf_data in performance_data:
                model_name = perf_data[0]
                if model_name in self.model_performance:
                    self.model_performance[model_name] = ModelPerformance(
                        model_name=model_name,
                        accuracy=perf_data[1],
                        precision=perf_data[2],
                        recall=perf_data[3],
                        f1_score=perf_data[4],
                        training_date=datetime.fromisoformat(perf_data[5]),
                        sample_size=perf_data[6],
                        improvement=perf_data[7]
                    )
            
            conn.close()
            
        except Exception as e:
            logging.error(f"Error loading model performance: {e}")
    
    def record_learning_episode(self, trade_outcome: str, setup_conditions: Dict, 
                              market_conditions: Dict, performance_metrics: Dict):
        """Record a learning episode from trade outcome"""
        try:
            episode_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.learning_episodes)}"
            
            # Generate learning insights
            learning_insights = self.analyze_episode_insights(
                trade_outcome, setup_conditions, market_conditions, performance_metrics
            )
            
            # Determine model updates
            model_updates = self.determine_model_updates(learning_insights)
            
            # Create learning episode
            episode = LearningEpisode(
                episode_id=episode_id,
                timestamp=datetime.now(),
                trade_outcome=trade_outcome,
                setup_conditions=setup_conditions,
                market_conditions=market_conditions,
                performance_metrics=performance_metrics,
                learning_insights=learning_insights,
                model_updates=model_updates
            )
            
            self.learning_episodes.append(episode)
            
            # Save to database
            self.save_learning_episode(episode)
            
            # Update adaptive parameters
            self.update_adaptive_parameters(episode)
            
            # Check if retraining is needed
            if self.should_retrain_models():
                self.retrain_models()
            
            logging.info(f"Recorded learning episode: {episode_id}")
            
            return episode
            
        except Exception as e:
            logging.error(f"Error recording learning episode: {e}")
            return None
    
    def analyze_episode_insights(self, trade_outcome: str, setup_conditions: Dict,
                               market_conditions: Dict, performance_metrics: Dict) -> Dict:
        """Analyze episode for learning insights"""
        insights = {
            'setup_effectiveness': 0.0,
            'market_condition_impact': 0.0,
            'parameter_effectiveness': {},
            'improvement_opportunities': [],
            'success_factors': []
        }
        
        # Analyze setup effectiveness
        setup_effectiveness = self.calculate_setup_effectiveness(
            trade_outcome, setup_conditions
        )
        insights['setup_effectiveness'] = setup_effectiveness
        
        # Analyze market condition impact
        market_impact = self.analyze_market_impact(
            trade_outcome, market_conditions
        )
        insights['market_condition_impact'] = market_impact
        
        # Analyze parameter effectiveness
        param_effectiveness = self.analyze_parameter_effectiveness(
            trade_outcome, setup_conditions
        )
        insights['parameter_effectiveness'] = param_effectiveness
        
        # Identify improvement opportunities
        improvements = self.identify_improvement_opportunities(
            trade_outcome, setup_conditions, performance_metrics
        )
        insights['improvement_opportunities'] = improvements
        
        # Identify success factors
        success_factors = self.identify_success_factors(
            trade_outcome, setup_conditions
        )
        insights['success_factors'] = success_factors
        
        return insights
    
    def calculate_setup_effectiveness(self, trade_outcome: str, setup_conditions: Dict) -> float:
        """Calculate effectiveness of trading setup"""
        effectiveness = 0.5  # Base effectiveness
        
        # Adjust based on outcome
        if trade_outcome == 'WIN':
            effectiveness += 0.3
        elif trade_outcome == 'LOSS':
            effectiveness -= 0.2
        
        # Adjust based on confidence
        confidence = setup_conditions.get('confidence', 0.5)
        effectiveness *= confidence
        
        # Adjust based on risk-reward
        rr_ratio = setup_conditions.get('risk_reward_ratio', 1.0)
        if rr_ratio > 2.0:
            effectiveness += 0.1
        elif rr_ratio < 1.0:
            effectiveness -= 0.1
        
        return max(0.0, min(1.0, effectiveness))
    
    def analyze_market_impact(self, trade_outcome: str, market_conditions: Dict) -> float:
        """Analyze impact of market conditions on trade outcome"""
        impact = 0.0
        
        # Analyze volatility impact
        volatility = market_conditions.get('volatility', 'medium')
        if volatility == 'high' and trade_outcome == 'WIN':
            impact += 0.2
        elif volatility == 'high' and trade_outcome == 'LOSS':
            impact -= 0.1
        
        # Analyze trend impact
        trend = market_conditions.get('trend', 'neutral')
        if trend == 'strong' and trade_outcome == 'WIN':
            impact += 0.15
        
        # Analyze sentiment impact
        sentiment = market_conditions.get('sentiment', 'neutral')
        if sentiment == 'bullish' and trade_outcome == 'WIN':
            impact += 0.1
        
        return impact
    
    def analyze_parameter_effectiveness(self, trade_outcome: str, setup_conditions: Dict) -> Dict:
        """Analyze effectiveness of different parameters"""
        effectiveness = {}
        
        # Confidence threshold effectiveness
        confidence = setup_conditions.get('confidence', 0.5)
        if trade_outcome == 'WIN' and confidence > 0.7:
            effectiveness['confidence_threshold'] = 'effective'
        elif trade_outcome == 'LOSS' and confidence > 0.7:
            effectiveness['confidence_threshold'] = 'overconfident'
        else:
            effectiveness['confidence_threshold'] = 'appropriate'
        
        # Risk-reward effectiveness
        rr_ratio = setup_conditions.get('risk_reward_ratio', 1.0)
        if trade_outcome == 'WIN' and rr_ratio > 1.5:
            effectiveness['risk_reward'] = 'optimal'
        elif trade_outcome == 'LOSS' and rr_ratio < 1.0:
            effectiveness['risk_reward'] = 'poor'
        else:
            effectiveness['risk_reward'] = 'acceptable'
        
        return effectiveness
    
    def identify_improvement_opportunities(self, trade_outcome: str, setup_conditions: Dict,
                                         performance_metrics: Dict) -> List[str]:
        """Identify improvement opportunities from episode"""
        opportunities = []
        
        if trade_outcome == 'LOSS':
            # Analyze why loss occurred
            confidence = setup_conditions.get('confidence', 0.5)
            if confidence > 0.8:
                opportunities.append("Reduce overconfidence in high-confidence setups")
            
            rr_ratio = setup_conditions.get('risk_reward_ratio', 1.0)
            if rr_ratio < 1.0:
                opportunities.append("Improve risk-reward ratio filtering")
            
            # Check if stop loss was too tight
            sl_hit_time = performance_metrics.get('sl_hit_time', 0)
            if sl_hit_time < 0.5:  # Hit SL too quickly
                opportunities.append("Adjust stop loss placement for volatility")
        
        elif trade_outcome == 'WIN':
            # Analyze winning patterns for optimization
            win_size = performance_metrics.get('win_size', 0)
            if win_size > 2.0:  # Large win
                opportunities.append("Replicate successful high-reward setups")
            
            duration = performance_metrics.get('duration', 0)
            if duration < 1.0:  # Quick win
                opportunities.append("Optimize for quick profitable setups")
        
        return opportunities
    
    def identify_success_factors(self, trade_outcome: str, setup_conditions: Dict) -> List[str]:
        """Identify success factors from episode"""
        success_factors = []
        
        if trade_outcome == 'WIN':
            # Identify what worked well
            confidence = setup_conditions.get('confidence', 0.5)
            if confidence > 0.8:
                success_factors.append("High confidence setup validation")
            
            patterns = setup_conditions.get('pattern_count', 0)
            if patterns >= 3:
                success_factors.append("Multiple pattern confluence")
            
            sentiment = setup_conditions.get('sentiment_alignment', False)
            if sentiment:
                success_factors.append("Positive sentiment alignment")
        
        return success_factors
    
    def determine_model_updates(self, learning_insights: Dict) -> Dict:
        """Determine model updates based on learning insights"""
        updates = {
            'parameter_adjustments': {},
            'feature_weights': {},
            'retraining_suggested': False,
            'confidence_calibration': 1.0
        }
        
        # Adjust parameters based on effectiveness
        param_effectiveness = learning_insights.get('parameter_effectiveness', {})
        
        if param_effectiveness.get('confidence_threshold') == 'overconfident':
            updates['parameter_adjustments']['confidence_threshold'] = -0.05
        elif param_effectiveness.get('confidence_threshold') == 'effective':
            updates['parameter_adjustments']['confidence_threshold'] = 0.02
        
        if param_effectiveness.get('risk_reward') == 'poor':
            updates['parameter_adjustments']['risk_reward_minimum'] = 0.1
        elif param_effectiveness.get('risk_reward') == 'optimal':
            updates['parameter_adjustments']['risk_reward_minimum'] = -0.05
        
        # Check if retraining is suggested
        setup_effectiveness = learning_insights.get('setup_effectiveness', 0.5)
        if setup_effectiveness < 0.3:
            updates['retraining_suggested'] = True
        
        return updates
    
    def update_adaptive_parameters(self, episode: LearningEpisode):
        """Update adaptive parameters based on learning"""
        model_updates = episode.model_updates.get('parameter_adjustments', {})
        
        for param_name, adjustment in model_updates.items():
            if param_name in self.adaptive_parameters:
                param = self.adaptive_parameters[param_name]
                new_value = param.current_value + adjustment
                
                # Ensure within optimal range
                optimal_min, optimal_max = param.optimal_range
                new_value = max(optimal_min, min(optimal_max, new_value))
                
                # Record adjustment
                adjustment_record = {
                    'timestamp': datetime.now().isoformat(),
                    'old_value': param.current_value,
                    'new_value': new_value,
                    'adjustment': adjustment,
                    'reason': 'learning_episode'
                }
                
                param.current_value = new_value
                param.adjustment_history.append(adjustment_record)
                param.last_updated = datetime.now()
                
                # Keep only last 100 adjustments
                if len(param.adjustment_history) > 100:
                    param.adjustment_history = param.adjustment_history[-100:]
                
                logging.info(f"Updated parameter {param_name}: {adjustment_record['old_value']} -> {new_value}")
        
        # Save updated parameters
        self.save_adaptive_parameters()
    
    def should_retrain_models(self) -> bool:
        """Determine if models should be retrained"""
        if len(self.learning_episodes) < 50:
            return False
        
        # Check performance degradation
        recent_episodes = self.learning_episodes[-50:]
        win_count = sum(1 for ep in recent_episodes if ep.trade_outcome == 'WIN')
        recent_win_rate = win_count / len(recent_episodes)
        
        # Check if win rate is below threshold
        if recent_win_rate < self.performance_threshold:
            return True
        
        # Check if significant learning has occurred
        improvement_opportunities = []
        for episode in recent_episodes:
            improvement_opportunities.extend(episode.learning_insights.get('improvement_opportunities', []))
        
        if len(improvement_opportunities) > 10:  # Many improvement opportunities
            return True
        
        return False
    
    def retrain_models(self):
        """Retrain AI models with new data"""
        try:
            logging.info("Starting model retraining...")
            
            # Prepare training data
            X, y = self.prepare_training_data()
            
            if len(X) < 100:
                logging.warning("Insufficient data for retraining")
                return
            
            # Retrain Random Forest
            rf_performance = self.retrain_random_forest(X, y)
            self.model_performance['random_forest'] = rf_performance
            
            # Retrain LSTM (if enough sequential data)
            if len(X) >= 200:
                lstm_performance = self.retrain_lstm(X, y)
                self.model_performance['lstm_network'] = lstm_performance
            
            # Update model performance in database
            self.save_model_performance()
            
            logging.info("Model retraining completed")
            
        except Exception as e:
            logging.error(f"Error in model retraining: {e}")
    
    def prepare_training_data(self):
        """Prepare training data from learning episodes"""
        X = []
        y = []
        
        for episode in self.learning_episodes:
            # Convert episode to features
            features = self.episode_to_features(episode)
            if features:
                X.append(features)
                
                # Convert outcome to label
                if episode.trade_outcome == 'WIN':
                    y.append(1)
                else:
                    y.append(0)
        
        return np.array(X), np.array(y)
    
    def episode_to_features(self, episode: LearningEpisode) -> List[float]:
        """Convert learning episode to feature vector"""
        features = []
        
        try:
            # Setup conditions features
            setup = episode.setup_conditions
            features.extend([
                setup.get('confidence', 0.5),
                setup.get('risk_reward_ratio', 1.0),
                setup.get('pattern_count', 0) / 10.0,  # Normalize
                setup.get('multi_tf_alignment', 0.5),
                setup.get('sentiment_alignment', 0.5)
            ])
            
            # Market conditions features
            market = episode.market_conditions
            features.extend([
                market.get('volatility', 0.5),
                market.get('trend_strength', 0.5),
                market.get('sentiment_score', 0.0),
                market.get('fear_greed_index', 50) / 100.0
            ])
            
            # Performance metrics features
            performance = episode.performance_metrics
            features.extend([
                performance.get('duration', 0) / 24.0,  # Normalize to days
                performance.get('max_drawdown', 0),
                performance.get('sharpe_ratio', 0)
            ])
            
            return features
            
        except Exception as e:
            logging.error(f"Error converting episode to features: {e}")
            return None
    
    def retrain_random_forest(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Retrain Random Forest model"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        
        # Evaluate performance
        y_pred = rf_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculate improvement
        old_performance = self.model_performance['random_forest']
        improvement = accuracy - old_performance.accuracy
        
        # Save model
        self.save_model(rf_model, 'random_forest')
        
        return ModelPerformance(
            model_name='random_forest',
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            training_date=datetime.now(),
            sample_size=len(X),
            improvement=improvement
        )
    
    def retrain_lstm(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Retrain LSTM model for sequential pattern recognition"""
        try:
            # Reshape data for LSTM (samples, timesteps, features)
            # This is simplified - in production, you'd use proper sequential data
            X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
            
            # Build LSTM model
            lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(1, X.shape[1])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            lstm_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = lstm_model.fit(
                X_reshaped, y,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Evaluate performance
            loss, accuracy = lstm_model.evaluate(X_reshaped, y, verbose=0)
            
            # Calculate improvement
            old_performance = self.model_performance['lstm_network']
            improvement = accuracy - old_performance.accuracy
            
            # Save model
            self.save_model(lstm_model, 'lstm_network')
            
            return ModelPerformance(
                model_name='lstm_network',
                accuracy=accuracy,
                precision=accuracy,  # Simplified
                recall=accuracy,     # Simplified
                f1_score=accuracy,   # Simplified
                training_date=datetime.now(),
                sample_size=len(X),
                improvement=improvement
            )
            
        except Exception as e:
            logging.error(f"Error retraining LSTM: {e}")
            return self.model_performance['lstm_network']
    
    def save_learning_episode(self, episode: LearningEpisode):
        """Save learning episode to database"""
        try:
            conn = sqlite3.connect('learning_system.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO learning_episodes 
                (episode_id, timestamp, trade_outcome, setup_conditions, market_conditions, 
                 performance_metrics, learning_insights, model_updates)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                episode.episode_id,
                episode.timestamp.isoformat(),
                episode.trade_outcome,
                json.dumps(episode.setup_conditions),
                json.dumps(episode.market_conditions),
                json.dumps(episode.performance_metrics),
                json.dumps(episode.learning_insights),
                json.dumps(episode.model_updates)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error saving learning episode: {e}")
    
    def save_adaptive_parameters(self):
        """Save adaptive parameters to database"""
        try:
            conn = sqlite3.connect('learning_system.db')
            cursor = conn.cursor()
            
            for param_name, param in self.adaptive_parameters.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO adaptive_parameters 
                    (parameter_name, current_value, optimal_min, optimal_max, adjustment_history, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    param_name,
                    param.current_value,
                    param.optimal_range[0],
                    param.optimal_range[1],
                    json.dumps(param.adjustment_history),
                    param.last_updated.isoformat()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error saving adaptive parameters: {e}")
    
    def save_model_performance(self):
        """Save model performance to database"""
        try:
            conn = sqlite3.connect('learning_system.db')
            cursor = conn.cursor()
            
            for model_name, performance in self.model_performance.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO model_performance 
                    (model_name, accuracy, precision, recall, f1_score, training_date, sample_size, improvement)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_name,
                    performance.accuracy,
                    performance.precision,
                    performance.recall,
                    performance.f1_score,
                    performance.training_date.isoformat(),
                    performance.sample_size,
                    performance.improvement
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error saving model performance: {e}")
    
    def save_model(self, model, model_name: str):
        """Save trained model to file"""
        try:
            models_dir = 'models'
            os.makedirs(models_dir, exist_ok=True)
            
            if model_name == 'random_forest':
                with open(f'{models_dir}/{model_name}.pkl', 'wb') as f:
                    pickle.dump(model, f)
            elif model_name == 'lstm_network':
                model.save(f'{models_dir}/{model_name}.h5')
            
            logging.info(f"Saved model: {model_name}")
            
        except Exception as e:
            logging.error(f"Error saving model {model_name}: {e}")
    
    def get_learning_insights(self) -> Dict:
        """Get comprehensive learning insights"""
        if len(self.learning_episodes) == 0:
            return {"message": "No learning data available"}
        
        # Calculate overall performance
        total_episodes = len(self.learning_episodes)
        win_episodes = sum(1 for ep in self.learning_episodes if ep.trade_outcome == 'WIN')
        loss_episodes = sum(1 for ep in self.learning_episodes if ep.trade_outcome == 'LOSS')
        win_rate = win_episodes / total_episodes if total_episodes > 0 else 0
        
        # Analyze recent performance
        recent_episodes = self.learning_episodes[-50:] if len(self.learning_episodes) >= 50 else self.learning_episodes
        recent_win_rate = sum(1 for ep in recent_episodes if ep.trade_outcome == 'WIN') / len(recent_episodes)
        
        # Get most common improvement opportunities
        all_improvements = []
        for episode in self.learning_episodes:
            all_improvements.extend(episode.learning_insights.get('improvement_opportunities', []))
        
        improvement_counts = {}
        for improvement in all_improvements:
            improvement_counts[improvement] = improvement_counts.get(improvement, 0) + 1
        
        top_improvements = sorted(improvement_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get parameter adjustment summary
        parameter_changes = {}
        for param_name, param in self.adaptive_parameters.items():
            if param.adjustment_history:
                latest_change = param.adjustment_history[-1]
                parameter_changes[param_name] = {
                    'current_value': param.current_value,
                    'last_change': latest_change['adjustment'],
                    'trend': 'increasing' if latest_change['adjustment'] > 0 else 'decreasing'
                }
        
        return {
            'performance_summary': {
                'total_episodes': total_episodes,
                'win_rate': win_rate,
                'recent_win_rate': recent_win_rate,
                'performance_trend': 'improving' if recent_win_rate > win_rate else 'stable' if recent_win_rate == win_rate else 'declining'
            },
            'model_performance': {
                model_name: {
                    'accuracy': perf.accuracy,
                    'improvement': perf.improvement,
                    'last_trained': perf.training_date.isoformat()
                } for model_name, perf in self.model_performance.items()
            },
            'top_improvement_areas': top_improvements,
            'parameter_optimization': parameter_changes,
            'learning_effectiveness': self.calculate_learning_effectiveness()
        }
    
    def calculate_learning_effectiveness(self) -> float:
        """Calculate how effective the learning system is"""
        if len(self.learning_episodes) < 20:
            return 0.5  # Default effectiveness
        
        # Calculate performance improvement over time
        early_episodes = self.learning_episodes[:20]
        recent_episodes = self.learning_episodes[-20:]
        
        early_win_rate = sum(1 for ep in early_episodes if ep.trade_outcome == 'WIN') / len(early_episodes)
        recent_win_rate = sum(1 for ep in recent_episodes if ep.trade_outcome == 'WIN') / len(recent_episodes)
        
        improvement = recent_win_rate - early_win_rate
        
        # Calculate parameter optimization effectiveness
        param_effectiveness = 0.0
        for param in self.adaptive_parameters.values():
            if len(param.adjustment_history) > 5:
                # Check if adjustments are converging
                recent_adjustments = [adj['adjustment'] for adj in param.adjustment_history[-5:]]
                adjustment_variance = np.var(recent_adjustments)
                param_effectiveness += 1.0 - min(adjustment_variance * 10, 1.0)
        
        param_effectiveness /= len(self.adaptive_parameters)
        
        # Combine metrics
        effectiveness = 0.6 * improvement + 0.4 * param_effectiveness
        return max(0.0, min(1.0, effectiveness))