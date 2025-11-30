"""
Babs AI Trading System - Data Validation
Market data validation and anomaly detection
"""
import numpy as np
from datetime import datetime, timedelta


class QuantumDataValidator:
    """Market data validation class"""
    
    def __init__(self):
        self.validation_threshold = 0.001  # 0.1% price difference tolerance
        self.volume_spike_threshold = 2.0   # 2x average volume = spike
        
    def cross_verify_prices(self, primary_data, secondary_data):
        """Cross-verify prices from multiple sources"""
        if not primary_data or not secondary_data:
            return False, "Missing data sources"
            
        primary_price = primary_data.get('close', 0)
        secondary_price = secondary_data.get('close', 0)
        
        if primary_price == 0 or secondary_price == 0:
            return False, "Invalid price data"
            
        price_diff = abs(primary_price - secondary_price) / primary_price
        
        if price_diff > self.validation_threshold:
            return False, f"Price discrepancy: {price_diff:.4%}"
            
        return True, "Price validation passed"
    
    def detect_anomalies(self, price_series, volume_series):
        """Statistical anomaly detection"""
        anomalies = []
        
        if len(price_series) < 2:
            return anomalies
        
        # Price anomaly detection using Z-score
        price_returns = np.diff(np.log(np.array(price_series) + 0.0001))
        
        if len(price_returns) > 0 and np.std(price_returns) > 0:
            price_z_scores = np.abs((price_returns - np.mean(price_returns)) / np.std(price_returns))
        else:
            price_z_scores = np.zeros(len(price_returns))
        
        # Volume anomaly detection
        volume_array = np.array(volume_series)
        volume_mean = np.mean(volume_array) if len(volume_array) > 0 else 0
        volume_std = np.std(volume_array) if len(volume_array) > 0 else 1
        
        for i, (price_z, volume) in enumerate(zip(price_z_scores, volume_series[1:])):
            if price_z > 3:  # 3 standard deviations
                anomalies.append(f"Price anomaly at index {i}, Z-score: {price_z:.2f}")
            if volume_std > 0 and volume > volume_mean + 3 * volume_std:
                anomalies.append(f"Volume spike at index {i}, Volume: {volume}")
                
        return anomalies
    
    def temporal_consistency_check(self, timestamps):
        """Ensure no time gaps or future timestamps"""
        now = datetime.now()
        gaps = []
        
        for i in range(1, len(timestamps)):
            time_diff = timestamps[i] - timestamps[i-1]
            if time_diff > timedelta(hours=2):
                gaps.append(f"Time gap between {timestamps[i-1]} and {timestamps[i]}")
            
            if timestamps[i] > now:
                gaps.append(f"Future timestamp detected: {timestamps[i]}")
                
        return gaps
    
    def get_confidence_score(self, market_data):
        """Calculate overall data confidence score (0-100)"""
        score = 100
        
        if not market_data:
            return 0
        
        # If market_data is a dict with simple values, return high score
        if isinstance(market_data, dict):
            if 'close' not in market_data:
                return 50
            if market_data.get('source') == 'Mock Data':
                return 60
            return 85
        
        # For complex data with arrays
        prices = market_data.get('prices', [])
        volumes = market_data.get('volumes', [])
        timestamps = market_data.get('timestamps', [])
        
        if prices and volumes:
            # Penalize for anomalies
            anomalies = self.detect_anomalies(prices, volumes)
            score -= len(anomalies) * 10
        
        if timestamps:
            # Penalize for temporal issues
            time_issues = self.temporal_consistency_check(timestamps)
            score -= len(time_issues) * 15
        
        # Volume sanity check
        if market_data.get('volume', 0) < 0:
            score -= 20
            
        return max(0, score)
    
    def get_current_timestamp(self):
        """Get current timestamp in ISO format"""
        return datetime.now().isoformat()
