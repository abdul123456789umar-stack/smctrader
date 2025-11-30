from flask import jsonify, request
from signal_ranker import SignalRankingEngine, DeliveryEngine, ReversalAlertSystem, RankedSignal, SignalPriority, DeliveryStatus
from datetime import datetime, timedelta
import json

# Initialize signal systems
ranking_engine = SignalRankingEngine()
delivery_engine = DeliveryEngine()
reversal_system = ReversalAlertSystem()

def setup_signal_routes(app):
    
    @app.route('/api/generate-signals')
    def generate_trading_signals():
        """Generate and rank trading signals"""
        try:
            # Get raw signals from AI engine
            from ai_routes import get_ai_signals
            raw_signals_data = get_ai_signals()
            raw_signals = raw_signals_data.get('signals', [])
            
            # Rank and filter signals
            ranked_signals = ranking_engine.rank_signals(raw_signals)
            
            # Deliver signals
            delivered_signals = []
            for signal in ranked_signals:
                # Save to database
                signal_id = ranking_engine.save_signal(signal)
                
                # Deliver through channels
                delivery_results = delivery_engine.deliver_signal(signal)
                
                # Update delivery status
                ranking_engine.update_delivery_status(signal_id, DeliveryStatus.SENT)
                
                # Start monitoring for reversals
                reversal_system.monitor_signal(signal)
                
                delivered_signals.append({
                    'id': signal_id,
                    'symbol': signal.symbol,
                    'direction': signal.direction,
                    'entry': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'confidence': signal.confidence,
                    'ranking_score': signal.ranking_score,
                    'priority': signal.priority.name,
                    'reasoning': signal.reasoning,
                    'delivery_channels': signal.delivery_channels,
                    'delivery_results': delivery_results,
                    'expiry': signal.expiry.isoformat(),
                    'risk_reward': signal.risk_reward_ratio
                })
            
            return jsonify({
                'status': 'success',
                'signals_generated': len(ranked_signals),
                'signals': delivered_signals,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/todays-signals')
    def get_todays_signals():
        """Get today's trading signals"""
        try:
            signals = ranking_engine.get_todays_signals()
            
            signal_data = []
            for signal in signals:
                signal_data.append({
                    'id': getattr(signal, 'id', None),
                    'symbol': signal.symbol,
                    'direction': signal.direction,
                    'entry': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'confidence': signal.confidence,
                    'ranking_score': signal.ranking_score,
                    'priority': signal.priority.name,
                    'reasoning': signal.reasoning,
                    'status': signal.delivery_status.value,
                    'timestamp': signal.timestamp.isoformat(),
                    'expiry': signal.expiry.isoformat(),
                    'risk_reward': signal.risk_reward_ratio,
                    'setup_quality': signal.setup_quality
                })
            
            return jsonify({
                'signals': signal_data,
                'total_count': len(signals),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/pending-setups')
    def get_pending_setups():
        """Get potential setups waiting for POI confirmation"""
        try:
            # This would analyze markets for potential setups
            pending_setups = generate_pending_setups()
            
            return jsonify({
                'pending_setups': pending_setups,
                'total_count': len(pending_setups),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/check-reversals')
    def check_reversal_alerts():
        """Check for potential reversal alerts"""
        try:
            # Get current prices (mock for demo)
            current_prices = get_current_prices()
            
            # Check for reversals
            reversal_alerts = reversal_system.check_reversals(current_prices)
            
            # Deliver reversal alerts
            for alert in reversal_alerts:
                delivery_engine.deliver_reversal_alert(alert)
            
            return jsonify({
                'alerts': reversal_alerts,
                'total_alerts': len(reversal_alerts),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/signal-performance')
    def get_signal_performance():
        """Get historical signal performance"""
        try:
            performance = calculate_signal_performance()
            
            return jsonify({
                'performance_metrics': performance,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/manual-signal', methods=['POST'])
    def create_manual_signal():
        """Create a manual trading signal"""
        try:
            data = request.json
            
            # Create ranked signal from manual data
            signal = RankedSignal(
                symbol=data['symbol'],
                direction=data['direction'],
                entry_price=data['entry_price'],
                stop_loss=data['stop_loss'],
                take_profit=data['take_profit'],
                confidence=data.get('confidence', 0.7),
                priority=SignalPriority[data.get('priority', 'MEDIUM')],
                ranking_score=data.get('ranking_score', 70),
                reasoning=data.get('reasoning', ['Manual signal']),
                timestamp=datetime.now(),
                expiry=datetime.now() + timedelta(hours=4),
                delivery_status=DeliveryStatus.PENDING,
                delivery_channels=data.get('delivery_channels', ['in_app']),
                risk_reward_ratio=data.get('risk_reward_ratio', 1.5),
                setup_quality=data.get('setup_quality', 0.7)
            )
            
            # Save and deliver
            signal_id = ranking_engine.save_signal(signal)
            delivery_results = delivery_engine.deliver_signal(signal)
            ranking_engine.update_delivery_status(signal_id, DeliveryStatus.SENT)
            
            return jsonify({
                'status': 'success',
                'signal_id': signal_id,
                'delivery_results': delivery_results
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

def generate_pending_setups():
    """Generate pending setups waiting for POI confirmation"""
    # This would analyze markets for potential setups
    # For demo, return mock data
    return [
        {
            'symbol': 'EUR/USD',
            'potential_direction': 'BUY',
            'waiting_for': 'Break above 1.0950 resistance',
            'confidence': 0.75,
            'expected_entry': 1.0960,
            'timeframe': '4H',
            'reasoning': ['Bullish OB formed', 'FVG present', 'Waiting for BOS confirmation']
        },
        {
            'symbol': 'GBP/JPY',
            'potential_direction': 'SELL',
            'waiting_for': 'Break below 183.20 support',
            'confidence': 0.68,
            'expected_entry': 183.00,
            'timeframe': '1H',
            'reasoning': ['Bearish OB identified', 'Liquidity sweep potential', 'Awaiting CHOCH']
        }
    ]

def get_current_prices():
    """Get current prices for all monitored symbols"""
    # Mock implementation - replace with real price feed
    return {
        'EUR/USD': 1.0920,
        'GBP/USD': 1.2750,
        'USD/JPY': 147.50,
        'BTC/USDT': 42500,
        'XAU/USD': 1980
    }

def calculate_signal_performance():
    """Calculate historical signal performance"""
    # This would query the database for performance metrics
    return {
        'total_signals': 24,
        'win_rate': 0.68,
        'avg_rr_ratio': 1.8,
        'best_performing_symbol': 'EUR/USD',
        'avg_holding_time': '2.5 hours',
        'performance_trend': 'improving'
    }