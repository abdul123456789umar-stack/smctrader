from flask import jsonify, request
from continuous_learning import ContinuousLearningEngine, LearningEpisode
from datetime import datetime
import json

# Initialize continuous learning engine
learning_engine = ContinuousLearningEngine()

def setup_learning_routes(app):
    
    @app.route('/api/learning/record-episode', methods=['POST'])
    def record_learning_episode():
        """Record a learning episode from trade outcome"""
        try:
            data = request.json
            
            episode = learning_engine.record_learning_episode(
                trade_outcome=data['trade_outcome'],
                setup_conditions=data['setup_conditions'],
                market_conditions=data['market_conditions'],
                performance_metrics=data['performance_metrics']
            )
            
            if episode:
                return jsonify({
                    "status": "success",
                    "episode_id": episode.episode_id,
                    "learning_insights": episode.learning_insights,
                    "model_updates": episode.model_updates
                })
            else:
                return jsonify({"error": "Failed to record learning episode"}), 500
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    # COMPLETING THE INCOMPLETE ROUTE:
    @app.route('/api/learning/insights')
    def get_learning_insights():
        """Get comprehensive learning insights and performance analytics"""
        try:
            insights = learning_engine.get_learning_insights()
            return jsonify({
                "status": "success",
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    # ADDING MISSING ROUTES:
    @app.route('/api/learning/performance')
    def get_model_performance():
        """Get current model performance metrics"""
        try:
            performance_data = {}
            for model_name, performance in learning_engine.model_performance.items():
                performance_data[model_name] = {
                    "accuracy": performance.accuracy,
                    "precision": performance.precision,
                    "recall": performance.recall,
                    "f1_score": performance.f1_score,
                    "improvement": performance.improvement,
                    "last_trained": performance.training_date.isoformat(),
                    "sample_size": performance.sample_size
                }
            
            return jsonify({
                "status": "success",
                "model_performance": performance_data
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/learning/parameters')
    def get_adaptive_parameters():
        """Get current adaptive parameters"""
        try:
            parameters_data = {}
            for param_name, param in learning_engine.adaptive_parameters.items():
                parameters_data[param_name] = {
                    "current_value": param.current_value,
                    "optimal_range": param.optimal_range,
                    "last_updated": param.last_updated.isoformat(),
                    "adjustment_count": len(param.adjustment_history)
                }
            
            return jsonify({
                "status": "success",
                "adaptive_parameters": parameters_data
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/learning/episodes')
    def get_learning_episodes():
        """Get recent learning episodes"""
        try:
            limit = request.args.get('limit', 50, type=int)
            recent_episodes = learning_engine.learning_episodes[-limit:] if learning_engine.learning_episodes else []
            
            episodes_data = []
            for episode in recent_episodes:
                episodes_data.append({
                    "episode_id": episode.episode_id,
                    "timestamp": episode.timestamp.isoformat(),
                    "trade_outcome": episode.trade_outcome,
                    "setup_conditions": episode.setup_conditions,
                    "performance_metrics": episode.performance_metrics,
                    "learning_insights": episode.learning_insights
                })
            
            return jsonify({
                "status": "success",
                "episodes": episodes_data,
                "total_episodes": len(learning_engine.learning_episodes)
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/learning/retrain', methods=['POST'])
    def trigger_retraining():
        """Manually trigger model retraining"""
        try:
            learning_engine.retrain_models()
            return jsonify({
                "status": "success",
                "message": "Model retraining initiated",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/learning/optimize', methods=['POST'])
    def optimize_parameters():
        """Optimize adaptive parameters based on recent performance"""
        try:
            # Analyze recent performance and optimize parameters
            recent_episodes = learning_engine.learning_episodes[-100:] if len(learning_engine.learning_episodes) >= 100 else learning_engine.learning_episodes
            
            if len(recent_episodes) < 20:
                return jsonify({"error": "Insufficient data for optimization"}), 400
            
            # Calculate win rate for recent episodes
            win_rate = sum(1 for ep in recent_episodes if ep.trade_outcome == 'WIN') / len(recent_episodes)
            
            # Adjust parameters based on performance
            adjustments = {}
            if win_rate < 0.6:
                # Increase confidence threshold if performance is poor
                adjustments['confidence_threshold'] = 0.02
            elif win_rate > 0.8:
                # Decrease confidence threshold if performance is excellent
                adjustments['confidence_threshold'] = -0.01
            
            # Apply adjustments
            for param_name, adjustment in adjustments.items():
                if param_name in learning_engine.adaptive_parameters:
                    param = learning_engine.adaptive_parameters[param_name]
                    new_value = param.current_value + adjustment
                    optimal_min, optimal_max = param.optimal_range
                    new_value = max(optimal_min, min(optimal_max, new_value))
                    
                    adjustment_record = {
                        'timestamp': datetime.now().isoformat(),
                        'old_value': param.current_value,
                        'new_value': new_value,
                        'adjustment': adjustment,
                        'reason': 'manual_optimization'
                    }
                    
                    param.current_value = new_value
                    param.adjustment_history.append(adjustment_record)
                    param.last_updated = datetime.now()
            
            # Save updated parameters
            learning_engine.save_adaptive_parameters()
            
            return jsonify({
                "status": "success",
                "message": "Parameters optimized",
                "adjustments_made": adjustments,
                "current_win_rate": win_rate
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/learning/status')
    def get_learning_status():
        """Get overall learning system status"""
        try:
            total_episodes = len(learning_engine.learning_episodes)
            win_episodes = sum(1 for ep in learning_engine.learning_episodes if ep.trade_outcome == 'WIN')
            win_rate = win_episodes / total_episodes if total_episodes > 0 else 0
            
            # Calculate learning effectiveness
            effectiveness = learning_engine.calculate_learning_effectiveness()
            
            # Check if retraining is needed
            retraining_needed = learning_engine.should_retrain_models()
            
            return jsonify({
                "status": "success",
                "learning_system": {
                    "total_episodes": total_episodes,
                    "win_rate": win_rate,
                    "learning_effectiveness": effectiveness,
                    "retraining_needed": retraining_needed,
                    "adaptive_parameters_count": len(learning_engine.adaptive_parameters),
                    "models_tracked": len(learning_engine.model_performance)
                }
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Register the learning routes with your Flask app
# Add this to your main app.py:
# from learning_routes import setup_learning_routes
# setup_learning_routes(app)