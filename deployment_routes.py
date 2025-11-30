from flask import jsonify, request
from deployment_manager import DeploymentManager, PerformanceOptimizer, SecurityHardener
from datetime import datetime
import subprocess
import os

# Global deployment manager instance
deployment_manager = None

def setup_deployment_routes(app):
    global deployment_manager
    
    # Initialize deployment systems
    deployment_manager = DeploymentManager(app)
    performance_optimizer = PerformanceOptimizer(app)
    security_hardener = SecurityHardener(app)
    
    @app.route('/api/deployment/status')
    def get_deployment_status():
        """Get deployment and system status"""
        try:
            system_status = deployment_manager.get_system_status()
            
            return jsonify({
                "deployment": {
                    "platform": deployment_manager.config.platform,
                    "environment": deployment_manager.config.environment,
                    "timestamp": datetime.now().isoformat()
                },
                "system_status": system_status,
                "services": {
                    "monitoring": deployment_manager.config.monitoring_enabled,
                    "auto_scaling": deployment_manager.config.auto_scaling,
                    "backup": deployment_manager.config.backup_enabled
                }
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/deployment/metrics')
    def get_system_metrics():
        """Get system metrics history"""
        try:
            metrics_data = []
            for metric in deployment_manager.metrics_history[-100:]:  # Last 100 metrics
                metrics_data.append({
                    "timestamp": metric.timestamp.isoformat(),
                    "cpu_percent": metric.cpu_percent,
                    "memory_percent": metric.memory_percent,
                    "disk_usage": metric.disk_usage,
                    "active_connections": metric.active_connections,
                    "request_count": metric.request_count,
                    "error_rate": metric.error_rate
                })
            
            return jsonify({
                "metrics": metrics_data,
                "summary": {
                    "total_metrics": len(metrics_data),
                    "time_range": f"Last {len(metrics_data)} minutes"
                }
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/deployment/health')
    def get_health_check():
        """Get detailed health check results"""
        try:
            system_status = deployment_manager.get_system_status()
            
            return jsonify({
                "health_checks": system_status['health_checks'],
                "overall_status": system_status['overall_status'],
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/deployment/backup', methods=['POST'])
    def perform_system_backup():
        """Perform system backup"""
        try:
            deployment_manager.perform_backup()
            
            return jsonify({
                "status": "success",
                "message": "Backup completed successfully",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/deployment/deploy', methods=['POST'])
    def deploy_application():
        """Deploy application to target platform"""
        try:
            success = deployment_manager.deploy_to_platform()
            
            if success:
                return jsonify({
                    "status": "success",
                    "message": f"Deployment to {deployment_manager.config.platform} completed",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                return jsonify({"error": "Deployment failed"}), 500
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/deployment/optimize', methods=['POST'])
    def optimize_performance():
        """Run performance optimizations"""
        try:
            performance_optimizer.optimize_queries()
            
            return jsonify({
                "status": "success",
                "message": "Performance optimizations completed",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/deployment/logs')
    def get_system_logs():
        """Get system logs"""
        try:
            log_file = 'logs/trading_system.log'
            if not os.path.exists(log_file):
                return jsonify({"error": "Log file not found"}), 404
            
            with open(log_file, 'r') as f:
                lines = f.readlines()[-100:]  # Last 100 lines
            
            return jsonify({
                "logs": lines,
                "total_lines": len(lines)
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/deployment/scale', methods=['POST'])
    def scale_application():
        """Scale application resources"""
        try:
            data = request.json
            scale_type = data.get('type', 'up')  # 'up' or 'down'
            
            deployment_manager.scale_resources()
            
            return jsonify({
                "status": "success",
                "message": f"Scaling {scale_type} initiated",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/deployment/config')
    def get_deployment_config():
        """Get deployment configuration"""
        try:
            return jsonify({
                "deployment_config": {
                    "platform": deployment_manager.config.platform,
                    "environment": deployment_manager.config.environment,
                    "monitoring_enabled": deployment_manager.config.monitoring_enabled,
                    "auto_scaling": deployment_manager.config.auto_scaling,
                    "backup_enabled": deployment_manager.config.backup_enabled
                },
                "api_keys_configured": list(deployment_manager.config.api_keys.keys())
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/deployment/restart', methods=['POST'])
    def restart_application():
        """Restart application (simulated)"""
        try:
            # In production, this would trigger a proper restart
            # For now, just log and return success
            deployment_manager.logger.info("Application restart requested")
            
            return jsonify({
                "status": "success",
                "message": "Restart initiated",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Production-ready main application setup
def create_production_app():
    from flask import Flask
    from flask_cors import CORS
    
    app = Flask(__name__)
    
    # Basic configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-change-in-production')
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Disable pretty print in production
    
    # Setup CORS
    CORS(app)
    
    # Setup all routes
    from user_routes import setup_user_routes
    from ai_routes import setup_ai_routes
    from sentiment_routes import setup_sentiment_routes
    from signal_routes import setup_signal_routes
    from chart_routes import setup_chart_routes
    
    setup_user_routes(app)
    setup_ai_routes(app)
    setup_sentiment_routes(app)
    setup_signal_routes(app)
    setup_chart_routes(app)
    setup_deployment_routes(app)
    
    # Setup deployment manager
    global deployment_manager
    deployment_manager = DeploymentManager(app)
    
    return app

# Production WSGI entry point
application = create_production_app()

if __name__ == '__main__':
    # Development server
    application.run(
        host='0.0.0.0',
        port=5000,
        debug=os.getenv('ENVIRONMENT') == 'development'
    )