import os
import sys
import logging
import subprocess
import requests
import time
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import psutil
import threading
from flask import Flask

@dataclass
class DeploymentConfig:
    platform: str  # 'pythonanywhere', 'render', 'railway'
    environment: str  # 'development', 'staging', 'production'
    api_keys: Dict[str, str]
    monitoring_enabled: bool
    auto_scaling: bool
    backup_enabled: bool

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    active_connections: int
    request_count: int
    error_rate: float
    timestamp: datetime

@dataclass
class HealthCheck:
    service: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    response_time: float
    last_check: datetime
    details: Dict

class DeploymentManager:
    def __init__(self, app: Flask):
        self.app = app
        self.config = self.load_deployment_config()
        self.metrics_history = []
        self.health_checks = {}
        self.setup_logging()
        self.setup_monitoring()
    
    def load_deployment_config(self) -> DeploymentConfig:
        """Load deployment configuration from environment"""
        return DeploymentConfig(
            platform=os.getenv('DEPLOYMENT_PLATFORM', 'pythonanywhere'),
            environment=os.getenv('ENVIRONMENT', 'development'),
            api_keys={
                'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY', 'demo'),
                
                'newsapi': os.getenv('NEWS_API_KEY', ''),
                'telegram': os.getenv('TELEGRAM_BOT_TOKEN', '')
            },
            monitoring_enabled=os.getenv('MONITORING_ENABLED', 'True').lower() == 'true',
            auto_scaling=os.getenv('AUTO_SCALING', 'False').lower() == 'true',
            backup_enabled=os.getenv('BACKUP_ENABLED', 'True').lower() == 'true'
        )
    
    def setup_logging(self):
        """Setup structured logging for production"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/trading_system.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('TradingSystem')
        self.logger.info("Deployment manager initialized")
    
    def setup_monitoring(self):
        """Setup system monitoring and health checks"""
        if self.config.monitoring_enabled:
            self.start_metrics_collection()
            self.start_health_checks()
            self.logger.info("System monitoring enabled")
    
    def start_metrics_collection(self):
        """Start collecting system metrics"""
        def collect_metrics():
            while True:
                try:
                    metrics = self.collect_system_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Keep only last 1000 metrics
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-1000:]
                    
                    # Log if system is under stress
                    if metrics.cpu_percent > 80 or metrics.memory_percent > 85:
                        self.logger.warning(f"System under stress: CPU {metrics.cpu_percent}%, Memory {metrics.memory_percent}%")
                    
                    time.sleep(60)  # Collect every minute
                    
                except Exception as e:
                    self.logger.error(f"Error collecting metrics: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
        metrics_thread.start()
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            active_connections=self.get_active_connections(),
            request_count=self.get_request_count(),
            error_rate=self.get_error_rate(),
            timestamp=datetime.now()
        )
    
    def get_active_connections(self) -> int:
        """Get number of active database connections"""
        try:
            conn = sqlite3.connect('trading_signals.db')
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master")
            # This is a simple check - in production, use connection pool metrics
            return 1
        except:
            return 0
        finally:
            conn.close()
    
    def get_request_count(self) -> int:
        """Get request count (simplified - in production use proper request tracking)"""
        return 0  # Would be implemented with request middleware
    
    def get_error_rate(self) -> float:
        """Get error rate (simplified - in production use proper error tracking)"""
        return 0.0
    
    def start_health_checks(self):
        """Start periodic health checks"""
        def perform_health_checks():
            while True:
                try:
                    self.perform_database_health_check()
                    self.perform_api_health_checks()
                    self.perform_service_health_checks()
                    time.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    self.logger.error(f"Error in health checks: {e}")
                    time.sleep(300)
        
        health_thread = threading.Thread(target=perform_health_checks, daemon=True)
        health_thread.start()
    
    def perform_database_health_check(self):
        """Perform database health check"""
        try:
            start_time = time.time()
            
            # Check main databases
            databases = ['users.db', 'trading_signals.db', 'trading_ai.db']
            
            for db_name in databases:
                if os.path.exists(db_name):
                    conn = sqlite3.connect(db_name)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    conn.close()
                    
                    status = 'healthy' if len(tables) > 0 else 'degraded'
                    response_time = time.time() - start_time
                    
                    self.health_checks[f'database_{db_name}'] = HealthCheck(
                        service=f'Database_{db_name}',
                        status=status,
                        response_time=response_time,
                        last_check=datetime.now(),
                        details={'table_count': len(tables)}
                    )
                else:
                    self.health_checks[f'database_{db_name}'] = HealthCheck(
                        service=f'Database_{db_name}',
                        status='unhealthy',
                        response_time=0.0,
                        last_check=datetime.now(),
                        details={'error': 'Database file not found'}
                    )
                    
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
    
    def perform_api_health_checks(self):
        """Perform external API health checks"""
        api_checks = [
            ('alpha_vantage', 'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=EURUSD&apikey=demo'),
            
            ('newsapi', 'https://newsapi.org/v2/top-headlines?country=us&apiKey=demo')
        ]
        
        for api_name, url in api_checks:
            try:
                start_time = time.time()
                response = requests.get(url, timeout=10)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    status = 'healthy'
                elif response.status_code in [401, 429]:
                    status = 'degraded'
                else:
                    status = 'unhealthy'
                
                self.health_checks[f'api_{api_name}'] = HealthCheck(
                    service=f'API_{api_name}',
                    status=status,
                    response_time=response_time,
                    last_check=datetime.now(),
                    details={'status_code': response.status_code}
                )
                
            except Exception as e:
                self.health_checks[f'api_{api_name}'] = HealthCheck(
                    service=f'API_{api_name}',
                    status='unhealthy',
                    response_time=0.0,
                    last_check=datetime.now(),
                    details={'error': str(e)}
                )
    
    def perform_service_health_checks(self):
        """Perform internal service health checks"""
        services = [
            ('ai_reasoning', self.check_ai_service),
            ('signal_generation', self.check_signal_service),
            ('data_processing', self.check_data_service)
        ]
        
        for service_name, check_function in services:
            try:
                start_time = time.time()
                status, details = check_function()
                response_time = time.time() - start_time
                
                self.health_checks[f'service_{service_name}'] = HealthCheck(
                    service=f'Service_{service_name}',
                    status=status,
                    response_time=response_time,
                    last_check=datetime.now(),
                    details=details
                )
                
            except Exception as e:
                self.health_checks[f'service_{service_name}'] = HealthCheck(
                    service=f'Service_{service_name}',
                    status='unhealthy',
                    response_time=0.0,
                    last_check=datetime.now(),
                    details={'error': str(e)}
                )
    
    def check_ai_service(self) -> Tuple[str, Dict]:
        """Check AI reasoning service health"""
        try:
            # Check if AI models are loaded and responsive
            from ai_reasoner import AIReasoningEngine
            ai_engine = AIReasoningEngine()
            
            # Simple test inference
            test_features = [[0.5] * 20]  # Mock features
            confidence = ai_engine.calculate_confidence(test_features, {}, {})
            
            return 'healthy', {'confidence_score': confidence}
        except Exception as e:
            return 'unhealthy', {'error': str(e)}
    
    def check_signal_service(self) -> Tuple[str, Dict]:
        """Check signal generation service health"""
        try:
            from signal_ranker import SignalRankingEngine
            ranking_engine = SignalRankingEngine()
            
            # Check if database is accessible
            signals = ranking_engine.get_todays_signals()
            
            return 'healthy', {'signal_count': len(signals)}
        except Exception as e:
            return 'unhealthy', {'error': str(e)}
    
    def check_data_service(self) -> Tuple[str, Dict]:
        """Check data processing service health"""
        try:
            from data_validation import QuantumDataValidator
            validator = QuantumDataValidator()
            
            # Test data validation
            test_data = {'close': 1.0950, 'volume': 1000}
            is_valid, message = validator.cross_verify_prices(test_data, test_data)
            
            return 'healthy', {'validation_passed': is_valid}
        except Exception as e:
            return 'unhealthy', {'error': str(e)}
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        healthy_checks = sum(1 for check in self.health_checks.values() if check.status == 'healthy')
        total_checks = len(self.health_checks)
        
        overall_status = 'healthy'
        if healthy_checks / total_checks < 0.7:
            overall_status = 'unhealthy'
        elif healthy_checks / total_checks < 0.9:
            overall_status = 'degraded'
        
        # Get latest metrics
        current_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            'overall_status': overall_status,
            'health_checks': {
                check.service: {
                    'status': check.status,
                    'response_time': check.response_time,
                    'last_check': check.last_check.isoformat(),
                    'details': check.details
                } for check in self.health_checks.values()
            },
            'current_metrics': {
                'cpu_percent': current_metrics.cpu_percent if current_metrics else 0,
                'memory_percent': current_metrics.memory_percent if current_metrics else 0,
                'disk_usage': current_metrics.disk_usage if current_metrics else 0,
                'active_connections': current_metrics.active_connections if current_metrics else 0,
                'timestamp': current_metrics.timestamp.isoformat() if current_metrics else None
            },
            'summary': {
                'total_checks': total_checks,
                'healthy_checks': healthy_checks,
                'degraded_checks': total_checks - healthy_checks - sum(1 for check in self.health_checks.values() if check.status == 'unhealthy'),
                'unhealthy_checks': sum(1 for check in self.health_checks.values() if check.status == 'unhealthy')
            }
        }
    
    def perform_backup(self):
        """Perform system backup"""
        if not self.config.backup_enabled:
            return
        
        try:
            backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"backups/{backup_time}"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Backup databases
            databases = ['users.db', 'trading_signals.db', 'trading_ai.db']
            for db_name in databases:
                if os.path.exists(db_name):
                    import shutil
                    shutil.copy2(db_name, f"{backup_dir}/{db_name}")
            
            # Backup logs
            if os.path.exists('logs'):
                shutil.copytree('logs', f"{backup_dir}/logs", dirs_exist_ok=True)
            
            # Backup configuration
            config_data = {
                'deployment_config': self.config.__dict__,
                'health_checks': {k: v.__dict__ for k, v in self.health_checks.items()},
                'timestamp': datetime.now().isoformat()
            }
            
            with open(f"{backup_dir}/config_backup.json", 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Backup completed: {backup_dir}")
            
            # Cleanup old backups (keep last 7 days)
            self.cleanup_old_backups()
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
    
    def cleanup_old_backups(self):
        """Cleanup backups older than 7 days"""
        try:
            backup_dir = "backups"
            if not os.path.exists(backup_dir):
                return
            
            for folder in os.listdir(backup_dir):
                folder_path = os.path.join(backup_dir, folder)
                if os.path.isdir(folder_path):
                    folder_time = datetime.fromtimestamp(os.path.getctime(folder_path))
                    if datetime.now() - folder_time > timedelta(days=7):
                        import shutil
                        shutil.rmtree(folder_path)
                        self.logger.info(f"Deleted old backup: {folder}")
                        
        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")
    
    def scale_resources(self):
        """Scale resources based on load (placeholder for cloud platforms)"""
        if not self.config.auto_scaling:
            return
        
        try:
            if self.metrics_history:
                recent_metrics = self.metrics_history[-10:]  # Last 10 minutes
                avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
                avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
                
                if avg_cpu > 80 or avg_memory > 85:
                    self.logger.warning("High resource usage detected - consider scaling up")
                elif avg_cpu < 20 and avg_memory < 30:
                    self.logger.info("Low resource usage - consider scaling down")
                    
        except Exception as e:
            self.logger.error(f"Scaling check failed: {e}")
    
    def deploy_to_platform(self):
        """Deploy to configured platform"""
        try:
            if self.config.platform == 'pythonanywhere':
                return self.deploy_to_pythonanywhere()
            elif self.config.platform == 'render':
                return self.deploy_to_render()
            elif self.config.platform == 'railway':
                return self.deploy_to_railway()
            else:
                raise ValueError(f"Unsupported platform: {self.config.platform}")
                
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            return False
    
    def deploy_to_pythonanywhere(self) -> bool:
        """Deploy to PythonAnywhere"""
        try:
            # This would contain PythonAnywhere specific deployment logic
            # For now, just log and return success
            self.logger.info("Deploying to PythonAnywhere...")
            
            # In production, this would:
            # 1. Upload files via API
            # 2. Configure web app
            # 3. Set up scheduled tasks
            # 4. Configure databases
            
            return True
        except Exception as e:
            self.logger.error(f"PythonAnywhere deployment failed: {e}")
            return False
    
    def deploy_to_render(self) -> bool:
        """Deploy to Render"""
        try:
            self.logger.info("Deploying to Render...")
            
            # Render deployment would use their Blueprint system
            # or direct API deployment
            
            return True
        except Exception as e:
            self.logger.error(f"Render deployment failed: {e}")
            return False
    
    def deploy_to_railway(self) -> bool:
        """Deploy to Railway"""
        try:
            self.logger.info("Deploying to Railway...")
            
            # Railway deployment via their CLI or API
            
            return True
        except Exception as e:
            self.logger.error(f"Railway deployment failed: {e}")
            return False

class PerformanceOptimizer:
    """Performance optimization for production environment"""
    
    def __init__(self, app: Flask):
        self.app = app
        self.setup_optimizations()
    
    def setup_optimizations(self):
        """Setup performance optimizations"""
        self.setup_caching()
        self.setup_compression()
        self.setup_database_optimizations()
    
    def setup_caching(self):
        """Setup response caching"""
        from flask_caching import Cache
        
        cache_config = {
            'CACHE_TYPE': 'SimpleCache',
            'CACHE_DEFAULT_TIMEOUT': 300  # 5 minutes
        }
        
        self.cache = Cache(self.app, config=cache_config)
    
    def setup_compression(self):
        """Setup response compression"""
        from flask_compress import Compress
        Compress(self.app)
    
    def setup_database_optimizations(self):
        """Setup database performance optimizations"""
        # Enable WAL mode for better concurrent reads
        for db_name in ['users.db', 'trading_signals.db', 'trading_ai.db']:
            try:
                conn = sqlite3.connect(db_name)
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('PRAGMA synchronous=NORMAL')
                conn.execute('PRAGMA cache_size=-64000')  # 64MB cache
                conn.execute('PRAGMA foreign_keys=ON')
                conn.close()
            except Exception as e:
                logging.getLogger('TradingSystem').error(f"Database optimization failed for {db_name}: {e}")
    
    def optimize_queries(self):
        """Analyze and optimize slow queries"""
        # This would analyze query performance and suggest optimizations
        # For now, it's a placeholder for production query optimization
        pass

class SecurityHardener:
    """Security hardening for production deployment"""
    
    def __init__(self, app: Flask):
        self.app = app
        self.setup_security()
    
    def setup_security(self):
        """Setup security measures"""
        self.setup_https()
        self.setup_cors()
        self.setup_rate_limiting()
        self.setup_headers()
    
    def setup_https(self):
        """Setup HTTPS enforcement"""
        if os.getenv('ENVIRONMENT') == 'production':
            from flask_talisman import Talisman
            Talisman(self.app, force_https=True)
    
    def setup_cors(self):
        """Setup CORS protection"""
        from flask_cors import CORS
        CORS(self.app, resources={
            r"/api/*": {
                "origins": os.getenv('ALLOWED_ORIGINS', '').split(',')
            }
        })
    
    def setup_rate_limiting(self):
        """Setup rate limiting"""
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
        
        self.limiter = Limiter(
            self.app,
            key_func=get_remote_address,
            default_limits=["200 per day", "50 per hour"]
        )
    
    def setup_headers(self):
        """Setup security headers"""
        @self.app.after_request
        def set_security_headers(response):
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            return response