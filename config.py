"""
Babs AI Trading System - Configuration
Central configuration management for all components
"""
import os


class Config:
    """Application configuration from environment variables"""
    
    # General Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'a_very_secret_key_that_should_be_changed')
    
    # Database - Handle Render's DATABASE_URL format
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///babs_trader.db')
    # Render uses postgres:// but SQLAlchemy needs postgresql://
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # AI API Keys
    OPENAI_KEY = os.environ.get('OPENAI_KEY')
    OPENROUTER_KEY = os.environ.get('OPENROUTER_KEY')
    BYTZE_KEY = os.environ.get('BYTZE_KEY')
    ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY')
    GOOGLE_GENAI_KEY = os.environ.get('GOOGLE_GENAI_KEY')
    
    # News and Sentiment
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
    
    # Notifications
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
    
    # Premium System
    PREMIUM_INVITES = os.environ.get('PREMIUM_INVITES', '').split(',')
    
    # Deployment
    ENVIRONMENT = os.environ.get('ENVIRONMENT', 'production')
    DEPLOYMENT_PLATFORM = os.environ.get('DEPLOYMENT_PLATFORM', 'render')
    
    # Feature Flags
    MONITORING_ENABLED = os.environ.get('MONITORING_ENABLED', 'True').lower() == 'true'
    AUTO_SCALING = os.environ.get('AUTO_SCALING', 'False').lower() == 'true'
    BACKUP_ENABLED = os.environ.get('BACKUP_ENABLED', 'True').lower() == 'true'
    
    # ML Features - Optional heavy dependencies
    USE_TENSORFLOW = os.environ.get('USE_TENSORFLOW', 'False').lower() == 'true'
    USE_TRANSFORMERS = os.environ.get('USE_TRANSFORMERS', 'False').lower() == 'true'
    USE_TALIB = os.environ.get('USE_TALIB', 'False').lower() == 'true'
    USE_BIOMETRIC = os.environ.get('USE_BIOMETRIC', 'False').lower() == 'true'
