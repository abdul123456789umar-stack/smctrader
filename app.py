"""
Babs AI Trading System - Main Application
Production-ready Flask application for Render deployment
"""
import os
import sys
from datetime import datetime

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

# Configuration
class Config:
    """Application configuration from environment variables"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'a_very_secret_key_that_should_be_changed')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///babs_trader.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # AI API Keys
    OPENAI_KEY = os.environ.get('OPENAI_KEY')
    OPENROUTER_KEY = os.environ.get('OPENROUTER_KEY')
    BYTZE_KEY = os.environ.get('BYTZE_KEY')
    ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY')
    GOOGLE_GENAI_KEY = os.environ.get('GOOGLE_GENAI_KEY')
    
    # Premium System
    PREMIUM_INVITES = os.environ.get('PREMIUM_INVITES', '').split(',')
    
    # Deployment
    ENVIRONMENT = os.environ.get('ENVIRONMENT', 'production')
    DEPLOYMENT_PLATFORM = os.environ.get('DEPLOYMENT_PLATFORM', 'render')


# Initialize Flask app
app = Flask(__name__, static_folder='frontend', template_folder='frontend')
app.config.from_object(Config)

# Enable CORS for frontend on different domain (Vercel)
CORS(app, origins=[
    "http://localhost:*",
    "https://*.vercel.app",
    "https://*.lovable.app",
    os.environ.get('FRONTEND_URL', '*')
])

# Initialize database
db = SQLAlchemy(app)


# Define the PremiumInvite Model
class PremiumInvite(db.Model):
    """Premium user invite model"""
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    invited_on = db.Column(db.DateTime, default=datetime.utcnow)
    used_count = db.Column(db.Integer, default=0)

    def __repr__(self):
        return f'<PremiumInvite {self.email}>'


def init_db():
    """Initialize database and populate initial invites"""
    with app.app_context():
        db.create_all()
        initial_invites = Config.PREMIUM_INVITES
        for email in initial_invites:
            email = email.strip()
            if email and not PremiumInvite.query.filter_by(email=email).first():
                invite = PremiumInvite(email=email)
                db.session.add(invite)
        db.session.commit()


# Premium user helpers (must be defined before importing routes)
def is_premium_user(email):
    """Check if user has premium access"""
    if not email:
        return False
    with app.app_context():
        invite = PremiumInvite.query.filter_by(email=email).first()
        return invite is not None


def log_premium_usage(email, feature):
    """Log premium feature usage"""
    with app.app_context():
        invite = PremiumInvite.query.filter_by(email=email).first()
        if invite:
            invite.used_count += 1
            db.session.commit()


# Routes
@app.route('/')
def home():
    """Home page"""
    return jsonify({
        "name": "Babs AI Trading System",
        "version": "1.0.0",
        "status": "running",
        "environment": Config.ENVIRONMENT,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/market-data/<symbol>')
def get_market_data(symbol):
    """Multi-source validated market data endpoint"""
    try:
        from api_connectors import MultiSourceDataFetcher
        from data_validation import QuantumDataValidator
        
        data_fetcher = MultiSourceDataFetcher()
        data_validator = QuantumDataValidator()
        
        validated_data = data_fetcher.get_validated_data(symbol)
        return jsonify({
            "symbol": symbol,
            "data": validated_data,
            "validation_score": data_validator.get_confidence_score(validated_data),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/installable')
def pwa_manifest():
    """Serve PWA manifest"""
    return jsonify({
        "name": "Babs AI Trading Pro",
        "short_name": "BabsAI",
        "start_url": "/",
        "display": "standalone",
        "theme_color": "#1e40af",
        "background_color": "#0f172a"
    })


# Register blueprints
def register_blueprints():
    """Register all route blueprints"""
    try:
        from smc_routes import smc_routes
        app.register_blueprint(smc_routes)
    except ImportError as e:
        print(f"Warning: Could not import smc_routes: {e}")
    
    try:
        from quiz_routes import quiz_routes
        app.register_blueprint(quiz_routes)
    except ImportError as e:
        print(f"Warning: Could not import quiz_routes: {e}")


def setup_routes():
    """Setup additional routes"""
    try:
        from user_routes import setup_user_routes
        setup_user_routes(app)
    except ImportError as e:
        print(f"Warning: Could not import user_routes: {e}")
    
    try:
        from ai_routes import setup_ai_routes
        setup_ai_routes(app)
    except ImportError as e:
        print(f"Warning: Could not import ai_routes: {e}")
    
    try:
        from chart_routes import setup_chart_routes
        setup_chart_routes(app)
    except ImportError as e:
        print(f"Warning: Could not import chart_routes: {e}")
    
    try:
        from sentiment_routes import setup_sentiment_routes
        setup_sentiment_routes(app)
    except ImportError as e:
        print(f"Warning: Could not import sentiment_routes: {e}")
    
    try:
        from signal_routes import setup_signal_routes
        setup_signal_routes(app)
    except ImportError as e:
        print(f"Warning: Could not import signal_routes: {e}")
    
    try:
        from learning_routes import setup_learning_routes
        setup_learning_routes(app)
    except ImportError as e:
        print(f"Warning: Could not import learning_routes: {e}")
    
    try:
        from deployment_routes import setup_deployment_routes
        setup_deployment_routes(app)
    except ImportError as e:
        print(f"Warning: Could not import deployment_routes: {e}")


# Serve static files (CSS, JS, images) from the 'frontend' folder
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(app.static_folder, filename)


# Initialize application
def create_app():
    """Application factory"""
    init_db()
    register_blueprints()
    setup_routes()
    return app


# WSGI application entry point
application = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('ENVIRONMENT', 'production') == 'development'
    application.run(host='0.0.0.0', port=port, debug=debug)
