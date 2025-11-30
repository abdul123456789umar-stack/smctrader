"""
Babs AI Trading System - Sentiment Routes
Routes for sentiment analysis endpoints
"""
from flask import jsonify, request
from datetime import datetime, timedelta
from typing import Dict, List

# Import with error handling
try:
    from sentiment_analyzer import SentimentAnalyzer, SentimentIntegration, MarketSentiment
    HAS_SENTIMENT = True
except ImportError as e:
    print(f"Warning: Sentiment analyzer not available: {e}")
    HAS_SENTIMENT = False


def setup_sentiment_routes(app):
    """Setup sentiment analysis routes"""
    
    # Initialize sentiment components if available
    sentiment_analyzer = SentimentAnalyzer() if HAS_SENTIMENT else None
    sentiment_integration = SentimentIntegration() if HAS_SENTIMENT else None
    
    @app.route('/api/sentiment/<symbol>')
    def get_sentiment_analysis(symbol):
        """Get comprehensive sentiment analysis for a symbol"""
        if not HAS_SENTIMENT:
            return jsonify({
                "error": "Sentiment analysis not available",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }), 503
        
        try:
            sentiment = sentiment_analyzer.analyze_news_sentiment(symbol)
            
            response = {
                'symbol': symbol,
                'sentiment': {
                    'overall': sentiment.overall_sentiment,
                    'score': sentiment.sentiment_score,
                    'confidence': sentiment.confidence,
                    'fear_greed_index': sentiment.fear_greed_index,
                    'timestamp': sentiment.timestamp.isoformat()
                },
                'news_metrics': sentiment.news_impact,
                'social_sentiment': sentiment.social_sentiment,
                'recommendations': generate_sentiment_recommendations(sentiment)
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/sentiment-enhanced-signal/<symbol>')
    def get_enhanced_signal(symbol):
        """Get AI signal enhanced with sentiment analysis"""
        if not HAS_SENTIMENT:
            return jsonify({
                "error": "Sentiment analysis not available",
                "symbol": symbol
            }), 503
        
        try:
            # Get sentiment analysis
            sentiment = sentiment_analyzer.analyze_news_sentiment(symbol)
            
            response = {
                'symbol': symbol,
                'sentiment_analysis': {
                    'overall_sentiment': sentiment.overall_sentiment,
                    'sentiment_score': sentiment.sentiment_score,
                    'fear_greed_index': sentiment.fear_greed_index,
                    'news_impact': sentiment.news_impact
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/market-sentiment-overview')
    def get_market_sentiment_overview():
        """Get sentiment overview for all major instruments"""
        if not HAS_SENTIMENT:
            return jsonify({
                "market_wide_sentiment": {"sentiment": "NEUTRAL", "confidence": 0.5},
                "instrument_sentiments": {},
                "timestamp": datetime.now().isoformat()
            })
        
        instruments = [
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD',
            'BTC/USDT', 'ETH/USDT', 'XAU/USD', 'XAG/USD'
        ]
        
        sentiment_overview = {}
        
        for symbol in instruments:
            try:
                sentiment = sentiment_analyzer.analyze_news_sentiment(symbol)
                sentiment_overview[symbol] = {
                    'sentiment': sentiment.overall_sentiment,
                    'score': sentiment.sentiment_score,
                    'confidence': sentiment.confidence,
                    'fear_greed_index': sentiment.fear_greed_index,
                    'news_volume': sentiment.news_impact.get('total_articles', 0)
                }
            except Exception as e:
                sentiment_overview[symbol] = {'error': str(e)}
        
        market_sentiment = calculate_market_wide_sentiment(sentiment_overview)
        
        return jsonify({
            'market_wide_sentiment': market_sentiment,
            'instrument_sentiments': sentiment_overview,
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/api/news-feed')
    def get_news_feed():
        """Get recent news feed for trading dashboard"""
        if not HAS_SENTIMENT:
            return jsonify({
                "symbol": request.args.get('symbol', 'EUR/USD'),
                "articles": [],
                "total_count": 0
            })
        
        try:
            symbol = request.args.get('symbol', 'EUR/USD')
            limit = int(request.args.get('limit', 10))
            
            news_articles = sentiment_analyzer.fetch_financial_news(symbol)
            analyzed_articles = []
            
            for article in news_articles[:limit]:
                analyzed_article = sentiment_analyzer.analyze_article_sentiment(article)
                analyzed_articles.append({
                    'title': analyzed_article.title,
                    'description': analyzed_article.description,
                    'source': analyzed_article.source,
                    'published_at': analyzed_article.published_at.isoformat(),
                    'sentiment': analyzed_article.sentiment_label,
                    'sentiment_score': analyzed_article.sentiment_score,
                    'impact': analyzed_article.impact_level,
                    'url': analyzed_article.url
                })
            
            return jsonify({
                'symbol': symbol,
                'articles': analyzed_articles,
                'total_count': len(analyzed_articles)
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/economic-calendar')
    def get_economic_calendar():
        """Get economic calendar events"""
        if not HAS_SENTIMENT:
            return jsonify({
                "currency": request.args.get('symbol', 'USD'),
                "events": [],
                "high_impact_count": 0,
                "period_days": 7
            })
        
        try:
            symbol = request.args.get('symbol', 'USD')
            days = int(request.args.get('days', 7))
            
            currency = symbol.split('/')[0] if '/' in symbol else symbol
            events = sentiment_analyzer.fetch_economic_events(currency)
            
            end_date = datetime.now() + timedelta(days=days)
            filtered_events = [
                event for event in events 
                if event.date <= end_date
            ]
            
            events_data = []
            for event in filtered_events:
                events_data.append({
                    'event': event.event,
                    'country': event.country,
                    'date': event.date.isoformat(),
                    'impact': event.impact,
                    'previous': event.previous,
                    'forecast': event.forecast,
                    'actual': event.actual,
                    'currency': event.currency
                })
            
            return jsonify({
                'currency': currency,
                'events': events_data,
                'high_impact_count': len([e for e in filtered_events if e.impact == 'HIGH']),
                'period_days': days
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500


def generate_sentiment_recommendations(sentiment) -> List[str]:
    """Generate trading recommendations based on sentiment"""
    recommendations = []
    
    if sentiment.overall_sentiment == 'BULLISH':
        recommendations.append("Consider long positions - positive sentiment alignment")
    elif sentiment.overall_sentiment == 'BEARISH':
        recommendations.append("Consider short positions - negative sentiment alignment")
    
    if sentiment.fear_greed_index > 75:
        recommendations.append("Caution: Extreme greed may indicate market top")
    elif sentiment.fear_greed_index < 25:
        recommendations.append("Opportunity: Extreme fear may indicate market bottom")
    
    if sentiment.news_impact.get('high_impact_articles', 0) > 0:
        recommendations.append("Monitor news closely - high impact events detected")
    
    if sentiment.news_impact.get('total_articles', 0) < 3:
        recommendations.append("Low news volume - technical analysis may be more reliable")
    
    return recommendations


def calculate_market_wide_sentiment(sentiment_overview: Dict) -> Dict:
    """Calculate market-wide sentiment from all instruments"""
    sentiments = [data.get('sentiment') for data in sentiment_overview.values() if 'sentiment' in data]
    scores = [data.get('score', 0) for data in sentiment_overview.values() if 'score' in data]
    
    if not sentiments:
        return {'sentiment': 'NEUTRAL', 'score': 0.0, 'confidence': 0.5}
    
    bullish_count = sentiments.count('BULLISH')
    bearish_count = sentiments.count('BEARISH')
    neutral_count = sentiments.count('NEUTRAL')
    
    total = len(sentiments)
    
    if bullish_count > bearish_count and bullish_count > neutral_count:
        overall_sentiment = 'BULLISH'
    elif bearish_count > bullish_count and bearish_count > neutral_count:
        overall_sentiment = 'BEARISH'
    else:
        overall_sentiment = 'NEUTRAL'
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    max_count = max(bullish_count, bearish_count, neutral_count)
    confidence = max_count / total if total > 0 else 0.5
    
    return {
        'sentiment': overall_sentiment,
        'score': avg_score,
        'confidence': confidence,
        'distribution': {
            'bullish': bullish_count,
            'bearish': bearish_count,
            'neutral': neutral_count
        }
    }
