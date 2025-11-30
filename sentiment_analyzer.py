import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from textblob import TextBlob
import numpy as np
from transformers import pipeline
import sqlite3
import re

@dataclass
class NewsArticle:
    title: str
    description: str
    source: str
    published_at: datetime
    url: str
    sentiment_score: float
    sentiment_label: str
    impact_level: str  # 'HIGH', 'MEDIUM', 'LOW'
    relevant_symbols: List[str]

@dataclass
class EconomicEvent:
    event: str
    country: str
    date: datetime
    impact: str  # 'HIGH', 'MEDIUM', 'LOW'
    previous: float
    forecast: float
    actual: float
    currency: str

@dataclass
class MarketSentiment:
    overall_sentiment: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: float
    sentiment_score: float
    fear_greed_index: float
    news_impact: Dict
    social_sentiment: Dict
    timestamp: datetime

class SentimentAnalyzer:
    def __init__(self):
        self.news_api_key = None  # Set your NewsAPI key
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis"
        )
        self.init_sentiment_models()
        
    def init_sentiment_models(self):
        """Initialize sentiment analysis models"""
        # Financial-specific sentiment lexicon
        self.financial_lexicon = {
            'bullish': 0.8, 'bearish': -0.8, 'rally': 0.7, 'plunge': -0.7,
            'surge': 0.6, 'tumble': -0.6, 'gain': 0.5, 'loss': -0.5,
            'positive': 0.4, 'negative': -0.4, 'strong': 0.3, 'weak': -0.3,
            'beat': 0.6, 'miss': -0.6, 'raise': 0.4, 'cut': -0.4,
            'optimistic': 0.5, 'pessimistic': -0.5, 'growth': 0.4, 'recession': -0.7,
            'inflation': -0.3, 'deflation': -0.2, 'hawkish': -0.4, 'dovish': 0.3
        }
    
    def analyze_news_sentiment(self, symbol: str) -> MarketSentiment:
        """Comprehensive sentiment analysis for a symbol"""
        try:
            # Get news articles
            news_articles = self.fetch_financial_news(symbol)
            
            # Analyze sentiment for each article
            analyzed_articles = []
            for article in news_articles:
                sentiment = self.analyze_article_sentiment(article)
                analyzed_articles.append(sentiment)
            
            # Calculate overall sentiment
            overall_sentiment = self.calculate_overall_sentiment(analyzed_articles)
            
            # Get social sentiment (mock for now)
            social_sentiment = self.get_social_sentiment(symbol)
            
            # Get fear and greed index
            fear_greed = self.get_fear_greed_index()
            
            # Get economic calendar impact
            economic_impact = self.get_economic_calendar_impact(symbol)
            
            return MarketSentiment(
                overall_sentiment=overall_sentiment['label'],
                confidence=overall_sentiment['confidence'],
                sentiment_score=overall_sentiment['score'],
                fear_greed_index=fear_greed,
                news_impact={
                    'total_articles': len(analyzed_articles),
                    'bullish_articles': len([a for a in analyzed_articles if a.sentiment_label == 'BULLISH']),
                    'bearish_articles': len([a for a in analyzed_articles if a.sentiment_label == 'BEARISH']),
                    'high_impact_articles': len([a for a in analyzed_articles if a.impact_level == 'HIGH'])
                },
                social_sentiment=social_sentiment,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            # Return neutral sentiment as fallback
            return MarketSentiment(
                overall_sentiment='NEUTRAL',
                confidence=0.5,
                sentiment_score=0.0,
                fear_greed_index=50.0,
                news_impact={'total_articles': 0, 'bullish_articles': 0, 'bearish_articles': 0, 'high_impact_articles': 0},
                social_sentiment={},
                timestamp=datetime.now()
            )
    
    def fetch_financial_news(self, symbol: str) -> List[NewsArticle]:
        """Fetch financial news for a symbol"""
        try:
            # Map symbols to search terms
            search_terms = self.get_search_terms(symbol)
            
            articles = []
            for term in search_terms:
                # Using free NewsAPI (you need to sign up for API key)
                url = f"https://newsapi.org/v2/everything?q={term}&sortBy=publishedAt&language=en&pageSize=10"
                headers = {"X-Api-Key": self.news_api_key} if self.news_api_key else {}
                
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    for article_data in data.get('articles', []):
                        article = NewsArticle(
                            title=article_data.get('title', ''),
                            description=article_data.get('description', ''),
                            source=article_data.get('source', {}).get('name', ''),
                            published_at=datetime.fromisoformat(article_data.get('publishedAt', '').replace('Z', '+00:00')),
                            url=article_data.get('url', ''),
                            sentiment_score=0.0,
                            sentiment_label='NEUTRAL',
                            impact_level='LOW',
                            relevant_symbols=[symbol]
                        )
                        articles.append(article)
            
            return articles[:20]  # Return top 20 articles
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            # Return mock data for demonstration
            return self.generate_mock_news(symbol)
    
    def get_search_terms(self, symbol: str) -> List[str]:
        """Get search terms for news API"""
        symbol_map = {
            'EUR/USD': ['EUR USD', 'Euro Dollar', 'European Central Bank', 'Federal Reserve'],
            'GBP/USD': ['GBP USD', 'Pound Dollar', 'Bank of England', 'UK economy'],
            'USD/JPY': ['USD JPY', 'Dollar Yen', 'Bank of Japan'],
            'BTC/USDT': ['Bitcoin', 'BTC', 'crypto', 'blockchain'],
            'ETH/USDT': ['Ethereum', 'ETH', 'crypto'],
            'XAU/USD': ['gold', 'XAU', 'precious metals', 'safe haven'],
        }
        return symbol_map.get(symbol, [symbol])
    
    def analyze_article_sentiment(self, article: NewsArticle) -> NewsArticle:
        """Analyze sentiment of a single news article"""
        try:
            # Combine title and description for analysis
            text = f"{article.title}. {article.description}"
            
            # Use transformer model for sentiment analysis
            result = self.sentiment_pipeline(text[:512])[0]  # Limit text length
            sentiment_score = result['score'] * (1 if result['label'] == 'POS' else -1)
            
            # Enhance with financial lexicon
            lexicon_score = self.analyze_with_financial_lexicon(text)
            combined_score = (sentiment_score + lexicon_score) / 2
            
            # Determine sentiment label and impact level
            sentiment_label = self.get_sentiment_label(combined_score)
            impact_level = self.determine_impact_level(text, combined_score)
            
            return NewsArticle(
                title=article.title,
                description=article.description,
                source=article.source,
                published_at=article.published_at,
                url=article.url,
                sentiment_score=combined_score,
                sentiment_label=sentiment_label,
                impact_level=impact_level,
                relevant_symbols=article.relevant_symbols
            )
            
        except Exception as e:
            print(f"Error analyzing article sentiment: {e}")
            return article
    
    def analyze_with_financial_lexicon(self, text: str) -> float:
        """Analyze sentiment using financial-specific lexicon"""
        words = re.findall(r'\b\w+\b', text.lower())
        scores = [self.financial_lexicon.get(word, 0) for word in words]
        
        if scores:
            return np.mean(scores)
        return 0.0
    
    def get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score >= 0.2:
            return 'BULLISH'
        elif score <= -0.2:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def determine_impact_level(self, text: str, score: float) -> str:
        """Determine impact level of news article"""
        text_lower = text.lower()
        
        # High impact keywords
        high_impact_keywords = [
            'rate decision', 'fomc', 'ecb', 'boe', 'boj', 'cpi', 'inflation',
            'non-farm payroll', 'nfp', 'gdp', 'recession', 'crisis', 'war',
            'election', 'brexit', 'default', 'bankruptcy', 'merger', 'acquisition'
        ]
        
        # Medium impact keywords
        medium_impact_keywords = [
            'earnings', 'profit', 'loss', 'revenue', 'forecast', 'outlook',
            'employment', 'unemployment', 'retail sales', 'manufacturing'
        ]
        
        if any(keyword in text_lower for keyword in high_impact_keywords):
            return 'HIGH'
        elif any(keyword in text_lower for keyword in medium_impact_keywords):
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def calculate_overall_sentiment(self, articles: List[NewsArticle]) -> Dict:
        """Calculate overall sentiment from multiple articles"""
        if not articles:
            return {'label': 'NEUTRAL', 'score': 0.0, 'confidence': 0.5}
        
        # Calculate weighted average sentiment
        scores = []
        weights = []
        
        for article in articles:
            weight = 1.0
            if article.impact_level == 'HIGH':
                weight = 3.0
            elif article.impact_level == 'MEDIUM':
                weight = 2.0
            
            scores.append(article.sentiment_score * weight)
            weights.append(weight)
        
        weighted_score = sum(scores) / sum(weights) if weights else 0.0
        
        # Calculate confidence based on number of articles and score consistency
        total_articles = len(articles)
        score_variance = np.var([a.sentiment_score for a in articles]) if articles else 0.0
        confidence = min(total_articles / 10, 1.0) * (1 - score_variance)
        
        return {
            'label': self.get_sentiment_label(weighted_score),
            'score': weighted_score,
            'confidence': confidence
        }
    
    def get_social_sentiment(self, symbol: str) -> Dict:
        """Get social media sentiment (mock implementation)"""
        # In production, integrate with Twitter API, StockTwits, etc.
        return {
            'twitter_sentiment': np.random.uniform(-0.5, 0.5),
            'reddit_sentiment': np.random.uniform(-0.5, 0.5),
            'total_mentions': np.random.randint(50, 500),
            'sentiment_trend': 'increasing' if np.random.random() > 0.5 else 'decreasing'
        }
    
    def get_fear_greed_index(self) -> float:
        """Get fear and greed index (mock implementation)"""
        # In production, integrate with alternative.me API or similar
        return np.random.uniform(20, 80)
    
    def get_economic_calendar_impact(self, symbol: str) -> Dict:
        """Get economic calendar impact for symbol"""
        try:
            events = self.fetch_economic_events(symbol)
            
            high_impact_events = [e for e in events if e.impact == 'HIGH']
            medium_impact_events = [e for e in events if e.impact == 'MEDIUM']
            
            return {
                'high_impact_count': len(high_impact_events),
                'medium_impact_count': len(medium_impact_events),
                'next_high_impact_event': self.get_next_high_impact_event(high_impact_events),
                'events_today': len([e for e in events if e.date.date() == datetime.now().date()])
            }
            
        except Exception as e:
            print(f"Error getting economic calendar: {e}")
            return {'high_impact_count': 0, 'medium_impact_count': 0, 'next_high_impact_event': None, 'events_today': 0}
    
    def fetch_economic_events(self, symbol: str) -> List[EconomicEvent]:
        """Fetch economic calendar events (mock implementation)"""
        # In production, integrate with Forex Factory API or similar
        currency = symbol.split('/')[0] if '/' in symbol else 'USD'
        
        mock_events = [
            EconomicEvent(
                event="CPI Data",
                country="US",
                date=datetime.now() + timedelta(hours=2),
                impact="HIGH",
                previous=3.2,
                forecast=3.1,
                actual=3.0,
                currency="USD"
            ),
            EconomicEvent(
                event="Interest Rate Decision",
                country="EU",
                date=datetime.now() + timedelta(days=1),
                impact="HIGH",
                previous=4.5,
                forecast=4.5,
                actual=4.5,
                currency="EUR"
            )
        ]
        
        return [e for e in mock_events if e.currency == currency]
    
    def get_next_high_impact_event(self, events: List[EconomicEvent]) -> Optional[EconomicEvent]:
        """Get the next high impact economic event"""
        future_events = [e for e in events if e.date > datetime.now()]
        if future_events:
            return min(future_events, key=lambda x: x.date)
        return None
    
    def generate_mock_news(self, symbol: str) -> List[NewsArticle]:
        """Generate mock news articles for demonstration"""
        mock_articles = [
            NewsArticle(
                title=f"{symbol} Shows Strong Bullish Momentum",
                description="Technical analysis indicates continued upward movement for the pair.",
                source="Financial Times",
                published_at=datetime.now() - timedelta(hours=1),
                url="https://example.com/news1",
                sentiment_score=0.7,
                sentiment_label="BULLISH",
                impact_level="MEDIUM",
                relevant_symbols=[symbol]
            ),
            NewsArticle(
                title=f"Market Uncertainty Affects {symbol}",
                description="Recent economic data has created volatility in the markets.",
                source="Bloomberg",
                published_at=datetime.now() - timedelta(hours=3),
                url="https://example.com/news2",
                sentiment_score=-0.3,
                sentiment_label="BEARISH",
                impact_level="LOW",
                relevant_symbols=[symbol]
            )
        ]
        return mock_articles

class SentimentIntegration:
    """Integrate sentiment analysis with AI trading signals"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def adjust_signal_confidence(self, trade_signal, sentiment: MarketSentiment) -> float:
        """Adjust trading signal confidence based on sentiment"""
        base_confidence = trade_signal.confidence
        
        # Sentiment alignment multiplier
        sentiment_multiplier = self.calculate_sentiment_multiplier(trade_signal, sentiment)
        
        # News impact adjustment
        news_impact_adjustment = self.calculate_news_impact_adjustment(sentiment)
        
        # Fear and greed adjustment
        fear_greed_adjustment = self.calculate_fear_greed_adjustment(sentiment.fear_greed_index)
        
        # Economic calendar adjustment
        economic_adjustment = self.calculate_economic_adjustment(sentiment.news_impact)
        
        # Calculate adjusted confidence
        adjusted_confidence = base_confidence * sentiment_multiplier
        adjusted_confidence += news_impact_adjustment
        adjusted_confidence += fear_greed_adjustment
        adjusted_confidence += economic_adjustment
        
        return max(0.0, min(1.0, adjusted_confidence))
    
    def calculate_sentiment_multiplier(self, trade_signal, sentiment: MarketSentiment) -> float:
        """Calculate sentiment alignment multiplier"""
        signal_direction = 1 if trade_signal.direction == 'BUY' else -1
        sentiment_direction = 1 if sentiment.overall_sentiment == 'BULLISH' else -1 if sentiment.overall_sentiment == 'BEARISH' else 0
        
        if signal_direction == sentiment_direction:
            # Positive alignment - boost confidence
            return 1.0 + (sentiment.confidence * 0.3)
        elif sentiment_direction == 0:
            # Neutral sentiment - minor impact
            return 1.0
        else:
            # Negative alignment - reduce confidence
            return 1.0 - (sentiment.confidence * 0.4)
    
    def calculate_news_impact_adjustment(self, sentiment: MarketSentiment) -> float:
        """Calculate adjustment based on news impact"""
        news_impact = sentiment.news_impact
        
        if news_impact['high_impact_articles'] > 0:
            # High impact news - be cautious
            return -0.1
        elif news_impact['total_articles'] > 5:
            # Significant news volume - slight positive
            return 0.05
        else:
            # Normal news environment
            return 0.0
    
    def calculate_fear_greed_adjustment(self, fear_greed_index: float) -> float:
        """Calculate adjustment based on fear and greed index"""
        if fear_greed_index > 80:
            # Extreme greed - be cautious
            return -0.15
        elif fear_greed_index > 60:
            # Greed - slight positive
            return 0.05
        elif fear_greed_index < 20:
            # Extreme fear - potential opportunity
            return 0.1
        elif fear_greed_index < 40:
            # Fear - neutral to slightly positive
            return 0.02
        else:
            # Neutral
            return 0.0
    
    def calculate_economic_adjustment(self, news_impact: Dict) -> float:
        """Calculate adjustment based on economic calendar"""
        if news_impact.get('high_impact_count', 0) > 0:
            # High impact events scheduled - reduce confidence
            return -0.1
        elif news_impact.get('events_today', 0) > 3:
            # Many events today - slight reduction
            return -0.05
        else:
            # Normal economic calendar
            return 0.0
    
    def generate_sentiment_reasoning(self, sentiment: MarketSentiment, original_confidence: float, adjusted_confidence: float) -> List[str]:
        """Generate reasoning for sentiment-based adjustments"""
        reasoning = []
        
        confidence_change = adjusted_confidence - original_confidence
        
        if abs(confidence_change) > 0.1:
            if confidence_change > 0:
                reasoning.append(f"📰 Sentiment Analysis: +{confidence_change:.1%} confidence boost from positive news alignment")
            else:
                reasoning.append(f"📰 Sentiment Analysis: {confidence_change:.1%} confidence reduction from news/sentiment factors")
        
        # Add specific sentiment insights
        if sentiment.overall_sentiment != 'NEUTRAL':
            reasoning.append(f"🎯 Market Sentiment: Overall {sentiment.overall_sentiment.lower()} bias ({sentiment.confidence:.0%} confidence)")
        
        if sentiment.fear_greed_index > 70:
            reasoning.append("⚠️  Market Psychology: Extreme greed detected - increased volatility risk")
        elif sentiment.fear_greed_index < 30:
            reasoning.append("💎 Market Psychology: Fear sentiment - potential buying opportunity")
        
        if sentiment.news_impact.get('high_impact_articles', 0) > 0:
            reasoning.append("📊 News Impact: High-impact news detected - monitor for volatility")
        
        return reasoning