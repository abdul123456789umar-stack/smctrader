from flask import jsonify, request
import pandas as pd
from smc_patterns import SMCPatternDetector, FibonacciAnalyzer
import json
from datetime import datetime

# Initialize detectors
smc_detector = SMCPatternDetector()
fib_analyzer = FibonacciAnalyzer()

def setup_chart_routes(app):
    
    @app.route('/api/chart-data/<symbol>')
    def get_chart_data(symbol):
        """Get comprehensive chart data with SMC patterns"""
        try:
            timeframe = request.args.get('timeframe', '1h')
            limit = int(request.args.get('limit', 100))
            
            # Get price data (mock for now - integrate with real data source)
            price_data = generate_mock_price_data(symbol, timeframe, limit)
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Detect SMC patterns
            patterns = smc_detector.detect_all_patterns(df, symbol, timeframe)
            
            # Calculate Fibonacci levels
            recent_high = df['high'].max()
            recent_low = df['low'].min()
            fib_levels = fib_analyzer.calculate_fib_levels(recent_high, recent_low)
            fib_clusters = fib_analyzer.find_fib_clusters(patterns, df['close'].iloc[-1])
            
            response = {
                'symbol': symbol,
                'timeframe': timeframe,
                'price_data': price_data,
                'patterns': {
                    'order_blocks': [ob.__dict__ for ob in patterns['order_blocks']],
                    'fair_value_gaps': [fvg.__dict__ for fvg in patterns['fair_value_gaps']],
                    'break_of_structure': [bos.__dict__ for bos in patterns['break_of_structure']],
                    'liquidity_levels': [ll.__dict__ for ll in patterns['liquidity_levels']],
                    'wyckoff_phases': patterns['wyckoff_phases']
                },
                'fibonacci': {
                    'levels': fib_levels,
                    'clusters': fib_clusters
                },
                'current_price': df['close'].iloc[-1],
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/multi-timeframe-analysis/<symbol>')
    def get_multi_timeframe_analysis(symbol):
        """Get analysis across multiple timeframes"""
        timeframes = ['5m', '15m', '1h', '4h', '1d']
        analysis = {}
        
        for tf in timeframes:
            try:
                price_data = generate_mock_price_data(symbol, tf, 100)
                df = pd.DataFrame(price_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                patterns = smc_detector.detect_all_patterns(df, symbol, tf)
                
                analysis[tf] = {
                    'bias': calculate_timeframe_bias(patterns),
                    'key_levels': extract_key_levels(patterns),
                    'confidence': calculate_analysis_confidence(patterns),
                    'patterns_count': len(patterns['order_blocks']) + len(patterns['fair_value_gaps'])
                }
                
            except Exception as e:
                analysis[tf] = {'error': str(e)}
        
        return jsonify({
            'symbol': symbol,
            'multi_timeframe_analysis': analysis,
            'composite_bias': calculate_composite_bias(analysis)
        })
    
    def generate_mock_price_data(symbol, timeframe, limit):
        """Generate mock price data for demonstration"""
        # In production, replace with real data from your APIs
        import random
        from datetime import datetime, timedelta
        
        data = []
        base_price = 1.0950 if 'EUR' in symbol else 183.50 if 'JPY' in symbol else 1985.0
        volatility = 0.002 if 'EUR' in symbol else 0.5 if 'JPY' in symbol else 10.0
        
        current_time = datetime.now()
        
        for i in range(limit):
            timestamp = current_time - timedelta(hours=limit - i)
            
            open_price = base_price + random.uniform(-volatility, volatility)
            close_price = open_price + random.uniform(-volatility * 2, volatility * 2)
            high = max(open_price, close_price) + abs(random.uniform(0, volatility))
            low = min(open_price, close_price) - abs(random.uniform(0, volatility))
            volume = random.randint(1000, 10000)
            
            data.append({
                'timestamp': timestamp.isoformat(),
                'open': round(open_price, 5),
                'high': round(high, 5),
                'low': round(low, 5),
                'close': round(close_price, 5),
                'volume': volume
            })
            
            base_price = close_price
        
        return data
    
    def calculate_timeframe_bias(patterns):
        """Calculate bullish/bearish bias from patterns"""
        bull_signals = len([ob for ob in patterns['order_blocks'] if ob.direction == 'bullish'])
        bear_signals = len([ob for ob in patterns['order_blocks'] if ob.direction == 'bearish'])
        
        if bull_signals > bear_signals:
            return 'bullish'
        elif bear_signals > bull_signals:
            return 'bearish'
        else:
            return 'neutral'
    
    def extract_key_levels(patterns):
        """Extract key support/resistance levels from patterns"""
        levels = []
        
        # Order Block levels
        for ob in patterns['order_blocks']:
            levels.append({
                'price': (ob.price_high + ob.price_low) / 2,
                'type': 'order_block',
                'direction': ob.direction,
                'strength': ob.strength
            })
        
        # Liquidity levels
        for ll in patterns['liquidity_levels']:
            levels.append({
                'price': ll.price,
                'type': 'liquidity',
                'direction': 'support' if ll.type == 'low' else 'resistance',
                'strength': ll.strength
            })
        
        return sorted(levels, key=lambda x: x['price'])
    
    def calculate_analysis_confidence(patterns):
        """Calculate confidence score for analysis"""
        total_patterns = (len(patterns['order_blocks']) + 
                         len(patterns['fair_value_gaps']) + 
                         len(patterns['break_of_structure']))
        
        if total_patterns == 0:
            return 0.0
        
        avg_strength = (sum(ob.strength for ob in patterns['order_blocks']) +
                       sum(fvg.strength for fvg in patterns['fair_value_gaps'])) / total_patterns
        
        return min(avg_strength * 100, 100.0)
    
    def calculate_composite_bias(analysis):
        """Calculate composite bias across all timeframes"""
        biases = []
        weights = {'5m': 0.1, '15m': 0.2, '1h': 0.3, '4h': 0.3, '1d': 0.1}
        
        for tf, data in analysis.items():
            if 'bias' in data:
                bias_score = 1 if data['bias'] == 'bullish' else -1 if data['bias'] == 'bearish' else 0
                biases.append(bias_score * weights.get(tf, 0.1))
        
        composite = sum(biases)
        if composite > 0.1:
            return 'bullish'
        elif composite < -0.1:
            return 'bearish'
        else:
            return 'neutral'