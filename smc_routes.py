"""
Babs AI Trading System - SMC Routes
Routes for Smart Money Concepts learning content
"""
import os
from flask import Blueprint, jsonify, send_from_directory, request
from user_system import is_premium_user, log_premium_usage

smc_routes = Blueprint('smc_routes', __name__)

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_smc_content():
    """Load the SMC content from the markdown file"""
    try:
        # Try multiple possible paths for the content file
        possible_paths = [
            os.path.join(BASE_DIR, 'smc_learning_content.md'),
            os.path.join(BASE_DIR, 'upload', 'smc_learning_content.md'),
            '/home/ubuntu/upload/smc_learning_content.md',
        ]
        
        content = None
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    content = f.read()
                break
        
        if not content:
            return get_default_smc_content()
        
        # Simple parsing to get sections
        sections = []
        current_section = {}
        for line in content.split('\n'):
            if line.startswith('## '):
                if current_section:
                    sections.append(current_section)
                current_section = {'title': line.replace('## ', '').strip(), 'content': []}
            elif current_section:
                current_section['content'].append(line)
        if current_section:
            sections.append(current_section)
        
        # Further parse content to separate definition, explanation, etc.
        parsed_sections = []
        for section in sections:
            title = section['title']
            text = '\n'.join(section['content'])
            
            definition = ""
            explanation = ""
            
            if '### Definition' in text:
                def_start = text.find('### Definition') + len('### Definition')
                def_end = text.find('### Explanation') if '### Explanation' in text else text.find('### Key Characteristics')
                if def_end == -1:
                    def_end = len(text)
                definition = text[def_start:def_end].strip()
            
            if '### Explanation' in text:
                exp_start = text.find('### Explanation') + len('### Explanation')
                exp_end = text.find('### Key Characteristics') if '### Key Characteristics' in text else len(text)
                explanation = text[exp_start:exp_end].strip()

            # Map title to image file
            image_map = {
                "Order Blocks (OB)": "order_block.png",
                "Fair Value Gaps (FVG)": "fair_value_gap.png",
                "Liquidity": "liquidity_sweep.png",
                "Break of Structure (BOS)": "break_of_structure.png",
                "Market Structure Shift (MSS)": "market_structure_shift.png",
                "Imbalance": "fair_value_gap.png",
                "Institutional Candles": "order_block.png",
                "Mitigation": "order_block.png",
                "Entry/Exit Setups": "break_of_structure.png"
            }

            parsed_sections.append({
                'title': title,
                'definition': definition,
                'explanation': explanation,
                'image_url': f'/api/smc/assets/{image_map.get(title, "default.png")}'
            })
            
        return parsed_sections
    except Exception as e:
        print(f"Error loading SMC content: {e}")
        return get_default_smc_content()


def get_default_smc_content():
    """Return default SMC content if file not found"""
    return [
        {
            'title': 'Order Blocks (OB)',
            'definition': 'Order Blocks are zones where institutional traders have placed significant orders.',
            'explanation': 'These zones act as support and resistance areas where price is likely to react.',
            'image_url': '/api/smc/assets/order_block.png'
        },
        {
            'title': 'Fair Value Gaps (FVG)',
            'definition': 'Fair Value Gaps are imbalances in price created by aggressive buying or selling.',
            'explanation': 'Price tends to return to fill these gaps before continuing its trend.',
            'image_url': '/api/smc/assets/fair_value_gap.png'
        },
        {
            'title': 'Break of Structure (BOS)',
            'definition': 'Break of Structure occurs when price breaks a significant swing high or low.',
            'explanation': 'BOS confirms the continuation of the current trend direction.',
            'image_url': '/api/smc/assets/break_of_structure.png'
        },
        {
            'title': 'Liquidity',
            'definition': 'Liquidity represents areas where stop losses are clustered.',
            'explanation': 'Institutional traders hunt these areas to fill their large orders.',
            'image_url': '/api/smc/assets/liquidity_sweep.png'
        }
    ]


SMC_CONTENT = load_smc_content()


@smc_routes.route('/api/smc/lessons', methods=['GET'])
def get_smc_lessons():
    """Returns the list of SMC lessons for premium users."""
    user_email = request.headers.get('X-User-Email')
    
    if not user_email or not is_premium_user(user_email):
        return jsonify({"error": "Premium access required for SMC lessons."}), 403
    
    log_premium_usage(user_email, "SMC_lessons_access")
    
    return jsonify(SMC_CONTENT), 200


@smc_routes.route('/api/smc/assets/<filename>', methods=['GET'])
def get_smc_asset(filename):
    """Serves the SMC image assets."""
    # Try multiple possible asset directories
    possible_dirs = [
        os.path.join(BASE_DIR, 'smc_assets'),
        os.path.join(BASE_DIR, 'upload', 'smc_assets'),
        os.path.join(BASE_DIR, 'static', 'smc_assets'),
    ]
    
    for asset_dir in possible_dirs:
        if os.path.exists(os.path.join(asset_dir, filename)):
            return send_from_directory(asset_dir, filename)
    
    # Return a placeholder if not found
    return jsonify({"error": f"Asset {filename} not found"}), 404
