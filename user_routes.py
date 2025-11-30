from flask import jsonify, request, session
from user_system import UserManager, BonusCodeManager, UserPreferencesManager, AuthMethod, SubscriptionTier, is_premium_user, log_premium_usage
from datetime import datetime
import base64

# Initialize user systems
user_manager = UserManager()
bonus_code_manager = BonusCodeManager()
preferences_manager = UserPreferencesManager()

def setup_user_routes(app):
    
    @app.route('/api/user/premium/status', methods=['GET'])
    def get_premium_status():
        """Check if the authenticated user is a premium user."""
        # NOTE: In a real app, user info would come from a verified token/session.
        # For now, we'll use a placeholder to get the email from the request.
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        user = user_manager.verify_auth_token(token)
        
        if not user:
            return jsonify({"is_premium": False, "message": "Authentication required"}), 401

        if is_premium_user(user.email):
            log_premium_usage(user.email, "status_check")
            return jsonify({"is_premium": True, "message": "Premium access granted"}), 200
        else:
            return jsonify({"is_premium": False, "message": "Standard access"}), 200

    @app.route('/api/admin/premium-invites', methods=['GET', 'POST', 'DELETE'])
    def manage_premium_invites():
        """Admin route to view, add, or remove premium invites."""
        # NOTE: This route requires strong admin authentication, which is omitted for this task.
        # We will assume a valid admin token is present for now.
        from app import db, PremiumInvite # Import here to avoid circular dependency

        # Authentication check placeholder (e.g., check if user is admin)
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        admin_user = user_manager.verify_auth_token(token)
        if not admin_user:
             return jsonify({"error": "Admin authentication required"}), 401
        # if admin_user.subscription_tier != SubscriptionTier.ADMIN: return 403

        if request.method == 'GET':
            invites = PremiumInvite.query.all()
            return jsonify([{
                "email": invite.email,
                "invited_on": invite.invited_on.isoformat(),
                "used_count": invite.used_count
            } for invite in invites]), 200

        data = request.get_json()
        email = data.get('email')
        if not email:
            return jsonify({"error": "Email is required"}), 400

        if request.method == 'POST':
            if PremiumInvite.query.filter_by(email=email).first():
                return jsonify({"message": f"Email {email} is already invited"}), 200
            
            new_invite = PremiumInvite(email=email)
            db.session.add(new_invite)
            db.session.commit()
            return jsonify({"message": f"Premium invite added for {email}"}), 201

        if request.method == 'DELETE':
            invite = PremiumInvite.query.filter_by(email=email).first()
            if invite:
                db.session.delete(invite)
                db.session.commit()
                return jsonify({"message": f"Premium invite removed for {email}"}), 200
            else:
                return jsonify({"error": f"Email {email} not found in invites"}), 404
    
    @app.route('/api/auth/register', methods=['POST'])
    
    @app.route('/api/auth/register', methods=['POST'])
    def register_user():
        """Register a new user"""
        try:
            data = request.json
            email = data.get('email')
            username = data.get('username')
            password = data.get('password')
            
            if not email or not username or not password:
                return jsonify({"error": "Email, username, and password are required"}), 400
            
            user = user_manager.create_user(email, username, password)
            
            if user:
                token = user_manager.generate_auth_token(user)
                return jsonify({
                    "status": "success",
                    "message": "User registered successfully",
                    "token": token,
                    "user": {
                        "id": user.id,
                        "email": user.email,
                        "username": user.username,
                        "subscription_tier": user.subscription_tier.value,
                        "created_at": user.created_at.isoformat()
                    }
                })
            else:
                return jsonify({"error": "User already exists"}), 400
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/auth/login', methods=['POST'])
    def login_user():
        """Login user with email/username and password"""
        try:
            data = request.json
            identifier = data.get('identifier')
            password = data.get('password')
            ip_address = request.remote_addr
            
            if not identifier or not password:
                return jsonify({"error": "Identifier and password are required"}), 400
            
            user = user_manager.authenticate_user(identifier, password, ip_address)
            
            if user:
                token = user_manager.generate_auth_token(user)
                return jsonify({
                    "status": "success",
                    "message": "Login successful",
                    "token": token,
                    "user": {
                        "id": user.id,
                        "email": user.email,
                        "username": user.username,
                        "subscription_tier": user.subscription_tier.value,
                        "last_login": user.last_login.isoformat() if user.last_login else None
                    }
                })
            else:
                return jsonify({"error": "Invalid credentials"}), 401
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/auth/biometric-login', methods=['POST'])
    def biometric_login():
        """Login user using biometric authentication"""
        try:
            data = request.json
            user_id = data.get('user_id')
            image_data = data.get('image_data')
            ip_address = request.remote_addr
            
            if not user_id or not image_data:
                return jsonify({"error": "User ID and image data are required"}), 400
            
            success = user_manager.authenticate_biometric(user_id, image_data, ip_address)
            
            if success:
                user = user_manager.get_user_by_id(user_id)
                token = user_manager.generate_auth_token(user)
                return jsonify({
                    "status": "success",
                    "message": "Biometric login successful",
                    "token": token,
                    "user": {
                        "id": user.id,
                        "email": user.email,
                        "username": user.username,
                        "subscription_tier": user.subscription_tier.value
                    }
                })
            else:
                return jsonify({"error": "Biometric authentication failed"}), 401
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/auth/voice-login', methods=['POST'])
    def voice_login():
        """Login user using voice authentication"""
        try:
            data = request.json
            user_id = data.get('user_id')
            audio_data = data.get('audio_data')
            ip_address = request.remote_addr
            
            if not user_id or not audio_data:
                return jsonify({"error": "User ID and audio data are required"}), 400
            
            success = user_manager.authenticate_voice(user_id, audio_data, ip_address)
            
            if success:
                user = user_manager.get_user_by_id(user_id)
                token = user_manager.generate_auth_token(user)
                return jsonify({
                    "status": "success",
                    "message": "Voice login successful",
                    "token": token,
                    "user": {
                        "id": user.id,
                        "email": user.email,
                        "username": user.username,
                        "subscription_tier": user.subscription_tier.value
                    }
                })
            else:
                return jsonify({"error": "Voice authentication failed"}), 401
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/auth/enroll-biometric', methods=['POST'])
    def enroll_biometric():
        """Enroll user's biometric data"""
        try:
            data = request.json
            user_id = data.get('user_id')
            image_data = data.get('image_data')
            
            if not user_id or not image_data:
                return jsonify({"error": "User ID and image data are required"}), 400
            
            success = user_manager.enroll_biometric(user_id, image_data)
            
            if success:
                return jsonify({
                    "status": "success",
                    "message": "Biometric data enrolled successfully"
                })
            else:
                return jsonify({"error": "Biometric enrollment failed"}), 400
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/auth/enroll-voice', methods=['POST'])
    def enroll_voice():
        """Enroll user's voice profile"""
        try:
            data = request.json
            user_id = data.get('user_id')
            audio_samples = data.get('audio_samples', [])
            
            if not user_id or not audio_samples:
                return jsonify({"error": "User ID and audio samples are required"}), 400
            
            success = user_manager.enroll_voice(user_id, audio_samples)
            
            if success:
                return jsonify({
                    "status": "success",
                    "message": "Voice profile enrolled successfully"
                })
            else:
                return jsonify({"error": "Voice enrollment failed"}), 400
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/user/profile', methods=['GET'])
    def get_user_profile():
        """Get current user profile"""
        try:
            token = request.headers.get('Authorization', '').replace('Bearer ', '')
            user = user_manager.verify_auth_token(token)
            
            if not user:
                return jsonify({"error": "Invalid token"}), 401
            
            return jsonify({
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "username": user.username,
                    "subscription_tier": user.subscription_tier.value,
                    "status": user.status.value,
                    "created_at": user.created_at.isoformat(),
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                    "subscription_expiry": user.subscription_expiry.isoformat() if user.subscription_expiry else None,
                    "has_biometric": user.biometric_data is not None,
                    "has_voice": user.voice_profile is not None
                }
            })
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/user/preferences', methods=['GET'])
    def get_user_preferences():
        """Get user preferences"""
        try:
            token = request.headers.get('Authorization', '').replace('Bearer ', '')
            user = user_manager.verify_auth_token(token)
            
            if not user:
                return jsonify({"error": "Invalid token"}), 401
            
            preferences = preferences_manager.get_preferences(user.id)
            
            return jsonify({
                "preferences": preferences
            })
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/user/preferences', methods=['PUT'])
    def update_user_preferences():
        """Update user preferences"""
        try:
            token = request.headers.get('Authorization', '').replace('Bearer ', '')
            user = user_manager.verify_auth_token(token)
            
            if not user:
                return jsonify({"error": "Invalid token"}), 401
            
            data = request.json
            preferences = data.get('preferences', {})
            
            success = preferences_manager.update_preferences(user.id, preferences)
            
            if success:
                return jsonify({
                    "status": "success",
                    "message": "Preferences updated successfully"
                })
            else:
                return jsonify({"error": "Failed to update preferences"}), 400
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/user/upgrade', methods=['POST'])
    def upgrade_subscription():
        """Upgrade user subscription"""
        try:
            token = request.headers.get('Authorization', '').replace('Bearer ', '')
            user = user_manager.verify_auth_token(token)
            
            if not user:
                return jsonify({"error": "Invalid token"}), 401
            
            data = request.json
            tier = data.get('tier', 'premium')
            duration_days = data.get('duration_days', 30)
            
            user_manager.upgrade_subscription(user.id, SubscriptionTier(tier), duration_days)
            
            return jsonify({
                "status": "success",
                "message": f"Subscription upgraded to {tier}"
            })
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/bonus-codes/generate', methods=['POST'])
    def generate_bonus_code():
        """Generate a new bonus code"""
        try:
            token = request.headers.get('Authorization', '').replace('Bearer ', '')
            user = user_manager.verify_auth_token(token)
            
            if not user:
                return jsonify({"error": "Invalid token"}), 401
            
            # Only premium users can generate bonus codes
            if user.subscription_tier != SubscriptionTier.PREMIUM:
                return jsonify({"error": "Premium subscription required"}), 403
            
            data = request.json
            max_uses = data.get('max_uses', 5)
            expiry_days = data.get('expiry_days', 30)
            
            bonus_code = bonus_code_manager.generate_bonus_code(user.id, max_uses, expiry_days)
            
            if bonus_code:
                return jsonify({
                    "status": "success",
                    "bonus_code": {
                        "code": bonus_code.code,
                        "max_uses": bonus_code.max_uses,
                        "expires_at": bonus_code.expires_at.isoformat()
                    }
                })
            else:
                return jsonify({"error": "Failed to generate bonus code"}), 400
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/bonus-codes/use', methods=['POST'])
    def use_bonus_code():
        """Use a bonus code to upgrade subscription"""
        try:
            token = request.headers.get('Authorization', '').replace('Bearer ', '')
            user = user_manager.verify_auth_token(token)
            
            if not user:
                return jsonify({"error": "Invalid token"}), 401
            
            data = request.json
            code = data.get('code')
            
            if not code:
                return jsonify({"error": "Bonus code is required"}), 400
            
            success = bonus_code_manager.use_bonus_code(code, user.id)
            
            if success:
                return jsonify({
                    "status": "success",
                    "message": "Bonus code applied successfully"
                })
            else:
                return jsonify({"error": "Invalid or expired bonus code"}), 400
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/bonus-codes', methods=['GET'])
    def get_bonus_codes():
        """Get user's generated bonus codes"""
        try:
            token = request.headers.get('Authorization', '').replace('Bearer ', '')
            user = user_manager.verify_auth_token(token)
            
            if not user:
                return jsonify({"error": "Invalid token"}), 401
            
            bonus_codes = bonus_code_manager.get_bonus_codes(user.id)
            
            codes_data = []
            for code in bonus_codes:
                codes_data.append({
                    "code": code.code,
                    "max_uses": code.max_uses,
                    "used_count": code.used_count,
                    "expires_at": code.expires_at.isoformat(),
                    "is_active": code.is_active
                })
            
            return jsonify({
                "bonus_codes": codes_data
            })
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/user/stats', methods=['GET'])
    def get_user_stats():
        """Get user statistics"""
        try:
            token = request.headers.get('Authorization', '').replace('Bearer ', '')
            user = user_manager.verify_auth_token(token)
            
            if not user:
                return jsonify({"error": "Invalid token"}), 401
            
            # Get login statistics
            conn = user_manager._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM login_attempts 
                WHERE user_id = ? AND success = TRUE
            ''', (user.id,))
            successful_logins = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT method, COUNT(*) FROM login_attempts 
                WHERE user_id = ? AND success = TRUE
                GROUP BY method
            ''', (user.id,))
            login_methods = cursor.fetchall()
            
            conn.close()
            
            return jsonify({
                "stats": {
                    "successful_logins": successful_logins,
                    "login_methods": {method: count for method, count in login_methods},
                    "account_age_days": (datetime.now() - user.created_at).days,
                    "subscription_status": user.subscription_tier.value
                }
            })
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

def get_user_from_token(request):
    """Utility function to get user from auth token"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    return user_manager.verify_auth_token(token)