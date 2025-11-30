"""
Babs AI Trading System - User System
User management, authentication, and subscription handling
"""
import sqlite3
import hashlib
import secrets
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
from config import Config

# Conditional imports for biometric features
HAS_BIOMETRIC = False
if Config.USE_BIOMETRIC:
    try:
        import face_recognition
        import speech_recognition as sr
        HAS_BIOMETRIC = True
    except ImportError:
        pass

try:
    import jwt
except ImportError:
    jwt = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image
    import io
    import base64
except ImportError:
    Image = None


class SubscriptionTier(Enum):
    FREE = "free"
    PREMIUM = "premium"


class AuthMethod(Enum):
    PASSWORD = "password"
    BIOMETRIC = "biometric"
    VOICE = "voice"


class UserStatus(Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"


@dataclass
class User:
    id: int
    email: str
    username: str
    subscription_tier: SubscriptionTier
    status: UserStatus
    created_at: datetime
    last_login: datetime
    biometric_data: Optional[str] = None
    voice_profile: Optional[str] = None
    subscription_expiry: Optional[datetime] = None


@dataclass
class BonusCode:
    code: str
    created_by: int
    max_uses: int
    used_count: int
    expires_at: datetime
    is_active: bool


@dataclass
class LoginAttempt:
    user_id: int
    method: AuthMethod
    timestamp: datetime
    success: bool
    ip_address: str


# Import premium helpers from app
def is_premium_user(email):
    """Check if user has premium access"""
    try:
        from app import is_premium_user as app_is_premium
        return app_is_premium(email)
    except ImportError:
        # Fallback to direct DB check
        return _check_premium_direct(email)


def _check_premium_direct(email):
    """Direct database check for premium status"""
    if not email:
        return False
    try:
        conn = sqlite3.connect('babs_trader.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM premium_invite WHERE email = ?', (email,))
        result = cursor.fetchone()
        conn.close()
        return result is not None
    except:
        return False


def log_premium_usage(email, feature):
    """Log premium feature usage"""
    try:
        from app import log_premium_usage as app_log_usage
        return app_log_usage(email, feature)
    except ImportError:
        print(f"Premium usage logged for {email}: {feature}")


class UserManager:
    """User management class"""
    
    def __init__(self):
        self.secret_key = Config.SECRET_KEY
        self.init_database()
    
    def init_database(self):
        """Initialize user database"""
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT,
                subscription_tier TEXT DEFAULT 'free',
                status TEXT DEFAULT 'active',
                biometric_data TEXT,
                voice_profile TEXT,
                subscription_expiry DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME,
                login_count INTEGER DEFAULT 0
            )
        ''')
        
        # Bonus codes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bonus_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                created_by INTEGER NOT NULL,
                max_uses INTEGER DEFAULT 5,
                used_count INTEGER DEFAULT 0,
                expires_at DATETIME,
                is_active BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (created_by) REFERENCES users (id)
            )
        ''')
        
        # Bonus code usage table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bonus_code_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                used_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (code_id) REFERENCES bonus_codes (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Login attempts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS login_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                method TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN NOT NULL,
                ip_address TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INTEGER PRIMARY KEY,
                theme TEXT DEFAULT 'dark',
                notifications_enabled BOOLEAN DEFAULT TRUE,
                risk_level TEXT DEFAULT 'medium',
                favorite_instruments TEXT DEFAULT '[]',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, email: str, username: str, password: str) -> Optional[User]:
        """Create a new user"""
        try:
            password_hash = self.hash_password(password)
            
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (email, username, password_hash, last_login)
                VALUES (?, ?, ?, ?)
            ''', (email, username, password_hash, datetime.now()))
            
            user_id = cursor.lastrowid
            
            # Create default preferences
            cursor.execute('''
                INSERT INTO user_preferences (user_id)
                VALUES (?)
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            
            return self.get_user_by_id(user_id)
            
        except sqlite3.IntegrityError:
            return None  # User already exists
    
    def authenticate_user(self, identifier: str, password: str, ip_address: str = "") -> Optional[User]:
        """Authenticate user with email/username and password"""
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM users 
            WHERE (email = ? OR username = ?) AND status = 'active'
        ''', (identifier, identifier))
        
        user_data = cursor.fetchone()
        
        if not user_data:
            self.record_login_attempt(None, AuthMethod.PASSWORD, False, ip_address)
            conn.close()
            return None
        
        user = self._row_to_user(user_data)
        
        if self.verify_password(password, user_data[3]):  # password_hash field
            # Update last login
            cursor.execute('''
                UPDATE users 
                SET last_login = ?, login_count = login_count + 1
                WHERE id = ?
            ''', (datetime.now(), user.id))
            
            conn.commit()
            conn.close()
            
            self.record_login_attempt(user.id, AuthMethod.PASSWORD, True, ip_address)
            return user
        else:
            conn.close()
            self.record_login_attempt(user.id, AuthMethod.PASSWORD, False, ip_address)
            return None
    
    def generate_auth_token(self, user: User) -> str:
        """Generate JWT authentication token"""
        if not jwt:
            return ""
        payload = {
            'user_id': user.id,
            'email': user.email,
            'subscription_tier': user.subscription_tier.value,
            'exp': datetime.now() + timedelta(days=7)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_auth_token(self, token: str) -> Optional[User]:
        """Verify JWT token and return user"""
        if not jwt:
            return None
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return self.get_user_by_id(payload['user_id'])
        except:
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user_data = cursor.fetchone()
        conn.close()
        
        if user_data:
            return self._row_to_user(user_data)
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user_data = cursor.fetchone()
        conn.close()
        
        if user_data:
            return self._row_to_user(user_data)
        return None
    
    def _row_to_user(self, row) -> User:
        """Convert database row to User object"""
        return User(
            id=row[0],
            email=row[1],
            username=row[2],
            subscription_tier=SubscriptionTier(row[4]) if row[4] else SubscriptionTier.FREE,
            status=UserStatus(row[5]) if row[5] else UserStatus.ACTIVE,
            biometric_data=row[6],
            voice_profile=row[7],
            subscription_expiry=datetime.fromisoformat(row[8]) if row[8] else None,
            created_at=datetime.fromisoformat(row[9]) if row[9] else datetime.now(),
            last_login=datetime.fromisoformat(row[10]) if row[10] else None
        )
    
    def hash_password(self, password: str) -> str:
        """Hash password using secure method"""
        salt = secrets.token_hex(16)
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex() + ':' + salt
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            stored_hash, salt = password_hash.split(':')
            computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
            return computed_hash == stored_hash
        except:
            return False
    
    def record_login_attempt(self, user_id: Optional[int], method: AuthMethod, success: bool, ip_address: str = ""):
        """Record login attempt for security monitoring"""
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO login_attempts (user_id, method, success, ip_address)
            VALUES (?, ?, ?, ?)
        ''', (user_id, method.value, success, ip_address))
        
        conn.commit()
        conn.close()
    
    def upgrade_subscription(self, user_id: int, tier: SubscriptionTier, duration_days: int = 30):
        """Upgrade user subscription"""
        expiry_date = datetime.now() + timedelta(days=duration_days)
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET subscription_tier = ?, subscription_expiry = ?
            WHERE id = ?
        ''', (tier.value, expiry_date, user_id))
        
        conn.commit()
        conn.close()


class BonusCodeManager:
    """Bonus code management"""
    
    def generate_bonus_code(self, user_id: int, max_uses: int = 5, expiry_days: int = 30) -> Optional[BonusCode]:
        """Generate a new bonus code"""
        code = secrets.token_urlsafe(8).upper()
        expires_at = datetime.now() + timedelta(days=expiry_days)
        
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO bonus_codes (code, created_by, max_uses, expires_at)
                VALUES (?, ?, ?, ?)
            ''', (code, user_id, max_uses, expires_at))
            
            conn.commit()
            conn.close()
            
            return BonusCode(
                code=code,
                created_by=user_id,
                max_uses=max_uses,
                used_count=0,
                expires_at=expires_at,
                is_active=True
            )
        except:
            return None
    
    def use_bonus_code(self, code: str, user_id: int) -> bool:
        """Use a bonus code"""
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Check if code exists and is valid
        cursor.execute('''
            SELECT * FROM bonus_codes 
            WHERE code = ? AND is_active = TRUE AND expires_at > ?
        ''', (code, datetime.now()))
        
        code_data = cursor.fetchone()
        
        if not code_data:
            conn.close()
            return False
        
        code_id = code_data[0]
        max_uses = code_data[3]
        used_count = code_data[4]
        
        if used_count >= max_uses:
            conn.close()
            return False
        
        # Check if user already used this code
        cursor.execute('''
            SELECT * FROM bonus_code_usage 
            WHERE code_id = ? AND user_id = ?
        ''', (code_id, user_id))
        
        if cursor.fetchone():
            conn.close()
            return False
        
        # Record usage
        cursor.execute('''
            INSERT INTO bonus_code_usage (code_id, user_id)
            VALUES (?, ?)
        ''', (code_id, user_id))
        
        cursor.execute('''
            UPDATE bonus_codes 
            SET used_count = used_count + 1
            WHERE id = ?
        ''', (code_id,))
        
        # Upgrade user subscription
        cursor.execute('''
            UPDATE users 
            SET subscription_tier = 'premium', subscription_expiry = ?
            WHERE id = ?
        ''', (datetime.now() + timedelta(days=30), user_id))
        
        conn.commit()
        conn.close()
        return True
    
    def get_bonus_codes(self, user_id: int) -> List[BonusCode]:
        """Get bonus codes created by user"""
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM bonus_codes WHERE created_by = ?
        ''', (user_id,))
        
        codes = []
        for row in cursor.fetchall():
            codes.append(BonusCode(
                code=row[1],
                created_by=row[2],
                max_uses=row[3],
                used_count=row[4],
                expires_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
                is_active=bool(row[6])
            ))
        
        conn.close()
        return codes


class UserPreferencesManager:
    """User preferences management"""
    
    def get_preferences(self, user_id: int) -> Dict:
        """Get user preferences"""
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_preferences WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'theme': row[1],
                'notifications_enabled': bool(row[2]),
                'risk_level': row[3],
                'favorite_instruments': json.loads(row[4]) if row[4] else []
            }
        return {}
    
    def update_preferences(self, user_id: int, preferences: Dict) -> bool:
        """Update user preferences"""
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE user_preferences 
                SET theme = ?, notifications_enabled = ?, risk_level = ?, 
                    favorite_instruments = ?, updated_at = ?
                WHERE user_id = ?
            ''', (
                preferences.get('theme', 'dark'),
                preferences.get('notifications_enabled', True),
                preferences.get('risk_level', 'medium'),
                json.dumps(preferences.get('favorite_instruments', [])),
                datetime.now(),
                user_id
            ))
            
            conn.commit()
            conn.close()
            return True
        except:
            return False
