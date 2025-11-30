import asyncio
import aiohttp
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
from telegram import Bot
from telegram.error import TelegramError
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import websockets
import jwt

class NotificationPriority(Enum):
    CRITICAL = "critical"     # Immediate action required
    HIGH = "high"             # Important trading signals
    MEDIUM = "medium"         # Market updates, performance alerts
    LOW = "low"               # System status, routine updates

class NotificationChannel(Enum):
    WEB_DASHBOARD = "web_dashboard"
    MOBILE_PUSH = "mobile_push"
    TELEGRAM = "telegram"
    EMAIL = "email"
    SMS = "sms"
    VOICE = "voice"
    WEBHOOK = "webhook"

class NotificationType(Enum):
    TRADING_SIGNAL = "trading_signal"
    MARKET_ALERT = "market_alert"
    PERFORMANCE_UPDATE = "performance_update"
    SYSTEM_STATUS = "system_status"
    RISK_WARNING = "risk_warning"
    LEARNING_INSIGHT = "learning_insight"
    MAINTENANCE = "maintenance"

@dataclass
class Notification:
    notification_id: str
    timestamp: datetime
    notification_type: NotificationType
    priority: NotificationPriority
    title: str
    message: str
    data: Dict[str, Any]
    channels: List[NotificationChannel]
    user_groups: List[str]
    expiration: datetime
    delivery_status: Dict[NotificationChannel, bool]
    read_status: bool
    actions: List[Dict]

@dataclass
class UserPreferences:
    user_id: str
    enabled_channels: List[NotificationChannel]
    quiet_hours: List[Tuple[int, int]]  # [(start_hour, end_hour)]
    priority_threshold: NotificationPriority
    symbol_filters: List[str]
    notification_types: List[NotificationType]
    mobile_tokens: List[str]
    telegram_chat_id: Optional[str]
    email: Optional[str]
    phone: Optional[str]

class NotificationOrchestrator:
    def __init__(self):
        self.active_notifications = []
        self.user_preferences = {}
        self.delivery_queue = asyncio.Queue()
        self.fallback_strategies = {}
        self.rate_limits = {}
        self.init_notification_system()
    
    def init_notification_system(self):
        """Initialize notification system"""
        self.setup_notification_database()
        self.load_user_preferences()
        self.setup_delivery_channels()
        self.start_notification_processor()
        self.start_health_monitoring()
        
        logging.info("Notification Orchestrator initialized")
    
    def setup_notification_database(self):
        """Setup notification database tables"""
        conn = sqlite3.connect('notifications.db')
        cursor = conn.cursor()
        
        # Notifications table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                notification_id TEXT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                notification_type TEXT NOT NULL,
                priority TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                data TEXT NOT NULL,
                channels TEXT NOT NULL,
                user_groups TEXT NOT NULL,
                expiration DATETIME NOT NULL,
                delivery_status TEXT NOT NULL,
                read_status BOOLEAN DEFAULT FALSE,
                actions TEXT NOT NULL
            )
        ''')
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                enabled_channels TEXT NOT NULL,
                quiet_hours TEXT NOT NULL,
                priority_threshold TEXT NOT NULL,
                symbol_filters TEXT NOT NULL,
                notification_types TEXT NOT NULL,
                mobile_tokens TEXT NOT NULL,
                telegram_chat_id TEXT,
                email TEXT,
                phone TEXT
            )
        ''')
        
        # Delivery logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS delivery_logs (
                log_id TEXT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                notification_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_user_preferences(self):
        """Load user notification preferences"""
        try:
            conn = sqlite3.connect('notifications.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM user_preferences')
            preferences_data = cursor.fetchall()
            
            for pref_data in preferences_data:
                user_pref = UserPreferences(
                    user_id=pref_data[0],
                    enabled_channels=[NotificationChannel(ch) for ch in json.loads(pref_data[1])],
                    quiet_hours=[tuple(h) for h in json.loads(pref_data[2])],
                    priority_threshold=NotificationPriority(pref_data[3]),
                    symbol_filters=json.loads(pref_data[4]),
                    notification_types=[NotificationType(nt) for nt in json.loads(pref_data[5])],
                    mobile_tokens=json.loads(pref_data[6]),
                    telegram_chat_id=pref_data[7],
                    email=pref_data[8],
                    phone=pref_data[9]
                )
                self.user_preferences[pref_data[0]] = user_pref
            
            conn.close()
            logging.info(f"Loaded preferences for {len(self.user_preferences)} users")
            
        except Exception as e:
            logging.error(f"Error loading user preferences: {e}")
    
    def setup_delivery_channels(self):
        """Setup delivery channels with configuration"""
        self.channel_configs = {
            NotificationChannel.WEB_DASHBOARD: {
                'enabled': True,
                'rate_limit': 100,  # messages per minute
                'retry_attempts': 3,
                'timeout': 10,
                'fallback': NotificationChannel.MOBILE_PUSH
            },
            NotificationChannel.MOBILE_PUSH: {
                'enabled': True,
                'rate_limit': 50,
                'retry_attempts': 5,
                'timeout': 30,
                'fallback': NotificationChannel.TELEGRAM
            },
            NotificationChannel.TELEGRAM: {
                'enabled': True,
                'rate_limit': 30,
                'retry_attempts': 3,
                'timeout': 15,
                'fallback': NotificationChannel.EMAIL
            },
            NotificationChannel.EMAIL: {
                'enabled': True,
                'rate_limit': 20,
                'retry_attempts': 2,
                'timeout': 60,
                'fallback': NotificationChannel.SMS
            },
            NotificationChannel.SMS: {
                'enabled': False,  # Requires paid service
                'rate_limit': 10,
                'retry_attempts': 1,
                'timeout': 30,
                'fallback': None
            },
            NotificationChannel.VOICE: {
                'enabled': False,  # Requires paid service
                'rate_limit': 5,
                'retry_attempts': 1,
                'timeout': 45,
                'fallback': None
            },
            NotificationChannel.WEBHOOK: {
                'enabled': True,
                'rate_limit': 100,
                'retry_attempts': 3,
                'timeout': 10,
                'fallback': None
            }
        }
    
    def start_notification_processor(self):
        """Start background notification processor"""
        def process_notifications():
            while True:
                try:
                    # Process delivery queue
                    if not self.delivery_queue.empty():
                        notification = self.delivery_queue.get_nowait()
                        self.deliver_notification(notification)
                    
                    # Clean up expired notifications
                    self.cleanup_expired_notifications()
                    
                    # Retry failed deliveries
                    self.retry_failed_deliveries()
                    
                    threading.Event().wait(0.1)  # Process every 100ms
                    
                except Exception as e:
                    logging.error(f"Error in notification processor: {e}")
                    threading.Event().wait(5)  # Wait 5 seconds on error
        
        processor_thread = threading.Thread(target=process_notifications, daemon=True)
        processor_thread.start()
    
    def start_health_monitoring(self):
        """Start delivery channel health monitoring"""
        def monitor_channels():
            while True:
                try:
                    self.check_channel_health()
                    threading.Event().wait(60)  # Check every minute
                except Exception as e:
                    logging.error(f"Error in channel monitoring: {e}")
                    threading.Event().wait(300)  # Wait 5 minutes on error
        
        monitor_thread = threading.Thread(target=monitor_channels, daemon=True)
        monitor_thread.start()
    
    async def create_notification(self, notification_type: NotificationType,
                                priority: NotificationPriority,
                                title: str, message: str, data: Dict[str, Any],
                                user_groups: List[str] = None,
                                channels: List[NotificationChannel] = None,
                                expiration_minutes: int = 1440) -> Notification:
        """Create and queue a new notification"""
        try:
            notification_id = f"notif_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Determine channels if not specified
            if channels is None:
                channels = self.determine_delivery_channels(priority, notification_type)
            
            # Set expiration
            expiration = datetime.now() + timedelta(minutes=expiration_minutes)
            
            # Determine actions based on notification type
            actions = self.determine_notification_actions(notification_type, data)
            
            notification = Notification(
                notification_id=notification_id,
                timestamp=datetime.now(),
                notification_type=notification_type,
                priority=priority,
                title=title,
                message=message,
                data=data,
                channels=channels,
                user_groups=user_groups or ['all'],
                expiration=expiration,
                delivery_status={channel: False for channel in channels},
                read_status=False,
                actions=actions
            )
            
            # Add to delivery queue
            await self.delivery_queue.put(notification)
            
            # Save to database
            self.save_notification(notification)
            
            logging.info(f"Created notification: {notification_id} - {title}")
            
            return notification
            
        except Exception as e:
            logging.error(f"Error creating notification: {e}")
            return None
    
    def determine_delivery_channels(self, priority: NotificationPriority,
                                  notification_type: NotificationType) -> List[NotificationChannel]:
        """Determine appropriate delivery channels based on priority and type"""
        base_channels = []
        
        # Base channels for all notifications
        base_channels.append(NotificationChannel.WEB_DASHBOARD)
        
        # Add channels based on priority
        if priority in [NotificationPriority.CRITICAL, NotificationPriority.HIGH]:
            base_channels.extend([
                NotificationChannel.MOBILE_PUSH,
                NotificationChannel.TELEGRAM
            ])
        
        if priority == NotificationPriority.CRITICAL:
            base_channels.extend([
                NotificationChannel.EMAIL,
                NotificationChannel.SMS
            ])
        
        # Add channels based on notification type
        if notification_type == NotificationType.TRADING_SIGNAL:
            base_channels.append(NotificationChannel.MOBILE_PUSH)
        
        if notification_type == NotificationType.RISK_WARNING:
            base_channels.append(NotificationChannel.TELEGRAM)
        
        # Remove duplicates and filter enabled channels
        unique_channels = list(set(base_channels))
        enabled_channels = [ch for ch in unique_channels 
                          if self.channel_configs[ch]['enabled']]
        
        return enabled_channels
    
    def determine_notification_actions(self, notification_type: NotificationType,
                                    data: Dict[str, Any]) -> List[Dict]:
        """Determine actionable items for notification"""
        actions = []
        
        if notification_type == NotificationType.TRADING_SIGNAL:
            actions.extend([
                {
                    "label": "View Signal Details",
                    "action": "open_signal",
                    "data": {"signal_id": data.get('signal_id')}
                },
                {
                    "label": "Open Chart",
                    "action": "open_chart",
                    "data": {"symbol": data.get('symbol')}
                }
            ])
        
        elif notification_type == NotificationType.RISK_WARNING:
            actions.append({
                "label": "Review Risk",
                "action": "open_risk_dashboard",
                "data": {}
            })
        
        elif notification_type == NotificationType.PERFORMANCE_UPDATE:
            actions.append({
                "label": "View Performance",
                "action": "open_performance",
                "data": {}
            })
        
        return actions
    
    def deliver_notification(self, notification: Notification):
        """Deliver notification through all channels"""
        try:
            # Get target users based on groups and filters
            target_users = self.get_target_users(notification.user_groups, notification.data)
            
            for channel in notification.channels:
                if self.channel_configs[channel]['enabled']:
                    # Check rate limits
                    if self.check_rate_limit(channel):
                        # Deliver to each user
                        for user_id in target_users:
                            if self.should_deliver_to_user(user_id, notification, channel):
                                self.deliver_to_user(user_id, notification, channel)
                    else:
                        logging.warning(f"Rate limit exceeded for channel: {channel}")
                        self.use_fallback_channel(notification, channel)
            
            # Update delivery status
            self.update_delivery_status(notification)
            
        except Exception as e:
            logging.error(f"Error delivering notification: {e}")
    
    def get_target_users(self, user_groups: List[str], data: Dict) -> List[str]:
        """Get target users based on groups and notification data"""
        target_users = []
        
        if 'all' in user_groups:
            # All users except those with specific filters
            for user_id, prefs in self.user_preferences.items():
                if self.passes_user_filters(user_id, data):
                    target_users.append(user_id)
        else:
            # Specific user groups (could be: 'premium', 'trial', 'admin', etc.)
            for user_id, prefs in self.user_preferences.items():
                if any(group in user_groups for group in self.get_user_groups(user_id)):
                    if self.passes_user_filters(user_id, data):
                        target_users.append(user_id)
        
        return target_users
    
    def passes_user_filters(self, user_id: str, data: Dict) -> bool:
        """Check if notification passes user's filters"""
        user_prefs = self.user_preferences.get(user_id)
        if not user_prefs:
            return False
        
        # Symbol filter
        symbol = data.get('symbol')
        if symbol and user_prefs.symbol_filters:
            if symbol not in user_prefs.symbol_filters and 'all' not in user_prefs.symbol_filters:
                return False
        
        # Notification type filter
        notification_type = data.get('notification_type')
        if notification_type and user_prefs.notification_types:
            if notification_type not in user_prefs.notification_types:
                return False
        
        # Quiet hours check
        if self.in_quiet_hours(user_prefs.quiet_hours):
            return False
        
        return True
    
    def in_quiet_hours(self, quiet_hours: List[Tuple[int, int]]) -> bool:
        """Check if current time is within user's quiet hours"""
        if not quiet_hours:
            return False
        
        current_hour = datetime.now().hour
        
        for start_hour, end_hour in quiet_hours:
            if start_hour <= end_hour:
                if start_hour <= current_hour < end_hour:
                    return True
            else:  # Overnight quiet hours (e.g., 22 to 6)
                if current_hour >= start_hour or current_hour < end_hour:
                    return True
        
        return False
    
    def get_user_groups(self, user_id: str) -> List[str]:
        """Get user's groups (simplified - in production, use proper user management)"""
        # This would come from your user management system
        # For now, return basic groups based on user_id pattern
        if user_id.startswith('premium_'):
            return ['premium', 'all']
        elif user_id.startswith('admin_'):
            return ['admin', 'all']
        else:
            return ['standard', 'all']
    
    def should_deliver_to_user(self, user_id: str, notification: Notification,
                             channel: NotificationChannel) -> bool:
        """Check if notification should be delivered to user via channel"""
        user_prefs = self.user_preferences.get(user_id)
        if not user_prefs:
            return False
        
        # Check if channel is enabled for user
        if channel not in user_prefs.enabled_channels:
            return False
        
        # Check priority threshold
        priority_order = {
            NotificationPriority.CRITICAL: 4,
            NotificationPriority.HIGH: 3,
            NotificationPriority.MEDIUM: 2,
            NotificationPriority.LOW: 1
        }
        
        user_threshold = priority_order[user_prefs.priority_threshold]
        notification_priority = priority_order[notification.priority]
        
        if notification_priority < user_threshold:
            return False
        
        return True
    
    def deliver_to_user(self, user_id: str, notification: Notification,
                       channel: NotificationChannel):
        """Deliver notification to specific user via channel"""
        try:
            user_prefs = self.user_preferences.get(user_id)
            if not user_prefs:
                return
            
            success = False
            error_msg = None
            
            if channel == NotificationChannel.WEB_DASHBOARD:
                success = self.deliver_web_dashboard(user_id, notification)
            
            elif channel == NotificationChannel.MOBILE_PUSH:
                success = self.deliver_mobile_push(user_id, notification, user_prefs)
            
            elif channel == NotificationChannel.TELEGRAM:
                success = self.deliver_telegram(user_id, notification, user_prefs)
            
            elif channel == NotificationChannel.EMAIL:
                success = self.deliver_email(user_id, notification, user_prefs)
            
            elif channel == NotificationChannel.SMS:
                success = self.deliver_sms(user_id, notification, user_prefs)
            
            elif channel == NotificationChannel.VOICE:
                success = self.deliver_voice(user_id, notification, user_prefs)
            
            elif channel == NotificationChannel.WEBHOOK:
                success = self.deliver_webhook(user_id, notification, user_prefs)
            
            # Log delivery attempt
            self.log_delivery_attempt(
                notification.notification_id, channel, user_id, success, error_msg
            )
            
            if not success:
                logging.warning(f"Failed to deliver notification {notification.notification_id} to {user_id} via {channel}")
                self.retry_delivery(user_id, notification, channel)
            
        except Exception as e:
            logging.error(f"Error delivering to user {user_id} via {channel}: {e}")
            self.log_delivery_attempt(
                notification.notification_id, channel, user_id, False, str(e)
            )
    
    def deliver_web_dashboard(self, user_id: str, notification: Notification) -> bool:
        """Deliver to web dashboard (real-time updates)"""
        try:
            # In production, this would use WebSockets or Server-Sent Events
            # For now, we'll simulate successful delivery
            
            # Store notification for user to retrieve
            self.store_web_notification(user_id, notification)
            
            # Could trigger real-time update via WebSocket
            self.trigger_websocket_update(user_id, notification)
            
            return True
            
        except Exception as e:
            logging.error(f"Error delivering to web dashboard: {e}")
            return False
    
    def deliver_mobile_push(self, user_id: str, notification: Notification,
                          user_prefs: UserPreferences) -> bool:
        """Deliver mobile push notification"""
        try:
            if not user_prefs.mobile_tokens:
                return False
            
            # This would integrate with FCM (Firebase Cloud Messaging) or APNS
            # For now, we'll simulate the integration
            
            payload = {
                'to': user_prefs.mobile_tokens[0],  # Use primary token
                'notification': {
                    'title': notification.title,
                    'body': notification.message,
                    'icon': 'notification_icon',
                    'click_action': 'OPEN_TRADING_APP'
                },
                'data': {
                    'notification_id': notification.notification_id,
                    'type': notification.notification_type.value,
                    'priority': notification.priority.value,
                    'actions': json.dumps(notification.actions),
                    'timestamp': notification.timestamp.isoformat()
                }
            }
            
            # In production, send actual push notification
            # response = requests.post(FCM_URL, json=payload, headers=AUTH_HEADERS)
            # return response.status_code == 200
            
            logging.info(f"Mobile push prepared for {user_id}: {notification.title}")
            return True  # Simulate success
            
        except Exception as e:
            logging.error(f"Error delivering mobile push: {e}")
            return False
    
    def deliver_telegram(self, user_id: str, notification: Notification,
                       user_prefs: UserPreferences) -> bool:
        """Deliver Telegram notification"""
        try:
            if not user_prefs.telegram_chat_id:
                return False
            
            # Initialize Telegram bot
            bot_token = "YOUR_TELEGRAM_BOT_TOKEN"  # Should be in environment variables
            bot = Bot(token=bot_token)
            
            # Format message for Telegram
            message = self.format_telegram_message(notification)
            
            # Send message
            bot.send_message(
                chat_id=user_prefs.telegram_chat_id,
                text=message,
                parse_mode='HTML',
                disable_web_page_preview=True
            )
            
            return True
            
        except TelegramError as e:
            logging.error(f"Telegram delivery error: {e}")
            return False
        except Exception as e:
            logging.error(f"Error delivering Telegram message: {e}")
            return False
    
    def deliver_email(self, user_id: str, notification: Notification,
                    user_prefs: UserPreferences) -> bool:
        """Deliver email notification"""
        try:
            if not user_prefs.email:
                return False
            
            # Create email message
            msg = MimeMultipart()
            msg['From'] = 'trading-system@yourdomain.com'
            msg['To'] = user_prefs.email
            msg['Subject'] = f"[{notification.priority.value.upper()}] {notification.title}"
            
            # Create HTML email body
            html_body = self.format_email_message(notification)
            msg.attach(MimeText(html_body, 'html'))
            
            # Send email (in production, use proper SMTP configuration)
            # with smtplib.SMTP('smtp.yourdomain.com', 587) as server:
            #     server.starttls()
            #     server.login('username', 'password')
            #     server.send_message(msg)
            
            logging.info(f"Email prepared for {user_prefs.email}: {notification.title}")
            return True  # Simulate success
            
        except Exception as e:
            logging.error(f"Error delivering email: {e}")
            return False
    
    def deliver_sms(self, user_id: str, notification: Notification,
                  user_prefs: UserPreferences) -> bool:
        """Deliver SMS notification (requires paid service)"""
        # Implementation would depend on SMS service provider (Twilio, etc.)
        logging.info(f"SMS would be sent to {user_prefs.phone}: {notification.title}")
        return False  # Not implemented
    
    def deliver_voice(self, user_id: str, notification: Notification,
                    user_prefs: UserPreferences) -> bool:
        """Deliver voice notification (requires paid service)"""
        # Implementation would depend on voice service provider
        logging.info(f"Voice call would be made to {user_prefs.phone}: {notification.title}")
        return False  # Not implemented
    
    def deliver_webhook(self, user_id: str, notification: Notification,
                      user_prefs: UserPreferences) -> bool:
        """Deliver webhook notification"""
        try:
            # This would send to configured webhook endpoints
            # For now, we'll simulate the delivery
            
            webhook_url = "USER_WEBHOOK_URL"  # Would come from user preferences
            
            payload = {
                'notification_id': notification.notification_id,
                'timestamp': notification.timestamp.isoformat(),
                'type': notification.notification_type.value,
                'priority': notification.priority.value,
                'title': notification.title,
                'message': notification.message,
                'data': notification.data,
                'actions': notification.actions
            }
            
            # In production:
            # response = requests.post(webhook_url, json=payload, timeout=10)
            # return response.status_code == 200
            
            logging.info(f"Webhook prepared for {user_id}: {notification.title}")
            return True
            
        except Exception as e:
            logging.error(f"Error delivering webhook: {e}")
            return False
    
    def format_telegram_message(self, notification: Notification) -> str:
        """Format notification for Telegram"""
        priority_emoji = {
            NotificationPriority.CRITICAL: "🚨",
            NotificationPriority.HIGH: "⚠️",
            NotificationPriority.MEDIUM: "📢",
            NotificationPriority.LOW: "💬"
        }
        
        emoji = priority_emoji.get(notification.priority, "📨")
        
        message = f"{emoji} <b>{notification.title}</b>\n\n"
        message += f"{notification.message}\n\n"
        
        if notification.data.get('symbol'):
            message += f"<b>Symbol:</b> {notification.data['symbol']}\n"
        
        if notification.data.get('price'):
            message += f"<b>Price:</b> {notification.data['price']}\n"
        
        if notification.actions:
            message += "\n<b>Quick Actions:</b>\n"
            for action in notification.actions:
                message += f"• {action['label']}\n"
        
        message += f"\n<code>ID: {notification.notification_id}</code>"
        
        return message
    
    def format_email_message(self, notification: Notification) -> str:
        """Format notification for email"""
        priority_colors = {
            NotificationPriority.CRITICAL: "#ff4444",
            NotificationPriority.HIGH: "#ff8800",
            NotificationPriority.MEDIUM: "#ffbb33",
            NotificationPriority.LOW: "#33b5e5"
        }
        
        color = priority_colors.get(notification.priority, "#33b5e5")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .header {{ background-color: {color}; color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
                .actions {{ margin-top: 20px; }}
                .action-btn {{ display: inline-block; padding: 10px 15px; margin: 5px; 
                            background-color: {color}; color: white; text-decoration: none; 
                            border-radius: 5px; }}
                .footer {{ margin-top: 20px; padding: 10px; background-color: #f8f9fa; 
                         font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>{notification.title}</h2>
                <p>Priority: {notification.priority.value.upper()}</p>
            </div>
            <div class="content">
                <p>{notification.message}</p>
                
                <div class="actions">
                    <h3>Quick Actions:</h3>
        """
        
        for action in notification.actions:
            html += f'<a href="#" class="action-btn">{action["label"]}</a>'
        
        html += f"""
                </div>
            </div>
            <div class="footer">
                <p>Notification ID: {notification.notification_id}</p>
                <p>Sent: {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <p>AI Trading System - Professional Trading Intelligence</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def store_web_notification(self, user_id: str, notification: Notification):
        """Store notification for web dashboard retrieval"""
        # This would store in a user-specific notifications table
        # For now, we'll simulate storage
        pass
    
    def trigger_websocket_update(self, user_id: str, notification: Notification):
        """Trigger real-time update via WebSocket"""
        # This would send via WebSocket connection
        # For now, we'll simulate the trigger
        pass
    
    def check_rate_limit(self, channel: NotificationChannel) -> bool:
        """Check if channel is within rate limits"""
        current_minute = datetime.now().strftime('%Y-%m-%d %H:%M')
        rate_key = f"{channel.value}_{current_minute}"
        
        current_count = self.rate_limits.get(rate_key, 0)
        limit = self.channel_configs[channel]['rate_limit']
        
        if current_count >= limit:
            return False
        
        self.rate_limits[rate_key] = current_count + 1
        return True
    
    def use_fallback_channel(self, notification: Notification, failed_channel: NotificationChannel):
        """Use fallback channel for delivery"""
        fallback = self.channel_configs[failed_channel]['fallback']
        if fallback and self.channel_configs[fallback]['enabled']:
            logging.info(f"Using fallback channel {fallback} for {failed_channel}")
            notification.channels.append(fallback)
            self.deliver_notification(notification)
    
    def retry_delivery(self, user_id: str, notification: Notification,
                      channel: NotificationChannel):
        """Retry failed delivery with exponential backoff"""
        # This would implement retry logic with backoff
        # For now, we'll log the retry attempt
        logging.info(f"Scheduling retry for {notification.notification_id} to {user_id} via {channel}")
    
    def log_delivery_attempt(self, notification_id: str, channel: NotificationChannel,
                           user_id: str, success: bool, error_msg: str = None):
        """Log delivery attempt"""
        try:
            conn = sqlite3.connect('notifications.db')
            cursor = conn.cursor()
            
            log_id = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            cursor.execute('''
                INSERT INTO delivery_logs 
                (log_id, timestamp, notification_id, channel, user_id, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                log_id,
                datetime.now().isoformat(),
                notification_id,
                channel.value,
                user_id,
                'success' if success else 'failed',
                error_msg
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error logging delivery attempt: {e}")
    
    def update_delivery_status(self, notification: Notification):
        """Update notification delivery status"""
        try:
            conn = sqlite3.connect('notifications.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE notifications 
                SET delivery_status = ?
                WHERE notification_id = ?
            ''', (
                json.dumps({ch.value: status for ch, status in notification.delivery_status.items()}),
                notification.notification_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error updating delivery status: {e}")
    
    def save_notification(self, notification: Notification):
        """Save notification to database"""
        try:
            conn = sqlite3.connect('notifications.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO notifications 
                (notification_id, timestamp, notification_type, priority, title, message,
                 data, channels, user_groups, expiration, delivery_status, actions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                notification.notification_id,
                notification.timestamp.isoformat(),
                notification.notification_type.value,
                notification.priority.value,
                notification.title,
                notification.message,
                json.dumps(notification.data),
                json.dumps([ch.value for ch in notification.channels]),
                json.dumps(notification.user_groups),
                notification.expiration.isoformat(),
                json.dumps({ch.value: status for ch, status in notification.delivery_status.items()}),
                json.dumps(notification.actions)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error saving notification: {e}")
    
    def cleanup_expired_notifications(self):
        """Clean up expired notifications"""
        try:
            conn = sqlite3.connect('notifications.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM notifications 
                WHERE expiration < ?
            ''', (datetime.now().isoformat(),))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error cleaning up expired notifications: {e}")
    
    def retry_failed_deliveries(self):
        """Retry failed deliveries"""
        try:
            conn = sqlite3.connect('notifications.db')
            cursor = conn.cursor()
            
            # Get recently failed deliveries
            cursor.execute('''
                SELECT * FROM delivery_logs 
                WHERE status = 'failed' AND retry_count < 3
                AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 10
            ''', ((datetime.now() - timedelta(hours=1)).isoformat(),))
            
            failed_deliveries = cursor.fetchall()
            
            for delivery in failed_deliveries:
                # Implement retry logic here
                # This would reload the notification and retry delivery
                pass
            
            conn.close()
            
        except Exception as e:
            logging.error(f"Error retrying failed deliveries: {e}")
    
    def check_channel_health(self):
        """Check health of all delivery channels"""
        health_status = {}
        
        for channel, config in self.channel_configs.items():
            if config['enabled']:
                # Simple health check - in production, would test actual connectivity
                health_status[channel] = {
                    'status': 'healthy',
                    'response_time': 0.1,  # Simulated
                    'last_check': datetime.now().isoformat()
                }
        
        return health_status
    
    def get_notification_stats(self) -> Dict:
        """Get notification system statistics"""
        try:
            conn = sqlite3.connect('notifications.db')
            cursor = conn.cursor()
            
            # Total notifications
            cursor.execute('SELECT COUNT(*) FROM notifications')
            total_notifications = cursor.fetchone()[0]
            
            # Delivery success rate
            cursor.execute('SELECT COUNT(*) FROM delivery_logs WHERE status = "success"')
            success_count = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM delivery_logs')
            total_deliveries = cursor.fetchone()[0]
            
            success_rate = success_count / total_deliveries if total_deliveries > 0 else 0
            
            # Recent activity
            cursor.execute('''
                SELECT COUNT(*) FROM notifications 
                WHERE timestamp > ?
            ''', ((datetime.now() - timedelta(hours=24)).isoformat(),))
            recent_notifications = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_notifications': total_notifications,
                'delivery_success_rate': success_rate,
                'recent_activity_24h': recent_notifications,
                'active_channels': len([ch for ch, config in self.channel_configs.items() if config['enabled']]),
                'registered_users': len(self.user_preferences)
            }
            
        except Exception as e:
            logging.error(f"Error getting notification stats: {e}")
            return {}