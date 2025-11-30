"""
Babs AI Trading System - AI Quiz System
Adaptive quiz system for trading education
"""
import sqlite3
import json
from datetime import datetime
from api_connectors import AIModelConnector

# Initialize AI Connector
ai_connector = AIModelConnector()


class QuizProgress:
    """Quiz progress model"""
    def __init__(self, user_email, topic, score=0, total_questions=0, difficulty='beginner', last_attempt=None):
        self.user_email = user_email
        self.topic = topic
        self.score = score
        self.total_questions = total_questions
        self.difficulty = difficulty
        self.last_attempt = last_attempt or datetime.utcnow()


class AIQuizSystem:
    """AI-powered quiz system"""
    
    def __init__(self):
        self.topics = [
            "Order Blocks (OB)", "Fair Value Gaps (FVG)", "Liquidity", 
            "Break of Structure (BOS)", "Market Structure Shift (MSS)", 
            "Imbalance", "Institutional Candles", "Mitigation", "Entry/Exit Setups"
        ]
        self.init_database()

    def init_database(self):
        """Initialize quiz database"""
        conn = sqlite3.connect('quiz.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quiz_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                topic TEXT NOT NULL,
                score INTEGER DEFAULT 0,
                total_questions INTEGER DEFAULT 0,
                difficulty TEXT DEFAULT 'beginner',
                last_attempt DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_email, topic)
            )
        ''')
        
        conn.commit()
        conn.close()

    def get_user_progress(self, user_email):
        """Retrieves the user's quiz progress across all topics."""
        conn = sqlite3.connect('quiz.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT topic, score, total_questions, difficulty, last_attempt 
            FROM quiz_progress WHERE user_email = ?
        ''', (user_email,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            'topic': row[0],
            'score': row[1],
            'total_questions': row[2],
            'difficulty': row[3],
            'last_attempt': row[4]
        } for row in rows]

    def get_next_difficulty(self, score, total_questions):
        """Adapts difficulty based on performance."""
        if total_questions == 0:
            return 'beginner'
        
        percentage = (score / total_questions) * 100
        
        if percentage >= 80:
            return 'expert'
        elif percentage >= 60:
            return 'intermediate'
        else:
            return 'beginner'

    def _get_progress(self, user_email, topic):
        """Get or create progress record"""
        conn = sqlite3.connect('quiz.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT score, total_questions, difficulty FROM quiz_progress 
            WHERE user_email = ? AND topic = ?
        ''', (user_email, topic))
        
        row = cursor.fetchone()
        
        if not row:
            cursor.execute('''
                INSERT INTO quiz_progress (user_email, topic) VALUES (?, ?)
            ''', (user_email, topic))
            conn.commit()
            conn.close()
            return QuizProgress(user_email, topic)
        
        conn.close()
        return QuizProgress(user_email, topic, row[0], row[1], row[2])

    def _update_progress(self, user_email, topic, score, total_questions, difficulty):
        """Update progress record"""
        conn = sqlite3.connect('quiz.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE quiz_progress 
            SET score = ?, total_questions = ?, difficulty = ?, last_attempt = ?
            WHERE user_email = ? AND topic = ?
        ''', (score, total_questions, difficulty, datetime.utcnow(), user_email, topic))
        
        conn.commit()
        conn.close()

    def generate_question(self, user_email, topic):
        """Generates a topic-based question using AI, adapting to user difficulty."""
        progress = self._get_progress(user_email, topic)
        difficulty = progress.difficulty
        
        system_prompt = (
            f"You are Babs AI, an expert trading assistant. Your task is to generate a single, "
            f"{difficulty} level quiz question about '{topic}' in Smart Money Concepts (SMC). "
            f"The question should be concise and require a short, factual answer. "
            f"Do not provide the answer. Format your response as a JSON object with 'question' and 'difficulty' keys."
        )
        
        prompt = f"Generate a {difficulty} level question about {topic}."
        
        try:
            response_text, _ = ai_connector.generate_text(prompt, system_prompt)
            
            # Attempt to parse the JSON response
            try:
                # Clean potential markdown code blocks
                clean_text = response_text.strip()
                if clean_text.startswith('```'):
                    clean_text = clean_text.split('\n', 1)[1]
                if clean_text.endswith('```'):
                    clean_text = clean_text.rsplit('```', 1)[0]
                clean_text = clean_text.strip()
                
                response_json = json.loads(clean_text)
                return {
                    'topic': topic,
                    'difficulty': difficulty,
                    'question': response_json.get('question', "Could not generate question."),
                    'raw_ai_response': response_text
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, use the raw response as the question
                return {
                    'topic': topic,
                    'difficulty': difficulty,
                    'question': response_text,
                    'raw_ai_response': response_text
                }
        except Exception as e:
            print(f"AI question generation failed: {e}")
            # Return a fallback question
            fallback_questions = {
                "Order Blocks (OB)": "What is an Order Block and how do you identify one on a chart?",
                "Fair Value Gaps (FVG)": "What causes a Fair Value Gap to form?",
                "Liquidity": "Where do institutional traders typically hunt for liquidity?",
                "Break of Structure (BOS)": "What confirms a Break of Structure in an uptrend?",
            }
            return {
                'topic': topic,
                'difficulty': difficulty,
                'question': fallback_questions.get(topic, f"Explain the key characteristics of {topic}."),
                'raw_ai_response': None
            }

    def score_answer(self, user_email, topic, question, answer):
        """Scores the user's answer and provides feedback using AI."""
        system_prompt = (
            "You are Babs AI, an expert trading assistant. Your task is to score a user's answer "
            "to a quiz question and provide summarized feedback and correction. "
            "Score the answer as 1 (Correct) or 0 (Incorrect). "
            "Format your response as a JSON object with 'score' (int), 'feedback' (string), and 'correction' (string) keys."
        )
        
        prompt = (
            f"Quiz Topic: {topic}\n"
            f"Question: {question}\n"
            f"User's Answer: {answer}\n"
            f"Score the answer and provide feedback."
        )
        
        try:
            response_text, _ = ai_connector.generate_text(prompt, system_prompt)
            
            try:
                # Clean potential markdown code blocks
                clean_text = response_text.strip()
                if clean_text.startswith('```'):
                    clean_text = clean_text.split('\n', 1)[1]
                if clean_text.endswith('```'):
                    clean_text = clean_text.rsplit('```', 1)[0]
                clean_text = clean_text.strip()
                
                response_json = json.loads(clean_text)
                score = response_json.get('score', 0)
                feedback = response_json.get('feedback', "No feedback provided.")
                correction = response_json.get('correction', "No correction provided.")
            except json.JSONDecodeError:
                # If parsing fails, try to determine score from response
                score = 1 if any(word in response_text.lower() for word in ['correct', 'right', 'good']) else 0
                feedback = response_text
                correction = "Please review the topic materials."
            
            # Update user progress
            progress = self._get_progress(user_email, topic)
            new_total = progress.total_questions + 1
            new_score = progress.score + score
            new_difficulty = self.get_next_difficulty(new_score, new_total)
            
            self._update_progress(user_email, topic, new_score, new_total, new_difficulty)

            return {
                'score': score,
                'feedback': feedback,
                'correction': correction,
                'new_difficulty': new_difficulty
            }
        except Exception as e:
            print(f"AI scoring failed: {e}")
            return {
                'score': 0,
                'feedback': "Scoring failed due to an AI service error.",
                'correction': "Please try again.",
                'new_difficulty': 'beginner'
            }
