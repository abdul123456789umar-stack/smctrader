"""
Babs AI Trading System - Quiz Routes
Routes for the adaptive quiz system
"""
from flask import Blueprint, jsonify, request
from user_system import is_premium_user, log_premium_usage
from ai_quiz_system import AIQuizSystem

quiz_routes = Blueprint('quiz_routes', __name__)
quiz_system = AIQuizSystem()


@quiz_routes.route('/api/quiz/topics', methods=['GET'])
def get_quiz_topics():
    """Returns the list of available quiz topics."""
    return jsonify(quiz_system.topics), 200


@quiz_routes.route('/api/quiz/question', methods=['POST'])
def get_quiz_question():
    """Generates a new quiz question for a given topic."""
    data = request.get_json() or {}
    topic = data.get('topic')
    user_email = request.headers.get('X-User-Email')

    if not user_email or not is_premium_user(user_email):
        return jsonify({"error": "Premium access required for AI Quiz."}), 403

    if not topic or topic not in quiz_system.topics:
        return jsonify({"error": "Invalid or missing topic."}), 400

    log_premium_usage(user_email, f"AI_Quiz_question_{topic}")
    
    question_data = quiz_system.generate_question(user_email, topic)
    return jsonify(question_data), 200


@quiz_routes.route('/api/quiz/submit', methods=['POST'])
def submit_quiz_answer():
    """Submits an answer for scoring and progress tracking."""
    data = request.get_json() or {}
    topic = data.get('topic')
    question = data.get('question')
    answer = data.get('answer')
    user_email = request.headers.get('X-User-Email')

    if not user_email or not is_premium_user(user_email):
        return jsonify({"error": "Premium access required for AI Quiz."}), 403

    if not all([topic, question, answer]):
        return jsonify({"error": "Missing topic, question, or answer."}), 400

    log_premium_usage(user_email, f"AI_Quiz_submit_{topic}")

    result = quiz_system.score_answer(user_email, topic, question, answer)
    return jsonify(result), 200


@quiz_routes.route('/api/quiz/progress', methods=['GET'])
def get_user_quiz_progress():
    """Returns the user's quiz progress across all topics."""
    user_email = request.headers.get('X-User-Email')

    if not user_email or not is_premium_user(user_email):
        return jsonify({"error": "Premium access required for AI Quiz progress."}), 403

    progress = quiz_system.get_user_progress(user_email)
    return jsonify(progress), 200
