# Babs AI Trading System - Backend

A sophisticated AI-powered trading platform with SMC pattern detection, adaptive quizzes, and multi-AI API support.

## 🚀 Quick Deploy to Render

### Option 1: Deploy via Render Dashboard
1. Push this `backend` folder to a GitHub repository
2. Go to [render.com](https://render.com) and create a new Web Service
3. Connect your GitHub repository
4. Render will auto-detect the `render.yaml` configuration
5. Add your environment variables in the Render dashboard

### Option 2: Deploy via render.yaml
1. Push to GitHub
2. Create a new "Blueprint" in Render
3. Point to your repository
4. Render will use the `render.yaml` configuration

## 📁 Project Structure

```
backend/
├── app.py              # Main Flask application
├── config.py           # Configuration management
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
├── .env.example        # Environment variables template
│
├── api_connectors.py   # Multi-AI API with fallback
├── smc_patterns.py     # SMC pattern detection
├── ai_quiz_system.py   # Adaptive quiz system
├── user_system.py      # User & subscription management
├── data_validation.py  # Market data validation
│
├── smc_routes.py       # SMC learning content routes
├── quiz_routes.py      # Quiz system routes
├── user_routes.py      # User management routes
├── sentiment_routes.py # Sentiment analysis routes
└── ...                 # Additional modules
```

## ⚙️ Environment Variables

Set these in your Render dashboard:

| Variable | Required | Description |
|----------|----------|-------------|
| `SECRET_KEY` | Yes | Flask secret key (auto-generated on Render) |
| `OPENROUTER_KEY` | Recommended | OpenRouter API key for AI features |
| `ALPHA_VANTAGE_KEY` | Recommended | For market data |
| `PREMIUM_INVITES` | Optional | Comma-separated premium user emails |
| `FRONTEND_URL` | Optional | Your Vercel frontend URL for CORS |

## 🔗 API Endpoints

### Health & Status
- `GET /health` - Health check
- `GET /` - API info

### Market Data
- `GET /api/market-data/<symbol>` - Get validated market data

### SMC Learning (Premium)
- `GET /api/smc/lessons` - Get SMC lessons
- `GET /api/smc/assets/<filename>` - Get lesson images

### Quiz System (Premium)
- `GET /api/quiz/topics` - Get available topics
- `POST /api/quiz/question` - Generate quiz question
- `POST /api/quiz/submit` - Submit answer
- `GET /api/quiz/progress` - Get user progress

### Sentiment Analysis
- `GET /api/sentiment/<symbol>` - Get sentiment analysis
- `GET /api/market-sentiment-overview` - Market overview
- `GET /api/news-feed` - Get news articles

## 🔧 Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys

# Run development server
python app.py
```

## 📝 Notes

- Heavy ML libraries (TensorFlow, face_recognition, TA-Lib) are disabled by default for faster deployment
- Enable them via environment variables if needed
- The system uses SQLite by default; Render can provide PostgreSQL
- All AI features have fallback mechanisms

## 🔐 Security

- JWT-based authentication
- Password hashing with PBKDF2
- CORS configured for your frontend domain
- Rate limiting recommended for production

## 📄 License

MIT License

## Ownership

- **Owner:** Abdulkadir umar
- **Hugging Face:** [MR-CODESPIKE](https://huggingface.co/MR-CODESPIKE)
