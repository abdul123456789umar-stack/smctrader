import sys
import os
from dotenv import load_dotenv

# Add the project directory to the path
project_home = os.path.abspath(os.path.dirname(__file__))
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Load environment variables from .env file
load_dotenv(os.path.join(project_home, '.env'))

# Import the Flask application instance
from app import app as application
