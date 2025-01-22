from api import create_app
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    # Create the Flask application
    app = create_app()
    CORS(app)  # Enable CORS for all routes
    
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)
except Exception as e:
    logger.error(f"Failed to create app: {str(e)}")
    raise 
