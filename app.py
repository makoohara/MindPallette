from api import create_app
from flask_cors import CORS

# Create the Flask application
app = create_app()
CORS(app)  # Enable CORS for all routes

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Enable debug mode for development 
