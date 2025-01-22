from api import create_app
from flask_cors import CORS

# Create the Flask application
app = create_app()
CORS(app)  # Enable CORS for all routes

if __name__ == '__main__':
    app.run(debug=True)  # Enable debug mode for development 