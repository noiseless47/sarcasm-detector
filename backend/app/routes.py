from flask import request, jsonify
from app import app  # Import the app instance from __init__.py
from app.utils import SarcasmAnalyzer
from flask_cors import CORS  # Add this import
import traceback  # Add this import for better error tracking

# Enable CORS with specific options
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

analyzer = SarcasmAnalyzer()  # Updated class name

@app.route('/', methods=['GET', 'POST', 'OPTIONS'])
def index():
    if request.method == 'OPTIONS':
        # Handle preflight request
        return '', 204
    
    if request.method == 'POST':
        try:
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
            
            data = request.get_json()
            if not data or 'text' not in data:
                return jsonify({"error": "No text provided"}), 400
            
            text = data['text']
            if not text or not isinstance(text, str):
                return jsonify({"error": "Invalid text format"}), 400

            print(f"Analyzing text: {text}")  # Debug print
            rating, explanation = analyzer.analyze_text(text)
            
            response = {
                "rating": rating,
                "explanation": explanation
            }
            print(f"Response: {response}")  # Debug print
            return jsonify(response)

        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error processing request: {str(e)}")
            print(f"Traceback: {error_traceback}")
            return jsonify({
                "error": "Internal server error",
                "details": str(e),
                "traceback": error_traceback
            }), 500
            
    return jsonify({"message": "API is running"})  # Return JSON instead of rendering template

@app.route('/clown-detector', methods=['POST'])
def clown_detector():
    # Your logic to process the image and text
    return jsonify({"rating": "75% Clown 🤡", "fun_insult": "Bro, you're one step away from juggling on the streets!"})

# Remove duplicate route
# @app.route('/', methods=['POST'])
# def analyze():
#     text = request.form.get('text')
#     if text:
#         rating, insult = analyzer.analyze_text(text)
#         return jsonify({"rating": rating, "fun_insult": insult}) 