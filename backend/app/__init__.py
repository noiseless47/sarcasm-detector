from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

from app import routes  # This import must be at the end to avoid circular imports 