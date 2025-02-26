from app import app
from app.utils import SarcasmAnalyzer

if __name__ == '__main__':
    analyzer = SarcasmAnalyzer()  # This will load the dataset and train the model
    app.run(debug=True, host='127.0.0.1', port=5000) 