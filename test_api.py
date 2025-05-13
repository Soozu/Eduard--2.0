from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    Simple test endpoint that returns mock recommendations
    without requiring the actual model to be trained.
    """
    data = request.get_json()
    query_text = data.get('query', '')
    
    if not query_text:
        return jsonify({'error': 'Query is required'}), 400
    
    # Extract simple keywords for demonstration
    detected_city = None
    detected_category = None
    
    city_keywords = {
        'manila': 'Manila',
        'cebu': 'Cebu',
        'boracay': 'Boracay',
        'palawan': 'Palawan',
        'baguio': 'Baguio'
    }
    
    category_keywords = {
        'beach': 'beach resort',
        'museum': 'museum',
        'hiking': 'natural attraction',
        'historical': 'historical site',
        'park': 'theme park',
        'food': 'restaurant'
    }
    
    # Simple keyword extraction
    query_lower = query_text.lower()
    for keyword, city in city_keywords.items():
        if keyword in query_lower:
            detected_city = city
            break
            
    for keyword, category in category_keywords.items():
        if keyword in query_lower:
            detected_category = category
            break
    
    # Create mock recommendations based on query
    recommendations = [
        {
            'name': 'Sample Destination 1',
            'city': detected_city or 'Manila',
            'category': detected_category or 'beach resort',
            'description': 'This is a sample destination description that would match your query for ' + query_text,
            'similarity': 0.95
        },
        {
            'name': 'Sample Destination 2',
            'city': detected_city or 'Cebu',
            'category': detected_category or 'historical site',
            'description': 'Another great place to visit based on your interests. This one has different features.',
            'similarity': 0.85
        },
        {
            'name': 'Sample Destination 3',
            'city': detected_city or 'Palawan',
            'category': detected_category or 'natural attraction',
            'description': 'A third option that might be of interest. Less similar but still worth considering.',
            'similarity': 0.75
        }
    ]
    
    return jsonify({
        'recommendations': recommendations,
        'detected_city': detected_city,
        'detected_category': detected_category
    })

if __name__ == '__main__':
    print("Starting test API server on http://localhost:5000")
    print("This is a TEST server that returns MOCK data for debugging the chat interface.")
    print("To use the real model, run app.py instead after training the model with revised.py")
    app.run(debug=True, port=5000) 