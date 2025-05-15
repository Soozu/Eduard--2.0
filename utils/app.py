from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import torch
from transformers import RobertaTokenizer
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import uuid
import requests
from datetime import datetime
import json
from collections import defaultdict, Counter
import time
import database as db
from auth_routes import auth_bp
import re
import importlib.util
import smtplib
import random
import string
from email_service import send_ticket_email

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Import from revised.py using importlib to avoid keyword issue with try.py
try:
    # Use importlib.util to load the module without using the 'try' keyword
    spec = importlib.util.spec_from_file_location("travel_model", "try.py")
    travel_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(travel_model)
    
    # Import the required components
    DestinationRecommender = travel_model.DestinationRecommender
    extract_query_info = travel_model.extract_query_info
    load_data = travel_model.load_data
    preprocess_data = travel_model.preprocess_data
except Exception as e:
    print(f"Error importing from try.py: {e}")
    print("Make sure try.py contains the required components.")
    raise

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Register the auth blueprint
app.register_blueprint(auth_bp)

# Initialize model and required components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Load model from saved state
def load_model():
    # Check if model output directory exists
    output_dir = './model_output/'
    model_path = os.path.join(output_dir, 'roberta_destination_model.pt')
    
    if not os.path.exists(model_path):
        print("Model file not found. Please run revised.py first to train the model.")
        return None, None, None
    
    # Load the destinations from database for label encoder
    # This replaces the CSV loading
    all_destinations = db.get_destinations(limit=1000)  # Get a large number to ensure we have all
    
    # Convert to DataFrame for compatibility with existing code
    df = pd.DataFrame(all_destinations)
    
    # Create a combined text field for the model (similar to preprocess_data)
    df['combined_text'] = df.apply(
        lambda row: f"{row['name']} {row['city']} {row['category']} {row['description']} {row['metadata'] or ''}", 
        axis=1
    )
    
    # Initialize model
    # We'll assume number of labels equals number of destinations for now
    num_labels = len(df)
    
    try:
        # Initialize model with the current number of labels
        model = DestinationRecommender(num_labels=num_labels).to(device)
        
        # Try to load saved weights
        saved_state = torch.load(model_path, map_location=device)
        model.load_state_dict(saved_state)
        print(f"Model loaded successfully with {num_labels} destinations.")
    except RuntimeError as e:
        # If there's a size mismatch, create a new model without loading weights
        print(f"Warning: Model state size mismatch. Creating a new model instance.")
        print(f"Error details: {e}")
        model = DestinationRecommender(num_labels=num_labels).to(device)
        print(f"Created a new model with {num_labels} destinations.")
    
    model.eval()
    
    return model, df, None  # We don't need label_encoder anymore

# Load embeddings from file or create them
def get_embeddings(model, df):
    # Use the updated destination embeddings file
    embeddings_path = './model_output/destination_embeddings.npy'
    
    try:
        if os.path.exists(embeddings_path):
            loaded_embeddings = np.load(embeddings_path)
            # Check if the shape matches our current data
            if len(loaded_embeddings) == len(df):
                print(f"Loaded embeddings for {len(loaded_embeddings)} destinations.")
                return loaded_embeddings
            else:
                print(f"Embeddings size mismatch. Expected {len(df)}, got {len(loaded_embeddings)}. Creating new embeddings...")
        else:
            print("Embeddings file not found. Creating new embeddings...")
        
        # If embeddings don't exist yet or don't match, we need to create them
        print(f"Generating embeddings for {len(df)} destinations...")
        embeddings = []
        
        with torch.no_grad():
            for idx, (_, row) in enumerate(df.iterrows()):
                if idx % 50 == 0:
                    print(f"Processing destination {idx}/{len(df)}...")
                text = row['combined_text']
                encoding = tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=512,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                ).to(device)
                
                outputs = model.roberta(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask']
                )
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding)
        
        embeddings = np.vstack(embeddings)
        
        # Save the embeddings
        try:
            os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
            np.save(embeddings_path, embeddings)
            print(f"Saved embeddings for {len(embeddings)} destinations.")
        except Exception as e:
            print(f"Warning: Failed to save embeddings: {e}")
        
        return embeddings
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        # Return a default empty array as fallback - this won't work well but prevents crashes
        return np.array([])

# Initialize everything at startup
try:
    print("Loading recommendation model...")
    model, df, label_encoder = load_model()
    
    if model is not None and df is not None:
        print("Loading embeddings...")
        embeddings = get_embeddings(model, df)
        if embeddings is None or len(embeddings) == 0:
            print("Warning: No embeddings available. Some features may not work correctly.")
    else:
        print("Warning: Model or data not available. Recommendation features will be limited.")
        embeddings = None
except Exception as e:
    print(f"Error initializing model: {e}")
    model = None
    df = None
    embeddings = None
    print("Starting with limited functionality. Recommendation features will not be available.")

# Expand the KnowledgeCache class with enhanced conversation context capabilities
class KnowledgeCache:
    def __init__(self, max_size=100, expiry_time=3600):  # 1 hour expiry
        self.cache = {}
        self.query_history = []
        self.max_size = max_size
        self.expiry_time = expiry_time
        self.category_preferences = defaultdict(Counter)
        self.city_preferences = defaultdict(Counter)
        self.session_contexts = {}
        self.conversation_history = defaultdict(list)
        self.sentiment_history = defaultdict(list)
        self.topic_transitions = defaultdict(list)
    
    def add(self, query, result, session_id=None):
        # Normalize query for better matching
        normalized_query = self._normalize_query(query)
        current_time = time.time()
        
        # If cache is full, remove oldest entry
        if len(self.cache) >= self.max_size:
            oldest_query = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_query]
        
        # Add to cache
        self.cache[normalized_query] = {
            'result': result,
            'timestamp': current_time,
            'hits': 1
        }
        
        # Track category and city preferences if available
        if isinstance(result, dict) and 'recommendations' in result:
            for rec in result.get('recommendations', []):
                if session_id:
                    if 'category' in rec:
                        self.category_preferences[session_id][rec['category']] += 1
                    if 'city' in rec:
                        self.city_preferences[session_id][rec['city']] += 1
        
        # Add to query history
        self.query_history.append({
            'query': query,
            'normalized': normalized_query,
            'timestamp': current_time,
            'session_id': session_id
        })
    
    def get(self, query):
        normalized_query = self._normalize_query(query)
        if normalized_query in self.cache:
            entry = self.cache[normalized_query]
            current_time = time.time()
            
            # Check if entry is expired
            if current_time - entry['timestamp'] > self.expiry_time:
                del self.cache[normalized_query]
                return None
            
            # Update hits and timestamp
            entry['hits'] += 1
            entry['timestamp'] = current_time
            return entry['result']
        
        return None
    
    def get_similar_query(self, query, threshold=0.8):
        """Find similar queries in cache above similarity threshold"""
        normalized_query = self._normalize_query(query)
        
        # For now use basic word overlap as similarity measure
        query_words = set(normalized_query.split())
        best_match = None
        best_score = threshold
        
        for cached_query in self.cache:
            cached_words = set(cached_query.split())
            
            # Calculate Jaccard similarity
            if not query_words or not cached_words:
                continue
                
            intersection = len(query_words.intersection(cached_words))
            union = len(query_words.union(cached_words))
            
            score = intersection / union if union > 0 else 0
            
            if score > best_score:
                best_score = score
                best_match = cached_query
        
        if best_match:
            return self.cache[best_match]['result']
        
        return None
    
    def update_session_context(self, session_id, context_data):
        """Update context for a specific session"""
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = {}
        
        self.session_contexts[session_id].update(context_data)
    
    def get_session_context(self, session_id):
        """Get context for a specific session"""
        return self.session_contexts.get(session_id, {})
    
    def get_preferred_categories(self, session_id, top_n=3):
        """Get preferred categories for a session"""
        if session_id not in self.category_preferences:
            return []
        
        return [cat for cat, _ in self.category_preferences[session_id].most_common(top_n)]
    
    def get_preferred_cities(self, session_id, top_n=3):
        """Get preferred cities for a session"""
        if session_id not in self.city_preferences:
            return []
        
        return [city for city, _ in self.city_preferences[session_id].most_common(top_n)]
    
    def _normalize_query(self, query):
        """Normalize query for better matching"""
        # Convert to lowercase
        query = query.lower().strip()
        
        # Remove extra whitespaces
        query = " ".join(query.split())
        
        # Remove common filler words for better matching
        filler_words = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "is", "are", "was", "were"}
        query_words = query.split()
        query_words = [word for word in query_words if word not in filler_words]
        
        return " ".join(query_words)
    
    def add_to_conversation(self, session_id, user_message, system_response, sentiment=None):
        """Add a conversation exchange to the history"""
        if not session_id:
            return
            
        # Get timestamp
        timestamp = time.time()
        
        # Extract topics from the message
        topics = self._extract_topics(user_message)
        
        # Analyze sentiment if not provided
        if sentiment is None:
            sentiment = self._analyze_sentiment(user_message)
        
        # Create conversation entry
        conversation_entry = {
            'timestamp': timestamp,
            'user_message': user_message,
            'system_response': system_response,
            'topics': topics,
            'sentiment': sentiment
        }
        
        # Add to conversation history
        self.conversation_history[session_id].append(conversation_entry)
        
        # Track sentiment over time
        self.sentiment_history[session_id].append({
            'timestamp': timestamp,
            'sentiment': sentiment
        })
        
        # Track topic transitions
        if len(self.conversation_history[session_id]) > 1:
            prev_entry = self.conversation_history[session_id][-2]
            prev_topics = prev_entry.get('topics', [])
            
            # Record the transition between topics
            for prev_topic in prev_topics:
                for curr_topic in topics:
                    if prev_topic != curr_topic:
                        self.topic_transitions[session_id].append({
                            'from': prev_topic,
                            'to': curr_topic,
                            'timestamp': timestamp
                        })
    
    def get_conversation_summary(self, session_id, max_entries=5):
        """Get a summary of recent conversation"""
        if not session_id or session_id not in self.conversation_history:
            return []
            
        # Get most recent entries
        recent_entries = self.conversation_history[session_id][-max_entries:]
        
        # Create summary
        summary = []
        for entry in recent_entries:
            summary.append({
                'user_message': entry['user_message'],
                'topics': entry['topics'],
                'sentiment': entry['sentiment']
            })
            
        return summary
    
    def get_frequently_discussed_topics(self, session_id, top_n=3):
        """Get the most frequently discussed topics in a session"""
        if not session_id or session_id not in self.conversation_history:
            return []
            
        # Collect all topics
        all_topics = []
        for entry in self.conversation_history[session_id]:
            all_topics.extend(entry.get('topics', []))
            
        # Count frequencies
        topic_counter = Counter(all_topics)
        
        # Return top N
        return [topic for topic, _ in topic_counter.most_common(top_n)]
    
    def get_conversation_context(self, session_id, query=None):
        """Get enhanced conversation context for better response generation"""
        if not session_id:
            return {}
            
        context = {
            'session_id': session_id,
            'preferences': {}
        }
        
        # Get basic session context
        session_context = self.get_session_context(session_id)
        context.update(session_context)
        
        # Add category preferences
        if session_id in self.category_preferences:
            preferred_categories = [cat for cat, _ in self.category_preferences[session_id].most_common(3)]
            context['preferences']['categories'] = preferred_categories
            
        # Add city preferences
        if session_id in self.city_preferences:
            preferred_cities = [city for city, _ in self.city_preferences[session_id].most_common(3)]
            context['preferences']['cities'] = preferred_cities
        
        # Add conversation history summary
        if session_id in self.conversation_history:
            context['conversation'] = {
                'exchanges': len(self.conversation_history[session_id]),
                'recent_topics': self.get_frequently_discussed_topics(session_id)
            }
            
            # Add recent sentiment trend
            if session_id in self.sentiment_history and len(self.sentiment_history[session_id]) > 0:
                recent_sentiments = [entry['sentiment'] for entry in self.sentiment_history[session_id][-3:]]
                avg_sentiment = sum(recent_sentiments) / len(recent_sentiments) if recent_sentiments else 0
                context['conversation']['sentiment_trend'] = avg_sentiment
        
        # If query is provided, analyze it for contextual relevance
        if query:
            context['current_query'] = {
                'topics': self._extract_topics(query),
                'sentiment': self._analyze_sentiment(query)
            }
            
            # Check for topic continuity
            if 'conversation' in context and 'recent_topics' in context['conversation']:
                query_topics = self._extract_topics(query)
                recent_topics = context['conversation']['recent_topics']
                context['current_query']['topic_continuity'] = any(topic in recent_topics for topic in query_topics)
        
        return context
    
    def _extract_topics(self, text):
        """Extract topics from a text"""
        # Simple topic extraction based on keywords
        travel_topics = {
            'destination': ['city', 'destination', 'place', 'location', 'country', 'visit'],
            'accommodation': ['hotel', 'resort', 'stay', 'accommodation', 'room', 'booking'],
            'transportation': ['flight', 'train', 'bus', 'car', 'transportation', 'travel'],
            'activities': ['activity', 'tour', 'sightseeing', 'adventure', 'experience'],
            'food': ['food', 'restaurant', 'cuisine', 'eat', 'dining'],
            'budget': ['budget', 'cost', 'price', 'expensive', 'cheap', 'affordable'],
            'planning': ['plan', 'itinerary', 'schedule', 'booking', 'reservation'],
            'weather': ['weather', 'season', 'temperature', 'climate'],
            'cultural': ['culture', 'history', 'museum', 'traditional', 'local'],
            'nature': ['nature', 'beach', 'mountain', 'lake', 'park', 'hiking']
        }
        
        found_topics = []
        text_lower = text.lower()
        
        for topic, keywords in travel_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)
                
        return found_topics
    
    def _analyze_sentiment(self, text):
        """Analyze sentiment of a text (basic version)"""
        # Simple sentiment analysis based on keywords
        positive_keywords = ['good', 'great', 'excellent', 'amazing', 'love', 'enjoy', 'nice', 'happy', 'beautiful', 
                           'wonderful', 'perfect', 'best', 'like', 'recommend', 'fantastic', 'awesome']
        negative_keywords = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'poor', 'disappointing', 'worst',
                           'horrible', 'avoid', 'expensive', 'dirty', 'crowded', 'dangerous']
        
        text_lower = text.lower()
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        # Simple scoring between -1 and 1
        total = positive_count + negative_count
        if total == 0:
            return 0  # Neutral
            
        return (positive_count - negative_count) / total

# Initialize knowledge cache
knowledge_cache = KnowledgeCache()

# Enhanced query understanding with context
def understand_query(query_text, session_id=None):
    """
    Enhanced query understanding with user context and semantic analysis
    """
    query_lower = query_text.lower().strip()
    
    # Check for common travel intents
    intents = {
        'find_destination': ['show me', 'recommend', 'suggest', 'find', 'looking for', 'want to visit', 'places to visit', 'want to see'],
        'explore_activity': ['activities', 'things to do', 'what can i do', 'what to do'],
        'plan_trip': ['plan', 'itinerary', 'schedule', 'trip to', 'travel to', 'want to go'],
        'get_info': ['tell me about', 'information about', 'what is', 'details about', 'facts about']
    }
    
    detected_intent = None
    intent_score = 0
    
    for intent, phrases in intents.items():
        for phrase in phrases:
            if phrase in query_lower:
                # Simple scoring - could be improved with ML models
                current_score = len(phrase.split())
                if current_score > intent_score:
                    detected_intent = intent
                    intent_score = current_score
    
    # If no specific intent detected, default to find_destination
    if not detected_intent:
        detected_intent = 'find_destination'
    
    # Look for budget indicators
    budget_indicators = {
        'low': ['cheap', 'budget', 'affordable', 'inexpensive', 'low cost', 'economical'],
        'medium': ['moderate', 'reasonable', 'mid-range', 'standard'],
        'high': ['luxury', 'expensive', 'high-end', 'premium', 'exclusive']
    }
    
    budget_preference = None
    for budget, terms in budget_indicators.items():
        if any(term in query_lower for term in terms):
            budget_preference = budget
            break
    
    # Extract trip type
    trip_type_indicators = {
        'adventure': ['adventure', 'hiking', 'trekking', 'outdoor', 'extreme', 'activities'],
        'relaxation': ['relax', 'peaceful', 'quiet', 'calm', 'unwind', 'spa', 'retreat'],
        'cultural': ['culture', 'history', 'historical', 'museum', 'tradition', 'heritage'],
        'family': ['family', 'kids', 'children', 'family-friendly'],
        'romantic': ['romantic', 'couple', 'honeymoon', 'anniversary', 'date']
    }
    
    trip_type = None
    for type_name, terms in trip_type_indicators.items():
        if any(term in query_lower for term in terms):
            trip_type = type_name
            break
    
    # Get user context if available
    user_context = knowledge_cache.get_session_context(session_id) if session_id else {}
    
    # Get user preferences if available
    preferred_categories = knowledge_cache.get_preferred_categories(session_id) if session_id else []
    preferred_cities = knowledge_cache.get_preferred_cities(session_id) if session_id else []
    
    # Extract city and category using the existing function
    available_cities = df['city'].unique().tolist()
    available_categories = df['category'].unique().tolist()
    
    # Update to use the enhanced extract_query_info from try.py with all parameters
    city, category, budget, cleaned_query, sentiment_info, budget_amount = extract_query_info(
        query_text, 
        available_cities, 
        available_categories,
        None  # available_budgets parameter
    )
    
    # Enhance with context if direct extraction failed
    if not city and preferred_cities and 'city' not in query_lower:
        # Use most preferred city as implicit context
        city = preferred_cities[0]
    
    if not category and preferred_categories and 'category' not in query_lower:
        # Use most preferred category as implicit context
        category = preferred_categories[0]
    
    # Use the original query if cleaned_query is empty
    if not cleaned_query.strip():
        cleaned_query = query_text
    
    query_understanding = {
        'original_query': query_text,
        'cleaned_query': cleaned_query,
        'detected_intent': detected_intent,
        'city': city,
        'category': category,
        'budget_preference': budget or budget_preference,
        'budget_amount': budget_amount,
        'trip_type': trip_type,
        'sentiment_info': sentiment_info,
        'user_context': user_context
    }
    
    return query_understanding

# Enhanced recommendation with caching and better understanding
@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        # Check for required components
        if model is None:
            print("Error: Model not loaded")
            return jsonify({
                'error': 'Model not loaded. Please check server logs for details.',
                'is_conversation': True,
                'message': "I'm sorry, but my recommendation engine is not available right now. Please try again later."
            }), 500
        
        if embeddings is None or len(embeddings) == 0:
            print("Error: Embeddings not available")
            return jsonify({
                'error': 'Embeddings not available. Please check server logs for details.',
                'is_conversation': True,
                'message': "I'm sorry, but my recommendation engine is not properly configured. Please try again later."
            }), 500
        
        if df is None or len(df) == 0:
            print("Error: Destination data not available")
            return jsonify({
                'error': 'Destination data not available.',
                'is_conversation': True,
                'message': "I don't have any destination data available right now. Please try again later."
            }), 500
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided in request'}), 400
            
        query_text = data.get('query', '')
        session_id = data.get('session_id', None)
        
        if not query_text:
            return jsonify({'error': 'Query is required'}), 400

        print(f"Processing query: '{query_text}' for session: {session_id}")
        
        # Check cache first for exact or similar queries
        cached_result = knowledge_cache.get(query_text)
        if cached_result:
            print(f"Found exact cache match for query: '{query_text}'")
            # If we have exact match, return the cached result
            return jsonify(cached_result)
        
        # Try similar query
        similar_result = knowledge_cache.get_similar_query(query_text)
        if similar_result:
            print(f"Found similar cache match for query: '{query_text}'")
            # If we have similar match, we'll use it but might want to refresh it
            return jsonify(similar_result)
        
        # Handle greetings and basic conversation
        query_lower = query_text.lower().strip()
        
        # Check for greetings
        greetings = ['hi', 'hello', 'hey', 'hi!', 'hello!', 'hey!', 'greetings', 'good morning', 'good afternoon', 'good evening']
        help_phrases = ['help', 'how does this work', 'what can you do', 'what do you do']
        thanks_phrases = ['thanks', 'thank you', 'thx', 'appreciate it']
        
        # Get conversation context safely
        try:
            conversation_context = knowledge_cache.get_conversation_context(session_id, query_text)
        except Exception as e:
            print(f"Error getting conversation context: {e}")
            conversation_context = {}
        
        # Check if this is a greeting or basic conversation
        if any(greeting == query_lower for greeting in greetings):
            # Personalize response based on conversation history
            message = "Hello! I'm your travel assistant. I can recommend destinations based on your preferences."
            
            # If returning user, make it more personalized
            if session_id and 'conversation' in conversation_context and conversation_context['conversation'].get('exchanges', 0) > 3:
                # Reference previous interests
                if 'preferences' in conversation_context and 'categories' in conversation_context['preferences']:
                    preferred_categories = conversation_context['preferences']['categories']
                    if preferred_categories:
                        message += f" I see you've been interested in {preferred_categories[0]}. Would you like more recommendations in that category?"
                elif 'preferences' in conversation_context and 'cities' in conversation_context['preferences']:
                    preferred_cities = conversation_context['preferences']['cities']
                    if preferred_cities:
                        message += f" You seem interested in {preferred_cities[0]}. Would you like to explore more about it?"
                else:
                    message += " Welcome back! How can I help with your travel plans today?"
            else:
                # First-time or infrequent user
                message += " Try asking about resorts, historical sites, nature spots, or specific cities you're interested in!"
            
            response = {
                'is_conversation': True,
                'message': message
            }
            
            # Add to cache and conversation history
            try:
                knowledge_cache.add(query_text, response, session_id)
                knowledge_cache.add_to_conversation(session_id, query_text, message)
            except Exception as e:
                print(f"Error updating knowledge cache: {e}")
            
            return jsonify(response)
        elif any(phrase in query_lower for phrase in help_phrases):
            # Simple help message as fallback in case of errors
            message = "I can help you find interesting places to visit based on your preferences. You can ask me things like 'Show me beach resorts' or 'I want to visit historical sites'."
            
            # Try to provide a more personalized help message if context is available
            try:
                # Consider conversation history for help message
                if 'preferences' in conversation_context:
                    if 'categories' in conversation_context['preferences'] and conversation_context['preferences']['categories']:
                        categories = conversation_context['preferences']['categories']
                        message += f" I notice you like {', '.join(categories[:2])}. You can ask about these specifically."
                    elif 'cities' in conversation_context['preferences'] and conversation_context['preferences']['cities']:
                        cities = conversation_context['preferences']['cities']
                        message += f" You seem interested in {', '.join(cities[:2])}. You can ask about destinations there."
                
                message += " You can ask me things like 'Show me beach resorts in Palawan' or 'I want to visit historical sites' or simply describe what you're looking for like 'I want a relaxing nature retreat with good views'."
            except Exception as e:
                print(f"Error generating personalized help message: {e}")
            
            response = {
                'is_conversation': True,
                'message': message
            }
            
            # Add to cache and conversation history
            try:
                knowledge_cache.add(query_text, response, session_id)
                knowledge_cache.add_to_conversation(session_id, query_text, message)
            except Exception as e:
                print(f"Error updating knowledge cache: {e}")
            
            return jsonify(response)
        elif any(phrase in query_lower for phrase in thanks_phrases):
            # Simple thanks response as fallback
            message = "You're welcome! Is there anything else you'd like to know about destinations or travel planning?"
            
            # Try to provide a more personalized thanks message if sentiment data is available
            try:
                # Personalize thanks response based on conversation history
                sentiment = 0.5  # Default positive sentiment
                
                if 'current_query' in conversation_context and 'sentiment' in conversation_context['current_query']:
                    sentiment = conversation_context['current_query']['sentiment']
                
                # More enthusiastic for positive sentiment
                if sentiment > 0.5:
                    message = "You're very welcome! I'm really glad I could help. Is there anything else you'd like to know about destinations or travel planning?"
                elif sentiment > 0:
                    message = "You're welcome! Is there anything else you'd like to know about destinations or travel planning?"
                else:
                    message = "You're welcome. I hope I was able to help. Let me know if you need anything else for your travel plans."
            except Exception as e:
                print(f"Error generating personalized thanks message: {e}")
            
            response = {
                'is_conversation': True,
                'message': message
            }
            
            # Add to cache and conversation history
            try:
                knowledge_cache.add(query_text, response, session_id)
                knowledge_cache.add_to_conversation(session_id, query_text, message)
            except Exception as e:
                print(f"Error updating knowledge cache: {e}")
            
            return jsonify(response)
        
        # Process query text as a fallback if recommendation or advanced features fail
        simple_response = None
        
        try:
            # Try the basic recommendation approach first
            print(f"Attempting to get simple recommendations for: '{query_text}'")
            simple_response = recommend_internal(query_text)
            
            if simple_response and 'recommendations' in simple_response and simple_response['recommendations']:
                print(f"Found {len(simple_response['recommendations'])} simple recommendations")
                return jsonify(simple_response)
        except Exception as e:
            print(f"Error in simple recommendation: {e}")
            # Continue with advanced recommendations
        
        # Full ML-based recommendation process starts here
        print("Proceeding with full ML-based recommendation process")
        
        # Get enhanced understanding of the query
        try:
            query_understanding = understand_query(query_text, session_id)
        except Exception as e:
            print(f"Error in query understanding: {e}")
            # Create a simple understanding as fallback
            query_understanding = {
                'original_query': query_text,
                'cleaned_query': query_text,
                'detected_intent': 'find_destination',
                'city': None,
                'category': None,
                'budget_preference': None,
                'trip_type': None,
                'user_context': {}
            }
        
        # Process the query with robust error handling
        try:
            # Tokenize and encode the query
            query_encoding = tokenizer(
                query_understanding['cleaned_query'],
                add_special_tokens=True,
                max_length=512,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            ).to(device)
            
            # Get query embedding
            with torch.no_grad():
                outputs = model.roberta(
                    input_ids=query_encoding['input_ids'],
                    attention_mask=query_encoding['attention_mask']
                )
                query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            similarity_series = pd.Series(similarities, index=df.index)
            
            # Apply filters
            filtered_df = df.copy()
            
            # Safe city filtering
            if query_understanding['city'] and isinstance(query_understanding['city'], str): 
                try:
                    city_mask = filtered_df['city'].str.lower() == query_understanding['city'].lower()
                    if any(city_mask):
                        filtered_df = filtered_df[city_mask]
                except Exception as e:
                    print(f"Error filtering by city: {e}")
            
            # Safe category filtering
            if query_understanding['category'] and isinstance(query_understanding['category'], str):
                try:
                    category_mask = filtered_df['category'].str.lower() == query_understanding['category'].lower()
                    if any(category_mask):
                        filtered_df = filtered_df[category_mask]
                except Exception as e:
                    print(f"Error filtering by category: {e}")
            
            # Safe budget filtering
            if query_understanding['budget_preference'] and isinstance(query_understanding['budget_preference'], str):
                try:
                    budget_mask = filtered_df['budget'].str.lower() == query_understanding['budget_preference'].lower()
                    if any(budget_mask):
                        filtered_df = filtered_df[budget_mask]
                except Exception as e:
                    print(f"Error filtering by budget: {e}")
            elif query_understanding['budget_amount'] and isinstance(query_understanding['budget_amount'], (int, float)):
                try:
                    # First convert budget to numeric
                    filtered_df['budget_numeric'] = pd.to_numeric(filtered_df['budget'], errors='coerce')
                    budget_mask = filtered_df['budget_numeric'] <= query_understanding['budget_amount']
                    if any(budget_mask):
                        filtered_df = filtered_df[budget_mask]
                except Exception as e:
                    print(f"Error filtering by budget amount: {e}")
            
            # Get filtered similarities
            filtered_indices = filtered_df.index
            filtered_similarities = similarity_series[filtered_indices]
            
            # Get top recommendations
            top_n = 3
            if len(filtered_similarities) == 0:
                # Fall back to simple response or return "no results" message
                if simple_response and 'recommendations' in simple_response and simple_response['recommendations']:
                    return jsonify(simple_response)
                
                response = {
                    'is_conversation': True,
                    'message': "I couldn't find any destinations matching your criteria. Can you try a different search or be more general?"
                }
                return jsonify(response)
            
            # Get top indices safely
            top_indices = filtered_similarities.nlargest(min(top_n, len(filtered_similarities))).index
            top_recommendations = []
            
            for idx in top_indices:
                try:
                    row = df.loc[idx]
                    # Convert similarity score to star rating (1-5 stars)
                    rating = min(5, max(1, int(float(similarity_series[idx]) * 5)))
                    
                    # Safely extract values with defaults for missing data
                    name = row.get('name', 'Unknown Destination')
                    city = row.get('city', 'Unknown Location')
                    category = row.get('category', 'Destination')
                    description = row.get('description', 'No description available.')
                    
                    top_recommendations.append({
                        'name': name,
                        'city': city,
                        'category': category,
                        'description': description,
                        'similarity': float(similarity_series[idx]),  # Keep original for sorting
                        'rating': rating  # Add star rating (1-5)
                    })
                except Exception as e:
                    print(f"Error processing recommendation at index {idx}: {e}")
                    # Skip this recommendation and continue
            
            # If we couldn't process any recommendations, use the simple response or return error
            if not top_recommendations:
                if simple_response and 'recommendations' in simple_response and simple_response['recommendations']:
                    return jsonify(simple_response)
                
                response = {
                    'is_conversation': True,
                    'message': "I had trouble processing recommendations for your query. Can you try something simpler?"
                }
                return jsonify(response)
            
            # Prepare response
            response = {
                'recommendations': top_recommendations,
                'detected_city': query_understanding.get('city'),
                'detected_category': query_understanding.get('category'),
                'query_understanding': {k: v for k, v in query_understanding.items() if k != 'user_context'}  # Don't return user context
            }
            
            # Add follow-up suggestions if available
            try:
                if session_id and conversation_context:
                    # Generate follow-up questions
                    follow_up_suggestions = generate_follow_up_suggestions(top_recommendations, conversation_context)
                    if follow_up_suggestions:
                        response['follow_up_suggestions'] = follow_up_suggestions
            except Exception as e:
                print(f"Error generating follow-up suggestions: {e}")
                # Continue without follow-up suggestions
            
            # Save to cache and update context
            try:
                # Save to cache for future use
                knowledge_cache.add(query_text, response, session_id)
                
                # If there's a session, update the context
                if session_id:
                    context_update = {
                        'last_query': query_text,
                        'last_response': response,
                        'last_response_timestamp': time.time()
                    }
                    
                    if query_understanding.get('city'):
                        context_update['last_city'] = query_understanding['city']
                    
                    if query_understanding.get('category'):
                        context_update['last_category'] = query_understanding['category']
                        
                    if query_understanding.get('trip_type'):
                        context_update['last_trip_type'] = query_understanding['trip_type']
                    
                    knowledge_cache.update_session_context(session_id, context_update)
                    
                    # Add to conversation history
                    sentiment = 0
                    if 'current_query' in conversation_context and 'sentiment' in conversation_context['current_query']:
                        sentiment = conversation_context['current_query']['sentiment']
                    
                    knowledge_cache.add_to_conversation(
                        session_id, 
                        query_text, 
                        "Provided recommendations for " + (query_understanding.get('city') or "destinations"),
                        sentiment
                    )
            except Exception as e:
                print(f"Error updating cache and context: {e}")
                # Continue without updating cache
            
            return jsonify(response)
        
        except Exception as e:
            # If advanced recommendation fails but we have simple recommendations, use those
            if simple_response and 'recommendations' in simple_response and simple_response['recommendations']:
                print(f"Using simple recommendations as fallback after error: {e}")
                return jsonify(simple_response)
            
            # Log the full error for debugging
            print(f"Error in recommendation process: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return a friendly message to the user
            return jsonify({
                'error': 'An unexpected error occurred while processing your request.',
                'is_conversation': True,
                'message': "I'm sorry, I'm having trouble processing your request right now. Please try again with a simpler query or try again later."
            }), 500
            
    except Exception as e:
        # Log the error but provide a friendly message to the user
        print(f"Fatal error in recommendation: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': 'An unexpected error occurred while processing your request.',
            'is_conversation': True,
            'message': "I'm sorry, I'm having trouble processing your request right now. Please try again with a simpler query or try again later."
        }), 500

# Function to generate follow-up suggestions
def generate_follow_up_suggestions(recommendations, conversation_context):
    """Generate contextual follow-up questions based on recommendations and conversation history"""
    suggestions = []
    
    if not recommendations:
        return suggestions
    
    # Extract key information from the recommendations
    cities = list(set(rec['city'] for rec in recommendations))
    categories = list(set(rec['category'] for rec in recommendations))
    
    # Generate city-based questions
    if cities:
        primary_city = cities[0]
        suggestions.append(f"What are some popular activities in {primary_city}?")
        
        # If there's conversation history about transportation
        if 'conversation' in conversation_context and 'recent_topics' in conversation_context['conversation']:
            if 'transportation' in conversation_context['conversation']['recent_topics']:
                suggestions.append(f"How can I get around in {primary_city}?")
                
        # If there's conversation history about food
        if 'conversation' in conversation_context and 'recent_topics' in conversation_context['conversation']:
            if 'food' in conversation_context['conversation']['recent_topics']:
                suggestions.append(f"What local food should I try in {primary_city}?")
    
    # Generate category-based questions
    if categories:
        primary_category = categories[0]
        # If accommodation was mentioned
        if 'conversation' in conversation_context and 'recent_topics' in conversation_context['conversation']:
            if 'accommodation' in conversation_context['conversation']['recent_topics']:
                suggestions.append(f"What are the best hotels near these {primary_category}?")
        
        # If budget was mentioned
        if 'conversation' in conversation_context and 'recent_topics' in conversation_context['conversation']:
            if 'budget' in conversation_context['conversation']['recent_topics']:
                suggestions.append(f"What's the typical cost to visit these {primary_category}?")
    
    # If planning was a topic, suggest itinerary creation
    if 'conversation' in conversation_context and 'recent_topics' in conversation_context['conversation']:
        if 'planning' in conversation_context['conversation']['recent_topics'] and cities:
            suggestions.append(f"Can you create an itinerary for {cities[0]}?")
    
    # Take only a few suggestions
    return suggestions[:3]

@app.route('/api/create_trip', methods=['POST'])
def create_trip():
    """
    Create a trip itinerary based on the given destination and preferences
    """
    try:
        # Get data from request
        data = request.get_json()
        destination = data.get('destination')
        travel_dates = data.get('travel_dates', '')
        travelers = data.get('travelers', 1)
        budget = data.get('budget', 'mid-range')
        interests = data.get('interests', [])
        
        # Validate destination
        if not destination:
            return jsonify({"error": "Destination is required"}), 400
        
        # Check if the destination exists in our database
        # First check for exact match
        destinations = db.get_destinations(limit=10, city=destination)
        
        # If no exact match, try a search
        if not destinations:
            destinations = db.search_destinations(destination, limit=10)
            
        # If still no match, suggest alternatives
        if not destinations:
            # Get all cities as suggestions
            cities = db.get_distinct_cities()
            
            # Return error with suggestions
            return jsonify({
                "error": "Destination not found",
                "suggestions": cities[:10]  # Limit to top 10
            }), 404
        
        # Use the first destination city as our main destination
        main_destination = destinations[0]['city']
        
        # Generate itinerary
        itinerary = generate_itinerary(destinations, travel_dates)
        
        # Get activities based on interests
        if interests:
            # Convert interests list to search string
            interest_query = " ".join(interests)
            activity_recommendations = db.search_destinations(interest_query, limit=20)
            
            # Only include activities in the same city
            activities = [act for act in activity_recommendations 
                         if act['city'].lower() == main_destination.lower()]
            
            # Add these to each day's itinerary if they're not already included
            existing_place_ids = []
            for day in itinerary:
                existing_place_ids.extend([place['id'] for place in day['places']])
            
            # Distribute remaining activities across days
            for i, activity in enumerate(activities):
                if activity['id'] not in existing_place_ids:
                    day_index = i % len(itinerary)
                    # Add only if we have space (maximum 4 activities per day)
                    if len(itinerary[day_index]['places']) < 4:
                        itinerary[day_index]['places'].append({
                            'id': activity['id'],
                            'name': activity['name'],
                            'category': activity['category'],
                            'description': activity['description'],
                            'rating': activity['ratings'] if activity['ratings'] is not None else 4.0
                        })
        
        # Create trip response
        trip = {
            "destination": main_destination,
            "travel_dates": travel_dates,
            "travelers": travelers,
            "budget": budget,
            "interests": interests,
            "itinerary": itinerary
        }
        
        # Save to database if user is authenticated
        session_id = request.args.get('session_id') or request.headers.get('X-Session-ID')
        if session_id:
            session = db.get_session(session_id)
            if session and session['user_id']:
                # Save trip to database
                db.create_trip(session['user_id'], trip)
                
        return jsonify({"trip": trip}), 200
        
    except Exception as e:
        print(f"Error creating trip: {e}")
        return jsonify({"error": str(e)}), 500

# Enhanced recommendation internal function
def recommend_internal(query_text, city=None, category=None):
    # Try to understand the query better using the enhanced extract_query_info from try.py
    available_cities = df['city'].unique().tolist() if df is not None else []
    available_categories = df['category'].unique().tolist() if df is not None else []
    available_budgets = None
    
    try:
        city, category, budget, cleaned_query, sentiment_info, budget_amount = extract_query_info(
            query_text,
            available_cities,
            available_categories,
            available_budgets
        )
    except Exception as e:
        print(f"Error in extract_query_info: {e}")
        # Fallback to basic extraction
        cleaned_query = query_text
        budget = None
        sentiment_info = None
        budget_amount = None
    
    # If a specific city or category was detected, use that for filtering
    detected_city = city or city
    detected_category = category or category
    
    # Check if this is a conversational query
    conversation_patterns = [
        (r'\b(hi|hello|hey|greetings)\b', "Hi there! I'm your travel assistant. How can I help you plan your trip today?"),
        (r'\bhow are you\b', "I'm doing great, thanks for asking! I'm ready to help you discover amazing travel destinations. What kind of place are you looking for?"),
        (r'\bthank you\b', "You're welcome! Is there anything else you'd like to know about travel destinations?"),
        (r'\bgoodbye|bye\b', "Goodbye! Feel free to come back when you're planning your next adventure.")
    ]
    
    for pattern, response in conversation_patterns:
        if re.search(pattern, query_text, re.IGNORECASE):
            return {
                "is_conversation": True,
                "message": response,
                "query_understanding": {
                    "detected_city": detected_city,
                    "detected_category": detected_category,
                    "budget": budget,
                    "budget_amount": budget_amount,
                    "sentiment_info": sentiment_info
                }
            }
    
    # If we have a city and/or category filter, use database query directly
    if detected_city or detected_category:
        # Get destinations from database with filters
        destinations = db.get_destinations(limit=10, city=detected_city, category=detected_category)
        
        # Format recommendations
        recommendations = []
        for dest in destinations:
            # Convert rating to a star rating between 1-5
            rating = 3  # Default
            if 'ratings' in dest and dest['ratings'] is not None:
                rating = min(5, max(1, int(float(dest['ratings']) * 5 / 10)))  # Assuming ratings is 0-10
            
            recommendations.append({
                'id': dest['id'],
                'name': dest['name'],
                'city': dest['city'],
                'category': dest['category'],
                'description': dest['description'],
                'rating': rating,
                'budget': dest.get('budget', 'medium')  # Add budget information if available
            })
            
        return {
            "is_conversation": False,
            "detected_city": detected_city,
            "detected_category": detected_category,
            "detected_budget": budget,
            "budget_amount": budget_amount,
            "sentiment_info": sentiment_info,
            "recommendations": recommendations,
            "query_understanding": {
                "detected_city": detected_city,
                "detected_category": detected_category,
                "budget": budget,
                "budget_amount": budget_amount,
                "sentiment_info": sentiment_info
            }
        }
    
    # If no specific filter, use full-text search in the database
    destinations = db.search_destinations(query_text, limit=10)
    
    # Format recommendations
    recommendations = []
    for dest in destinations:
        # Convert rating to a star rating between 1-5
        rating = 3  # Default
        if 'ratings' in dest and dest['ratings'] is not None:
            rating = min(5, max(1, int(float(dest['ratings']) * 5 / 10)))  # Assuming ratings is 0-10
        
        recommendations.append({
            'id': dest['id'],
            'name': dest['name'],
            'city': dest['city'],
            'category': dest['category'],
            'description': dest['description'],
            'rating': rating,
            'budget': dest.get('budget', 'medium')  # Add budget information if available
        })
        
    return {
        "is_conversation": False,
        "detected_city": detected_city,
        "detected_category": detected_category,
        "detected_budget": budget,
        "budget_amount": budget_amount,
        "sentiment_info": sentiment_info,
        "recommendations": recommendations,
        "query_understanding": {
            "detected_city": detected_city,
            "detected_category": detected_category,
            "budget": budget,
            "budget_amount": budget_amount,
            "sentiment_info": sentiment_info
        }
    }

# Helper function to generate a simple itinerary based on recommendations
def generate_itinerary(recommendations, travel_dates):
    if not recommendations:
        return []
    
    # Parse travel dates to determine duration
    try:
        from datetime import datetime
        date_parts = travel_dates.split(' to ')
        if len(date_parts) == 2:
            start_date = datetime.strptime(date_parts[0], '%Y-%m-%d')
            end_date = datetime.strptime(date_parts[1], '%Y-%m-%d')
            duration = (end_date - start_date).days + 1
        else:
            duration = 3  # Default duration if date format is not recognized
    except:
        duration = 3  # Default duration if parsing fails
    
    # Limit duration to number of recommendations available
    duration = min(duration, len(recommendations))
    
    # Create a simple day-by-day itinerary
    itinerary = []
    for day in range(1, duration + 1):
        # Select places for this day (1-2 places per day)
        day_places = recommendations[(day-1)*2:day*2]
        
        # Skip if no places for this day
        if not day_places:
            continue
            
        daily_plan = {
            'day': day,
            'places': day_places,
            'meals': [
                {'type': 'Breakfast', 'suggestion': 'Local breakfast options'},
                {'type': 'Lunch', 'suggestion': 'Restaurant near ' + day_places[0]['name']},
                {'type': 'Dinner', 'suggestion': 'Evening dining experience'}
            ],
            'transportation': 'Local transport or taxi',
            'estimated_cost': calculate_estimate_cost(day_places)
        }
        
        itinerary.append(daily_plan)
    
    return itinerary

# Helper function to calculate estimated cost
def calculate_estimate_cost(places):
    # Very simple cost estimator based on number of places and type
    base_cost = 1000  # Base cost in PHP
    for place in places:
        if 'resort' in place['category'].lower():
            base_cost += 3000
        elif 'historical' in place['category'].lower():
            base_cost += 500
        elif 'museum' in place['category'].lower():
            base_cost += 800
        elif 'natural' in place['category'].lower():
            base_cost += 1200
        else:
            base_cost += 1000
    
    return f"₱{base_cost}"

@app.route('/api/routing', methods=['POST'])
def get_routing():
    """
    API endpoint for calculating routes between multiple points.
    
    Expected JSON input:
    {
        "points": [
            {"lat": 14.5995, "lng": 120.9842, "name": "Starting Point"}, 
            {"lat": 14.6760, "lng": 121.0437, "name": "Destination 1"},
            ...
        ],
        "vehicle": "car" // optional, defaults to "car"
    }
    
    Returns route information and points to draw on the map.
    """
    data = request.get_json()
    
    if not data or 'points' not in data or len(data['points']) < 2:
        return jsonify({'error': 'At least two points are required for routing'}), 400
    
    points = data['points']
    vehicle = data.get('vehicle', 'car')
    
    # GraphHopper API key - replace with your own if needed
    api_key = '07693a69-9493-445a-a2bf-6d035b9329b6'
    
    # Format the points for the GraphHopper API
    point_params = []
    for point in points:
        if 'lat' not in point or 'lng' not in point:
            return jsonify({'error': 'Each point must have lat and lng coordinates'}), 400
        point_params.append(f"point={point['lat']},{point['lng']}")
    
    # Build the URL for the GraphHopper API
    base_url = 'https://graphhopper.com/api/1/route'
    params = '&'.join(point_params)
    url = f"{base_url}?{params}&vehicle={vehicle}&key={api_key}&type=json&points_encoded=false"
    
    try:
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            route_data = response.json()
            
            # Extract relevant information for the frontend
            paths = route_data.get('paths', [])
            if not paths:
                return jsonify({'error': 'No route found'}), 404
            
            # Get the first path (the optimal route)
            path = paths[0]
            
            # Prepare the response
            result = {
                'distance': path.get('distance', 0),  # in meters
                'time': path.get('time', 0),  # in milliseconds
                'points': path.get('points', {}).get('coordinates', []),
                'instructions': path.get('instructions', [])
            }
            
            # Add distance in kilometers and time in minutes
            result['distance_km'] = round(result['distance'] / 1000, 1)
            result['time_min'] = round(result['time'] / (1000 * 60), 0)
            
            return jsonify(result)
        else:
            return jsonify({'error': f'GraphHopper API Error: {response.text}'}), response.status_code
            
    except Exception as e:
        return jsonify({'error': f'Error calculating route: {str(e)}'}), 500

@app.route('/api/geocode', methods=['GET'])
def geocode():
    """
    API endpoint for geocoding place names to coordinates.
    
    Expected query parameters:
    - q: The query text (e.g., "Intramuros, Manila")
    - limit: (optional) Maximum number of results to return (default: 5)
    
    Returns a list of geocoded results with coordinates.
    """
    query = request.args.get('q', '')
    limit = request.args.get('limit', 5, type=int)
    
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    # GraphHopper API key
    api_key = '07693a69-9493-445a-a2bf-6d035b9329b6'
    
    # Build the URL for the GraphHopper Geocoding API
    url = f"https://graphhopper.com/api/1/geocode?q={query}&limit={limit}&key={api_key}"
    
    try:
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            geocode_data = response.json()
            hits = geocode_data.get('hits', [])
            
            # Format the results
            results = []
            for hit in hits:
                results.append({
                    'name': hit.get('name', ''),
                    'country': hit.get('country', ''),
                    'city': hit.get('city', ''),
                    'state': hit.get('state', ''),
                    'street': hit.get('street', ''),
                    'housenumber': hit.get('housenumber', ''),
                    'point': {
                        'lat': hit.get('point', {}).get('lat', 0),
                        'lng': hit.get('point', {}).get('lng', 0)
                    }
                })
            
            return jsonify({'results': results})
        else:
            return jsonify({'error': f'Geocoding API Error: {response.text}'}), response.status_code
            
    except Exception as e:
        return jsonify({'error': f'Error geocoding: {str(e)}'}), 500

@app.route('/api/route_for_itinerary', methods=['POST'])
def route_for_itinerary():
    """
    API endpoint for generating routes for a full itinerary day.
    
    Expected JSON input:
    {
        "day": {
            "places": [
                {"name": "Place 1", "city": "City 1", "category": "Category 1"},
                {"name": "Place 2", "city": "City 2", "category": "Category 2"},
                ...
            ]
        },
        "startingPoint": {"lat": 14.5995, "lng": 120.9842, "name": "Hotel"} // optional
    }
    
    Returns complete route information for the day.
    """
    data = request.get_json()
    
    if not data or 'day' not in data or 'places' not in data['day'] or len(data['day']['places']) < 1:
        return jsonify({'error': 'At least one place is required in the itinerary'}), 400
    
    # Extract places from the itinerary
    places = data['day']['places']
    
    # Starting point (optional)
    starting_point = data.get('startingPoint', None)
    
    # We need to geocode each place to get coordinates
    points = []
    
    # Add starting point if provided
    if starting_point and 'lat' in starting_point and 'lng' in starting_point:
        points.append({
            'lat': starting_point['lat'],
            'lng': starting_point['lng'],
            'name': starting_point.get('name', 'Starting Point')
        })
    
    # Geocode each place in the itinerary
    for place in places:
        place_name = place.get('name', '')
        city = place.get('city', '')
        
        if not place_name:
            continue
            
        # Build geocoding query
        query = f"{place_name}, {city}" if city else place_name
        
        try:
            # Call our own geocoding endpoint to avoid code duplication
            geocode_url = f"{request.host_url.rstrip('/')}/api/geocode?q={query}&limit=1"
            geocode_response = requests.get(geocode_url)
            
            if geocode_response.status_code == 200:
                geocode_data = geocode_response.json()
                
                if geocode_data.get('results') and len(geocode_data['results']) > 0:
                    result = geocode_data['results'][0]
                    points.append({
                        'lat': result['point']['lat'],
                        'lng': result['point']['lng'],
                        'name': place_name
                    })
            
        except Exception as e:
            # Log the error but continue with other places
            print(f"Error geocoding {place_name}: {str(e)}")
    
    # If we couldn't geocode any places, return an error
    if len(points) < 2:
        if starting_point and len(points) == 1:
            return jsonify({'error': 'Could not geocode any places in the itinerary'}), 400
        else:
            return jsonify({'error': 'At least two points are required for routing'}), 400
    
    # Call our routing endpoint
    try:
        routing_url = f"{request.host_url.rstrip('/')}/api/routing"
        routing_response = requests.post(
            routing_url, 
            json={'points': points, 'vehicle': 'car'},
            headers={'Content-Type': 'application/json'}
        )
        
        if routing_response.status_code == 200:
            route_data = routing_response.json()
            
            # Enhance the response with place information
            result = {
                'route': route_data,
                'points': points,
                'total_distance_km': route_data.get('distance_km', 0),
                'total_time_min': route_data.get('time_min', 0)
            }
            
            return jsonify(result)
        else:
            return jsonify({'error': f'Routing error: {routing_response.text}'}), routing_response.status_code
            
    except Exception as e:
        return jsonify({'error': f'Error generating route for itinerary: {str(e)}'}), 500

# Create a session endpoint for maintaining user context
@app.route('/api/create_session', methods=['POST'])
def create_session():
    """Create a new session for tracking user interactions"""
    try:
        # Generate a new session ID
        result = db.create_session()
        
        if 'error' in result:
            return jsonify(result), 500
            
        session_id = result['session_id']
        
        # Initialize session in KnowledgeCache
        knowledge_cache.session_contexts[session_id] = {
            'created_at': time.time(),
            'preferences': {}
        }
        
        return jsonify({"session_id": session_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add new API endpoint for creating tickets
@app.route('/api/create_ticket', methods=['POST'])
def create_ticket():
    """
    Create a new ticket for tracking an itinerary.
    
    Expected JSON input:
    {
        "email": "user@example.com",
        "trip_id": "12345" or null,
        "itinerary": {...} (Optional, full itinerary data)
    }
    
    Returns:
    {
        "ticket_id": "WTO-ABC123",
        "status": "created"
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        email = data.get('email')
        if not email:
            return jsonify({"error": "Email is required"}), 400
            
        # Generate a unique ticket ID in the format "WTO-ABC123"
        # Using a combination of letters and numbers for security
        random_chars = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        ticket_id = f"WTO-{random_chars}"
        
        # Get trip ID or itinerary
        trip_id = data.get('trip_id')
        itinerary = data.get('itinerary')
        
        if not trip_id and not itinerary:
            return jsonify({"error": "Either trip_id or itinerary must be provided"}), 400
        
        # If trip_id is provided, get the trip details
        travel_data = itinerary
        if trip_id and not itinerary:
            travel_data = db.get_trip_by_id(trip_id)
            if not travel_data:
                return jsonify({"error": f"Trip with ID {trip_id} not found"}), 404
            
        # Create ticket in database
        ticket = {
            "ticket_id": ticket_id,
            "email": email,
            "trip_id": trip_id,
            "itinerary": travel_data,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Save ticket to database
        result = db.create_ticket(ticket)
        
        if "error" in result:
            return jsonify(result), 500
        
        # Send email notification with ticket details
        try:
            email_sent = send_ticket_email(email, ticket_id, travel_data)
            email_status = "Email notification sent." if email_sent else "Email notification failed to send."
        except Exception as e:
            print(f"Error sending email: {e}")
            email_status = "Email notification failed to send."
        
        # Return the ticket ID to the user
        return jsonify({
            "ticket_id": ticket_id,
            "status": "created",
            "email_status": email_status,
            "message": "Your ticket has been created. You can use this ticket ID to track your itinerary."
        }), 201
        
    except Exception as e:
        print(f"Error creating ticket: {e}")
        return jsonify({"error": str(e)}), 500

# Add API endpoint for looking up a ticket
@app.route('/api/tickets/<ticket_id>', methods=['GET'])
def get_ticket(ticket_id):
    """
    Get ticket details by ticket ID.
    
    URL Parameters:
    - ticket_id: The ticket ID to look up
    
    Query Parameters (optional):
    - email: Email address for validation
    
    Returns the ticket details if found.
    """
    try:
        # Get email from query parameters (optional for validation)
        email = request.args.get('email')
        
        # Look up ticket in database
        ticket = db.get_ticket(ticket_id)
        
        if not ticket:
            return jsonify({"error": "Ticket not found"}), 404
            
        # If email is provided, validate that it matches
        if email and ticket['email'] != email:
            return jsonify({"error": "Email does not match ticket record"}), 403
            
        # Get trip details if trip_id is available
        trip_data = None
        if ticket.get('trip_id'):
            trip_data = db.get_trip_by_id(ticket['trip_id'])
        
        # Use ticket's itinerary if available
        itinerary_data = ticket.get('itinerary')
            
        # Create a more comprehensive response
        response = {
            "ticket": {
                "ticket_id": ticket['ticket_id'],
                "status": ticket['status'],
                "created_at": ticket['created_at'],
                "updated_at": ticket['updated_at'],
                # Include masked email for privacy
                "email_masked": mask_email(ticket['email']),
                # Include trip details if available
                "trip": trip_data or itinerary_data or {}
            }
        }
        
        # Add more ticket metadata
        if trip_data or itinerary_data:
            travel_info = trip_data or itinerary_data
            response["ticket"]["destination"] = travel_info.get('destination', 'Not specified')
            response["ticket"]["travel_dates"] = travel_info.get('travel_dates', 'Not specified')
            response["ticket"]["travelers"] = travel_info.get('travelers', 1)
            
            # Calculate days until trip
            try:
                if travel_info.get('travel_dates'):
                    dates = travel_info['travel_dates'].split(' to ')
                    if len(dates) > 0:
                        start_date = datetime.strptime(dates[0], '%Y-%m-%d')
                        today = datetime.now()
                        days_until = (start_date - today).days
                        if days_until >= 0:
                            response["ticket"]["days_until_trip"] = days_until
            except Exception as e:
                print(f"Error calculating days until trip: {e}")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error retrieving ticket: {e}")
        return jsonify({"error": str(e)}), 500

# Add API endpoint for getting all tickets for an email
@app.route('/api/tickets', methods=['GET'])
def get_tickets_by_email():
    """
    Get all tickets associated with an email address.
    
    Query Parameters:
    - email: Email address to look up tickets for
    
    Returns a list of tickets.
    """
    try:
        email = request.args.get('email')
        
        if not email:
            return jsonify({"error": "Email parameter is required"}), 400
            
        # Get tickets from database
        tickets = db.get_tickets_by_email(email)
        
        # Enhance ticket data with additional information
        enhanced_tickets = []
        for ticket in tickets:
            # Get basic ticket info
            ticket_info = {
                "ticket_id": ticket['ticket_id'],
                "status": ticket['status'],
                "created_at": ticket['created_at'],
                "updated_at": ticket['updated_at']
            }
            
            # Add destination and date information if available
            travel_info = ticket.get('itinerary') or {}
            if ticket.get('trip_id'):
                try:
                    trip_data = db.get_trip_by_id(ticket['trip_id'])
                    if trip_data:
                        travel_info = trip_data
                except Exception as e:
                    print(f"Error fetching trip data: {e}")
            
            # Add summary information
            ticket_info["destination"] = travel_info.get('destination', 'Not specified')
            ticket_info["travel_dates"] = travel_info.get('travel_dates', 'Not specified')
            
            # Calculate days until trip
            try:
                if travel_info.get('travel_dates'):
                    dates = travel_info['travel_dates'].split(' to ')
                    if len(dates) > 0:
                        start_date = datetime.strptime(dates[0], '%Y-%m-%d')
                        today = datetime.now()
                        days_until = (start_date - today).days
                        if days_until >= 0:
                            ticket_info["days_until_trip"] = days_until
            except Exception as e:
                print(f"Error calculating days until trip: {e}")
            
            enhanced_tickets.append(ticket_info)
        
        # Return list of tickets
        return jsonify({
            "tickets": enhanced_tickets
        }), 200
        
    except Exception as e:
        print(f"Error retrieving tickets: {e}")
        return jsonify({"error": str(e)}), 500

# Helper function to mask email for privacy
def mask_email(email):
    """Mask part of the email address for privacy"""
    if not email or '@' not in email:
        return email
        
    parts = email.split('@')
    username = parts[0]
    domain = parts[1]
    
    # Show first 2 characters and last character of username
    if len(username) > 3:
        masked_username = username[0:2] + '*' * (len(username) - 3) + username[-1]
    else:
        masked_username = username[0] + '*' * (len(username) - 1)
        
    return f"{masked_username}@{domain}"

@app.route('/api/tickets/<ticket_id>/delete', methods=['DELETE'])
def delete_ticket(ticket_id):
    """
    Delete a ticket by ticket ID.
    
    URL Parameters:
    - ticket_id: The ticket ID to delete
    
    Query Parameters:
    - email: Email address for verification
    
    Returns success or error message.
    """
    try:
        # Get email from query parameters for verification
        email = request.args.get('email')
        
        if not email:
            return jsonify({"error": "Email parameter is required for verification"}), 400
            
        # Delete ticket from database with email verification
        result = db.delete_ticket(ticket_id, email)
        
        if "error" in result:
            return jsonify(result), 404 if "not found" in result["error"].lower() else 400
            
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error deleting ticket: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """
    Get all distinct categories available in the database.
    
    Returns a list of category names.
    """
    try:
        # Get all distinct categories from the database
        categories = db.get_distinct_categories()
        
        return jsonify({
            "categories": categories
        }), 200
        
    except Exception as e:
        print(f"Error retrieving categories: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/tickets/<ticket_id>/update', methods=['PUT'])
def update_ticket():
    """
    Update a ticket's status and send an email notification.
    
    URL Parameters:
    - ticket_id: The ticket ID to update
    
    Expected JSON input:
    {
        "status": "confirmed" | "cancelled" | "completed",
        "email": "user@example.com" (optional, for verification)
    }
    
    Returns success or error message.
    """
    try:
        ticket_id = request.view_args.get('ticket_id')
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Get the new status
        new_status = data.get('status')
        if not new_status:
            return jsonify({"error": "Status is required"}), 400
            
        # Validate status value
        valid_statuses = ['active', 'confirmed', 'cancelled', 'completed', 'pending']
        if new_status not in valid_statuses:
            return jsonify({"error": f"Invalid status. Must be one of: {', '.join(valid_statuses)}"}), 400
        
        # Get email for verification (optional)
        email = data.get('email')
        
        # Get the current ticket
        ticket = db.get_ticket(ticket_id)
        if not ticket:
            return jsonify({"error": "Ticket not found"}), 404
            
        # Verify email if provided
        if email and ticket['email'] != email:
            return jsonify({"error": "Email does not match ticket record"}), 403
        
        # Update the ticket status
        result = db.update_ticket_status(ticket_id, new_status)
        
        if "error" in result:
            return jsonify(result), 400
            
        # Get trip data for the email notification
        travel_data = ticket.get('itinerary') or {}
        if ticket.get('trip_id'):
            try:
                trip_data = db.get_trip_by_id(ticket['trip_id'])
                if trip_data:
                    travel_data = trip_data
            except Exception as e:
                print(f"Error fetching trip data: {e}")
        
        # Prepare status-specific message for email
        status_messages = {
            'confirmed': 'Your travel booking has been confirmed!',
            'cancelled': 'Your travel booking has been cancelled.',
            'completed': 'Your trip has been marked as completed. We hope you enjoyed your journey!',
            'pending': 'Your booking is pending confirmation.',
            'active': 'Your booking is now active.'
        }
        
        # Add status message to travel data for email
        travel_data['status_message'] = status_messages.get(new_status, f'Your booking status has been updated to: {new_status}')
        travel_data['status'] = new_status
        
        # Send email notification about status change
        try:
            email_sent = send_ticket_email(ticket['email'], ticket_id, travel_data)
            email_status = "Status update notification sent." if email_sent else "Status update notification failed to send."
        except Exception as e:
            print(f"Error sending status update email: {e}")
            email_status = "Status update notification failed to send."
        
        return jsonify({
            "success": True,
            "ticket_id": ticket_id,
            "status": new_status,
            "email_status": email_status,
            "message": f"Ticket status updated to '{new_status}'."
        }), 200
        
    except Exception as e:
        print(f"Error updating ticket: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 