from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
import os
from werkzeug.security import generate_password_hash, check_password_hash
import json
import random
from datetime import datetime
from flask_mail import Mail, Message
from flask_socketio import SocketIO, emit
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import requests
import pickle
import numpy as np
import openai
from threading import Lock
import os
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from flask_socketio import SocketIO

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)

# Initialize Flask app
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['MONGO_URI'] = os.getenv('MONGO_URI', 'mongodb://localhost:27017/medipredict')

# Initialize extensions
socketio = SocketIO(app)
mail = Mail(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Set up MongoDB client
client = MongoClient(app.config['MONGO_URI'])
db = client.medipredict

# Add is_authenticated to Jinja2 environment
app.jinja_env.globals.update(is_authenticated=lambda: 'user_id' in session)

# Collections
users_collection = db['users']
predictions_collection = db['predictions']
chat_history_collection = db['chat_history']

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.email = user_data['email']
        self.full_name = user_data.get('full_name', '')
        self.password_hash = user_data['password_hash']
        self.created_at = user_data.get('created_at')
        self.last_login = datetime.utcnow()

    def get_id(self):
        return self.id

    def update_last_login(self):
        users_collection.update_one(
            {'_id': ObjectId(self.id)},
            {'$set': {'last_login': datetime.utcnow()}}
        )

@login_manager.user_loader
def load_user(user_id):
    try:
        user_data = users_collection.find_one({'_id': ObjectId(user_id)})
        if user_data:
            return User(user_data)
        return None
    except Exception as e:
        print(f"Error loading user: {str(e)}")
        return None

# Function to save user
def save_user(user_data):
    try:
        # Check if user already exists
        existing_user = users_collection.find_one({'email': user_data['email']})
        if existing_user:
            # Update existing user
            users_collection.update_one(
                {'email': user_data['email']},
                {'$set': user_data}
            )
            return str(existing_user['_id'])
        else:
            # Create new user
            result = users_collection.insert_one(user_data)
            return str(result.inserted_id)
    except Exception as e:
        print(f"Error saving user: {str(e)}")
        return None

# Function to get user by email
def get_user_by_email(email):
    try:
        user_data = users_collection.find_one({'email': email})
        if user_data:
            return User(user_data)
        return None
    except Exception as e:
        print(f"Error getting user: {str(e)}")
        return None

# Function to save user data
def save_user_data(user_data):
    try:
        user = users_collection.find_one({'email': user_data['email']})
        if user:
            # Update existing user
            users_collection.update_one(
                {'email': user_data['email']},
                {'$set': user_data}
            )
            return user['_id']
        else:
            # Create new user
            result = users_collection.insert_one(user_data)
            return result.inserted_id
    except Exception as e:
        print(f"Error saving user: {str(e)}")
        return None

# Function to get user data
def get_user(user_id):
    try:
        return users_collection.find_one({'_id': ObjectId(user_id)})
    except Exception as e:
        print(f"Error getting user: {str(e)}")
        return None

# Function to save prediction
def save_prediction(prediction_data):
    try:
        result = predictions_collection.insert_one(prediction_data)
        return result.inserted_id
    except Exception as e:
        print(f"Error saving prediction: {str(e)}")
        return None

# Function to get user predictions
def get_user_predictions(user_id):
    try:
        predictions = predictions_collection.find({'user_id': user_id})
        return list(predictions)
    except Exception as e:
        print(f"Error getting predictions: {str(e)}")
        return []

# Function to save chat message
def save_chat_message(message_data):
    try:
        result = chat_history_collection.insert_one(message_data)
        return result.inserted_id
    except Exception as e:
        print(f"Error saving chat message: {str(e)}")
        return None

# Function to get chat history
def get_chat_history(user_id):
    try:
        messages = chat_history_collection.find({'user_id': user_id})
        return list(messages)
    except Exception as e:
        print(f"Error getting chat history: {str(e)}")
        return []

# Add is_authenticated to Jinja2 environment
@app.context_processor
def inject_is_authenticated():
    return dict(is_authenticated=lambda: 'user_id' in session)

# Function to check if user is authenticated
def is_authenticated():
    user_id = session.get('user_id')
    if not user_id:
        return False
    user = get_user(user_id)
    return user is not None

# Load ML models for disease prediction
def load_ml_models():
    try:
        models = {}
        model_files = {
            'diabetes': 'diabetes.pkl',
            'heart': 'heart.pkl',
            'parkinsons': 'parkinsons_model.pkl',
            'kidney': 'kidney.pkl',
            'liver': 'liver.pkl',
            'breast_cancer': 'breast_cancer.pkl'
        }
        
        # Get the absolute path to the models directory
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        
        # Check if models directory exists
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        # Load each model
        for disease, file_name in model_files.items():
            file_path = os.path.join(models_dir, file_name)
            
            # Check if model file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Model file not found: {file_path}")
            
            try:
                # Load the model using pickle
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
                models[disease] = model
                logging.info(f"Successfully loaded {disease} model")
            except Exception as e:
                logging.error(f"Error loading {disease} model: {str(e)}")
                models[disease] = None
        
        # Check if all required models were loaded
        missing_models = [d for d, m in models.items() if m is None]
        if missing_models:
            logging.warning(f"Warning: Missing models for diseases: {', '.join(missing_models)}")
            
        return models
    except Exception as e:
        logging.error(f"Error loading ML models: {str(e)}")
        return {}

# Load ML models at startup
models = load_ml_models()

# Function to check if a model is loaded
def is_model_loaded(disease):
    return disease in models and models[disease] is not None

# Function to get model
def get_model(disease):
    if not is_model_loaded(disease):
        raise ValueError(f"Model for {disease} is not loaded")
    return models[disease]

# Disease Parameters and Input Ranges
DISEASE_PARAMETERS = {
    "diabetes": {
        "pregnancy_count": {
            "description": "Number of times pregnant",
            "range": (0, 20),
            "unit": "times"
        },
        "glucose_level": {
            "description": "Plasma glucose concentration (2 hours in an oral glucose tolerance test)",
            "range": (70, 180),
            "unit": "mg/dL"
        },
        "blood_pressure": {
            "description": "Diastolic blood pressure",
            "range": (60, 120),
            "unit": "mm Hg"
        },
        "skin_thickness": {
            "description": "Triceps skin fold thickness",
            "range": (0, 100),
            "unit": "mm"
        },
        "insulin_level": {
            "description": "2-Hour serum insulin",
            "range": (0, 846),
            "unit": "mu U/ml"
        },
        "bmi": {
            "description": "Body mass index",
            "range": (18.5, 40),
            "unit": "weight in kg/(height in m)²"
        },
        "diabetes_pedigree": {
            "description": "Diabetes pedigree function (hereditary factor)",
            "range": (0.078, 2.42),
            "unit": ""
        },
        "age": {
            "description": "Age",
            "range": (21, 81),
            "unit": "years"
        }
    },
    "heart": {
        "age": {
            "description": "Age",
            "range": (20, 100),
            "unit": "years"
        },
        "sex": {
            "description": "Sex (0 = female, 1 = male)",
            "range": (0, 1),
            "unit": ""
        },
        "chest_pain": {
            "description": "Chest pain type (0-3)",
            "range": (0, 3),
            "unit": ""
        },
        "resting_bp": {
            "description": "Resting blood pressure",
            "range": (90, 200),
            "unit": "mm Hg"
        },
        "cholesterol": {
            "description": "Serum cholesterol",
            "range": (100, 600),
            "unit": "mg/dl"
        },
        "fasting_bs": {
            "description": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
            "range": (0, 1),
            "unit": ""
        },
        "resting_ecg": {
            "description": "Resting electrocardiographic results (0-2)",
            "range": (0, 2),
            "unit": ""
        },
        "max_hr": {
            "description": "Maximum heart rate achieved",
            "range": (60, 220),
            "unit": "bpm"
        },
        "exercise_angina": {
            "description": "Exercise induced angina (1 = yes, 0 = no)",
            "range": (0, 1),
            "unit": ""
        },
        "st_depression": {
            "description": "ST depression induced by exercise relative to rest",
            "range": (0, 6.2),
            "unit": "mm"
        },
        "slope": {
            "description": "Slope of the peak exercise ST segment (0-2)",
            "range": (0, 2),
            "unit": ""
        },
        "vessels": {
            "description": "Number of major vessels colored by fluoroscopy (0-4)",
            "range": (0, 4),
            "unit": ""
        },
        "thalassemia": {
            "description": "Thalassemia (0-3)",
            "range": (0, 3),
            "unit": ""
        }
    },
    "parkinsons": {
        "mdvp_fo": {
            "description": "Average vocal fundamental frequency",
            "range": (80, 260),
            "unit": "Hz"
        },
        "mdvp_fhi": {
            "description": "Maximum vocal fundamental frequency",
            "range": (100, 600),
            "unit": "Hz"
        },
        "mdvp_flo": {
            "description": "Minimum vocal fundamental frequency",
            "range": (60, 240),
            "unit": "Hz"
        },
        "mdvp_jitter_pct": {
            "description": "Measure of variation in fundamental frequency",
            "range": (0, 2),
            "unit": "%"
        },
        "mdvp_jitter_abs": {
            "description": "Absolute measure of variation in fundamental frequency",
            "range": (0, 0.0001),
            "unit": "ms"
        },
        "mdvp_rap": {
            "description": "Relative amplitude perturbation",
            "range": (0, 0.02),
            "unit": ""
        },
        "mdvp_ppq": {
            "description": "Five-point period perturbation quotient",
            "range": (0, 0.02),
            "unit": ""
        },
        "jitter_ddp": {
            "description": "Average absolute difference of differences between cycles",
            "range": (0, 0.03),
            "unit": ""
        },
        "mdvp_shimmer": {
            "description": "Local shimmer",
            "range": (0, 0.2),
            "unit": ""
        },
        "mdvp_shimmer_db": {
            "description": "Local shimmer in decibels",
            "range": (0, 2),
            "unit": "dB"
        }
    },
    "liver": {
        "age": {
            "description": "Age of the patient",
            "range": (4, 90),
            "unit": "years"
        },
        "total_bilirubin": {
            "description": "Total bilirubin level",
            "range": (0.1, 50),
            "unit": "mg/dL"
        },
        "direct_bilirubin": {
            "description": "Direct bilirubin level",
            "range": (0.1, 20),
            "unit": "mg/dL"
        },
        "alk_phosphate": {
            "description": "Alkaline phosphotase level",
            "range": (20, 300),
            "unit": "IU/L"
        },
        "sgpt": {
            "description": "Serum glutamic pyruvic transaminase level",
            "range": (1, 300),
            "unit": "IU/L"
        },
        "sgot": {
            "description": "Serum glutamic oxaloacetic transaminase level",
            "range": (1, 300),
            "unit": "IU/L"
        },
        "total_proteins": {
            "description": "Total proteins level",
            "range": (5.5, 9),
            "unit": "g/dL"
        },
        "albumin": {
            "description": "Albumin level",
            "range": (2.5, 5.5),
            "unit": "g/dL"
        },
        "albumin_globulin_ratio": {
            "description": "Ratio of albumin to globulin",
            "range": (0.3, 2.5),
            "unit": ""
        }
    },
    "kidney": {
        "age": {
            "description": "Age of the patient",
            "range": (2, 90),
            "unit": "years"
        },
        "blood_pressure": {
            "description": "Blood pressure",
            "range": (50, 180),
            "unit": "mm Hg"
        },
        "specific_gravity": {
            "description": "Specific gravity of urine",
            "range": (1.005, 1.030),
            "unit": ""
        },
        "albumin": {
            "description": "Albumin level",
            "range": (0, 5),
            "unit": ""
        },
        "sugar": {
            "description": "Sugar level",
            "range": (0, 5),
            "unit": ""
        },
        "red_blood_cells": {
            "description": "Red blood cells (0 = normal, 1 = abnormal)",
            "range": (0, 1),
            "unit": ""
        },
        "pus_cell": {
            "description": "Pus cell (0 = normal, 1 = abnormal)",
            "range": (0, 1),
            "unit": ""
        },
        "pus_cell_clumps": {
            "description": "Pus cell clumps (0 = not present, 1 = present)",
            "range": (0, 1),
            "unit": ""
        },
        "bacteria": {
            "description": "Bacteria (0 = not present, 1 = present)",
            "range": (0, 1),
            "unit": ""
        },
        "blood_glucose_random": {
            "description": "Random blood glucose level",
            "range": (70, 490),
            "unit": "mg/dL"
        },
        "blood_urea": {
            "description": "Blood urea level",
            "range": (1.5, 100),
            "unit": "mg/dL"
        },
        "serum_creatinine": {
            "description": "Serum creatinine level",
            "range": (0.4, 15),
            "unit": "mg/dL"
        },
        "sodium": {
            "description": "Sodium level",
            "range": (111, 160),
            "unit": "mEq/L"
        },
        "potassium": {
            "description": "Potassium level",
            "range": (2.5, 7.5),
            "unit": "mEq/L"
        },
        "hemoglobin": {
            "description": "Hemoglobin level",
            "range": (3.1, 17.8),
            "unit": "g/dL"
        }
    },
    "breast_cancer": {
        "radius_mean": {
            "description": "Mean of distances from center to points on the perimeter",
            "range": (6.5, 28),
            "unit": "mm"
        },
        "texture_mean": {
            "description": "Standard deviation of gray-scale values",
            "range": (9.7, 40),
            "unit": ""
        },
        "perimeter_mean": {
            "description": "Mean size of the core tumor",
            "range": (43, 190),
            "unit": "mm"
        },
        "area_mean": {
            "description": "Mean area of the core tumor",
            "range": (140, 2500),
            "unit": "sq. mm"
        },
        "smoothness_mean": {
            "description": "Mean of local variation in radius lengths",
            "range": (0.05, 0.16),
            "unit": ""
        },
        "compactness_mean": {
            "description": "Mean of perimeter^2 / area - 1.0",
            "range": (0.02, 0.35),
            "unit": ""
        },
        "concavity_mean": {
            "description": "Mean of severity of concave portions of the contour",
            "range": (0, 0.43),
            "unit": ""
        },
        "concave_points_mean": {
            "description": "Mean number of concave portions of the contour",
            "range": (0, 0.2),
            "unit": ""
        },
        "symmetry_mean": {
            "description": "Mean symmetry",
            "range": (0.1, 0.3),
            "unit": ""
        },
        "fractal_dimension_mean": {
            "description": "Mean 'coastline approximation' - 1",
            "range": (0.05, 0.1),
            "unit": ""
        }
    }
}

# OpenAI Configuration
openai.api_key = os.getenv('OPENAI_API_KEY')

# System Prompts for Medical AI Assistant
SYSTEM_PROMPTS = {
    "symptom_analysis": """
    You are a professional Medical AI Assistant. Your role is to:
    1. Analyze symptoms and assess severity
    2. Ask follow-up questions to refine diagnosis
    3. Provide evidence-based medical advice
    4. Identify emergency symptoms and suggest urgent care
    5. Offer preventive care recommendations
    
    Guidelines:
    - Always ask about symptom severity and duration
    - Consider underlying conditions and risk factors
    - Provide clear, actionable recommendations
    - Identify red flags for emergency care
    - Format responses in clear, structured JSON
    """,
    "disease_prediction": """
    You are a professional Medical AI Assistant specializing in disease prediction. Your role is to:
    1. Analyze input parameters for disease risk
    2. Provide evidence-based risk assessment
    3. Offer personalized prevention recommendations
    4. Identify high-risk factors
    5. Suggest appropriate medical follow-up
    
    Guidelines:
    - Base recommendations on latest medical research
    - Consider lifestyle factors in risk assessment
    - Provide specific, actionable advice
    - Format responses in clear, structured JSON
    """,
    "general_health": """
    You are a professional Medical AI Assistant specializing in general health and wellness. Your role is to:
    1. Provide evidence-based health guidance
    2. Offer personalized lifestyle recommendations
    3. Suggest preventive care measures
    4. Address common health concerns
    5. Promote healthy habits
    
    Guidelines:
    - Base recommendations on scientific evidence
    - Consider individual health profiles
    - Provide practical, achievable advice
    - Format responses in clear, structured JSON
    """,
    "skincare": """
    You are a professional Medical AI Assistant specializing in dermatology. Your role is to:
    1. Analyze skin conditions and symptoms
    2. Provide evidence-based skincare recommendations
    3. Suggest appropriate treatments
    4. Identify potential skin concerns
    5. Offer preventive skincare advice
    
    Guidelines:
    - Consider skin type and condition
    - Base recommendations on dermatological research
    - Provide specific product recommendations
    - Format responses in clear, structured JSON
    """,
    "lifestyle_wellness": """
    You are a professional Medical AI Assistant specializing in lifestyle wellness. Your role is to:
    1. Provide evidence-based fitness recommendations
    2. Offer mental health support
    3. Suggest stress management techniques
    4. Promote balanced nutrition
    5. Support overall wellness
    
    Guidelines:
    - Create personalized wellness plans
    - Consider individual health status
    - Provide practical, achievable goals
    - Format responses in clear, structured JSON
    """
}

# Hospital Finder Configuration
HOSPITAL_TYPES = {
    "general": "hospital",
    "emergency": "emergency",
    "specialist": "specialist",
    "pediatric": "pediatric",
    "ophthalmic": "ophthalmic",
    "cardiac": "cardiac",
    "cancer": "cancer"
}

# Function to analyze symptoms using OpenAI
def analyze_symptoms_with_ai(symptoms):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["symptom_analysis"]},
                {"role": "user", "content": f"Analyze these symptoms: {symptoms}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing symptoms: {str(e)}"

# Function to get professional recommendations
def get_professional_recommendations(disease, inputs):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["disease_prediction"]},
                {"role": "user", "content": f"Analyze these inputs for {disease}: {inputs}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting recommendations: {str(e)}"

# Function to validate inputs
def validate_inputs(disease, inputs):
    errors = []
    for param, value in inputs.items():
        if param in DISEASE_PARAMETERS[disease]:
            min_val, max_val = DISEASE_PARAMETERS[disease][param]["range"]
            try:
                value = float(value)
                if value < min_val or value > max_val:
                    errors.append(f"{param} must be between {min_val} and {max_val}")
            except ValueError:
                errors.append(f"{param} must be a number")
    return errors

# Function to find nearby hospitals
def find_nearby_hospitals(location, hospital_type="general", max_results=5):
    try:
        url = f"https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            'query': f'"{location}" {HOSPITAL_TYPES.get(hospital_type, "hospital")}',
            'key': os.getenv('GOOGLE_API_KEY'),
            'type': 'hospital',
            'radius': '5000',
            'language': 'en'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get('status') != 'OK':
            return {"error": f"Error finding hospitals: {data.get('status')}"}
        
        # Process results
        hospitals = []
        for result in data.get('results', [])[:max_results]:
            hospital = {
                'name': result.get('name', 'Unknown Hospital'),
                'address': result.get('vicinity', 'Address not available'),
                'rating': result.get('rating', 0),
                'open_now': result.get('opening_hours', {}).get('open_now', False),
                'types': result.get('types', []),
                'phone': result.get('formatted_phone_number', 'Phone not available'),
                'website': result.get('website', 'Website not available')
            }
            hospitals.append(hospital)
        
        return {
            'hospitals': hospitals,
            'total_results': len(data.get('results', [])),
            'location': location,
            'hospital_type': hospital_type
        }
        
    except Exception as e:
        return {"error": f"Error finding hospitals: {str(e)}"}

# Function to get hospital details
def get_hospital_details(place_id):
    try:
        url = f"https://maps.googleapis.com/maps/api/place/details/json"
        params = {
            'place_id': place_id,
            'key': os.getenv('GOOGLE_API_KEY'),
            'fields': 'name,rating,formatted_phone_number,website,opening_hours,photos'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get('status') != 'OK':
            return {"error": f"Error getting hospital details: {data.get('status')}"}
        
        result = data.get('result', {})
        return {
            'name': result.get('name', 'Unknown Hospital'),
            'rating': result.get('rating', 0),
            'phone': result.get('formatted_phone_number', 'Phone not available'),
            'website': result.get('website', 'Website not available'),
            'opening_hours': result.get('opening_hours', {}),
            'photos': result.get('photos', [])
        }
        
    except Exception as e:
        return {"error": f"Error getting hospital details: {str(e)}"}

# Function to get hospital reviews
def get_hospital_reviews(place_id, max_reviews=5):
    try:
        url = f"https://maps.googleapis.com/maps/api/place/details/json"
        params = {
            'place_id': place_id,
            'key': os.getenv('GOOGLE_API_KEY'),
            'fields': 'reviews'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get('status') != 'OK':
            return {"error": f"Error getting hospital reviews: {data.get('status')}"}
        
        result = data.get('result', {})
        reviews = result.get('reviews', [])[:max_reviews]
        
        formatted_reviews = []
        for review in reviews:
            formatted_reviews.append({
                'author': review.get('author_name', 'Anonymous'),
                'rating': review.get('rating', 0),
                'text': review.get('text', 'No review text'),
                'time': review.get('relative_time_description', 'Unknown time')
            })
        
        return {
            'reviews': formatted_reviews,
            'total_reviews': len(reviews)
        }
        
    except Exception as e:
        return {"error": f"Error getting hospital reviews: {str(e)}"}

# Enhanced Chat API for better conversation flow
def chat_api():
    if not is_authenticated():
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        data = request.json
        message = data.get('message', '').lower()
        user_id = session.get('user_id')
        
        # Save user message
        save_chat_message({
            'user_id': user_id,
            'message': message,
            'timestamp': datetime.utcnow(),
            'type': 'user'
        })
        
        # Process message and generate response
        response = process_chat_message(message)
        
        # Save AI response
        save_chat_message({
            'user_id': user_id,
            'message': response,
            'timestamp': datetime.utcnow(),
            'type': 'ai'
        })
        
        return jsonify({
            'response': response,
            'type': 'ai_response'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Enhanced message processing for better conversation flow
def process_chat_message(message):
    try:
        # Check for greetings
        greetings = ['hello', 'hi', 'hey', 'greetings']
        if any(greet in message for greet in greetings):
            return "Hi there! I'm here to help with your health concerns. How can I assist you today?"
        
        # Check for disease-related queries
        diseases = ['diabetes', 'heart disease', 'parkinsons', 'kidney disease', 'liver disease', 'breast cancer']
        if any(disease in message for disease in diseases):
            disease = next((d for d in diseases if d in message), None)
            return f"I can help you with {disease.title()}. Please share your symptoms or test results, and I'll provide a detailed analysis."
        
        # Check for symptom analysis
        symptoms = ['symptoms', 'feeling', 'pain', 'nausea', 'fatigue']
        if any(symptom in message for symptom in symptoms):
            return "I can help analyze your symptoms. Please describe what you're experiencing, and I'll provide a professional assessment."
        
        # Check for hospital search
        hospital_keywords = ['hospital', 'clinic', 'emergency', 'nearby', 'medical center']
        if any(keyword in message for keyword in hospital_keywords):
            return "I can help you find nearby hospitals. Please let me know your location and any specific type of hospital you're looking for."
        
        # Check for general health advice
        health_keywords = ['health', 'nutrition', 'exercise', 'fitness', 'wellness']
        if any(keyword in message for keyword in health_keywords):
            return "I can provide personalized health advice. What specific aspect of your health would you like to discuss?"
        
        # Default response
        return "I'm here to help! You can ask about:\n\n1. Disease prediction and risk assessment\n2. Symptom analysis\n3. Finding nearby hospitals\n4. General health advice\n\nWhat would you like to know more about?"
        
    except Exception as e:
        return f"I'm sorry, I encountered an error: {str(e)}. Please try rephrasing your question."

# Enhanced prediction route with better conversation flow
def predict():
    if not is_authenticated():
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        data = request.json
        disease = data.get('disease').lower()
        inputs = data.get('inputs', {})
        
        # Validate inputs
        errors = validate_inputs(disease, inputs)
        if errors:
            return jsonify({
                'error': f"I noticed some issues with your inputs: {', '.join(errors)}\n\nPlease provide values within the valid ranges for accurate results."
            }), 400
        
        # Get prediction with confidence score
        prediction = predict_specific_disease(disease, inputs)
        
        if 'error' in prediction:
            return jsonify({
                'response': f"I'm sorry, I encountered an issue: {prediction['error']}\n\nWould you like to try again or ask about something else?"
            })
        
        # Prepare prediction data for database
        prediction_data = {
            "user_id": session.get('user_id'),
            "disease": disease,
            "inputs": inputs,
            "prediction": prediction["prediction"],
            "confidence_score": prediction["confidence_score"],
            "recommendations": prediction["recommendations"],
            "risk_factors": prediction["risk_factors"],
            "timestamp": datetime.utcnow()
        }
        
        # Save prediction to database
        prediction_id = save_prediction(prediction_data)
        if prediction_id:
            # Format response in a conversational way
            response = f"Based on your inputs, here's what I found:\n\n"\
                       f"Prediction: {prediction['prediction']}\n"\
                       f"Confidence: {prediction['confidence_score']}\n\n"\
                       f"{prediction['risk_factors']}\n\n"\
                       f"My recommendations:\n"\
                       f"{'\n'.join([f'• {rec}' for rec in prediction['recommendations']])}\n\n"\
                       f"Would you like to know more about any of these points or discuss other health concerns?"
            
            return jsonify({
                'response': response,
                'type': 'prediction_result',
                'prediction_id': str(prediction_id)
            })
        else:
            return jsonify({
                'error': "Sorry, there was an issue saving your prediction. Please try again."
            }), 500
        
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({
            'error': f"I'm sorry, I encountered an error: {str(e)}\n\nPlease try again or ask about something else."
        }), 500

# Update predict_specific_disease to use the new model loading functions
def predict_specific_disease(disease, user_inputs):
    try:
        # Check if disease is supported
        if disease not in DISEASE_PARAMETERS:
            return {"error": f"Unsupported disease: {disease}"}
        
        # Get the model
        model = get_model(disease)
        
        # Prepare input data
        input_data = []
        for param, value in user_inputs.items():
            if param in DISEASE_PARAMETERS[disease]:
                min_val = DISEASE_PARAMETERS[disease][param]['range'][0]
                max_val = DISEASE_PARAMETERS[disease][param]['range'][1]
                
                try:
                    value = float(value)
                    if value < min_val or value > max_val:
                        return {"error": f"{param} must be between {min_val} and {max_val}"}
                    input_data.append(value)
                except ValueError:
                    return {"error": f"{param} must be a number"}
        
        # Make prediction
        prediction = model.predict([input_data])[0]
        
        # Get confidence score
        try:
            probabilities = model.predict_proba([input_data])[0]
            confidence_score = max(probabilities) * 100
            confidence_score = f"{confidence_score:.2f}%"
        except:
            confidence_score = "N/A"
        
        # Get risk factors analysis
        risk_factors = analyze_risk_factors(disease, user_inputs)
        
        # Get recommendations
        recommendations = get_professional_recommendations(disease, user_inputs)
        
        return {
            "prediction": "Positive" if prediction == 1 else "Negative",
            "confidence_score": confidence_score,
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"Error in predict_specific_disease: {str(e)}")
        return {"error": str(e)}

@app.route('/api/predict', methods=['POST'])
def predict_route():
    return predict()

@app.route('/api/chat', methods=['POST'])
def chat_api_route():
    return chat_api()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember_me = request.form.get('remember_me')
        
        user = get_user_by_email(email)
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            user.update_last_login()
            session['user_id'] = user.id
            if remember_me:
                session.permanent = True
            flash('Successfully logged in!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        full_name = request.form.get('full_name')
        
        if get_user_by_email(email):
            flash('Email already registered', 'error')
            return redirect(url_for('signup'))
        
        if len(password) < 8:
            flash('Password must be at least 8 characters long', 'error')
            return redirect(url_for('signup'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))
        
        user_data = {
            'email': email,
            'password_hash': generate_password_hash(password),
            'full_name': full_name,
            'created_at': datetime.utcnow(),
            'last_login': datetime.utcnow()
        }
        
        user_id = save_user_data(user_data)
        if user_id:
            user = User(user_data)
            login_user(user)
            flash('Account created successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Failed to create account', 'error')
            return redirect(url_for('signup'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    try:
        # Update last login time
        user = current_user
        user.update_last_login()
        
        # Clear the session
        session.clear()
        
        # Log out the user
        logout_user()
        
        flash('You have been logged out successfully!', 'success')
        return redirect(url_for('login'))
    except Exception as e:
        flash(f'Error logging out: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

# Dashboard Metrics
DASHBOARD_METRICS = {
    'total_predictions': 0,
    'recent_predictions': [],
    'accuracy_stats': {
        'diabetes': 0,
        'heart_disease': 0,
        'liver': 0,
        'breast_cancer': 0
    }
}

@app.route('/dashboard')
@login_required
def dashboard():
    # Get metrics
    metrics = DASHBOARD_METRICS
    
    # Get predictions
    predictions = get_user_predictions(current_user.id)
    
    return render_template('dashboard.html', 
                         metrics=metrics, 
                         predictions=predictions)

@app.route('/api/diagnosis', methods=['POST'])
@login_required
def diagnose():
    try:
        data = request.get_json()
        
        # Mock diagnosis logic
        symptoms = data.get('symptoms', '')
        medical_history = data.get('medicalHistory', '')
        family_history = data.get('familyHistory', '')
        lifestyle = data.get('lifestyle', '')
        
        # Generate mock results
        conditions = ['Heart Disease', 'Diabetes', 'Hypertension', 'Asthma']
        risk_levels = ['Low', 'Medium', 'High']
        
        condition = random.choice(conditions)
        risk_level = random.choice(risk_levels)
        
        recommendations = [
            "Consult a healthcare professional",
            "Monitor symptoms",
            "Follow prescribed treatment"
        ]
        
        next_steps = [
            "Schedule an appointment with your physician",
            "Follow prescribed medication",
            "Monitor symptoms closely"
        ]
        
        return jsonify({
            'condition': condition,
            'riskLevel': risk_level,
            'recommendations': recommendations,
            'nextSteps': next_steps,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat')
@login_required
def chat():
    return render_template('chat.html',
                         current_user={'username': current_user.full_name})

@app.route('/predictions', methods=['GET', 'POST'])
@login_required
def predictions():
    # Get chat summaries and predictions
    summaries = []
    for summary in CHAT_SUMMARIES:
        if 'summary' in summary:
            summaries.append({
                'date': summary['date'],
                'summary': summary['summary']
            })
    
    return render_template('prediction_summary.html',
                         summaries=summaries,
                         predictions=get_user_predictions(current_user.id))

@app.route('/find_hospitals')
@login_required
def find_hospitals():
    return render_template('find_hospitals.html')

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html',
                         current_user={'name': get_user(session['user_id'])['full_name']})

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    try:
        data = request.get_json()
        user = get_user(session['user_id'])
        
        if 'full_name' in data:
            user['full_name'] = data['full_name']
        if 'email' in data:
            user['email'] = data['email']
        
        users_collection.update_one(
            {'_id': user['_id']},
            {'$set': user}
        )
        
        return jsonify({'message': 'Profile updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    try:
        data = request.get_json()
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        confirm_password = data.get('confirm_password')
        
        user = get_user(session['user_id'])
        
        if not check_password_hash(user['password_hash'], current_password):
            return jsonify({'error': 'Current password is incorrect'}), 400
            
        if new_password != confirm_password:
            return jsonify({'error': 'New passwords do not match'}), 400
            
        if len(new_password) < 8:
            return jsonify({'error': 'New password must be at least 8 characters long'}), 400
            
        user['password_hash'] = generate_password_hash(new_password)
        
        users_collection.update_one(
            {'_id': user['_id']},
            {'$set': user}
        )
        
        return jsonify({'message': 'Password changed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/prediction/<prediction_id>')
@login_required
def prediction_detail(prediction_id):
    try:
        # Convert prediction_id from string to ObjectId
        prediction_id = ObjectId(prediction_id)
        
        # Get the prediction from the database
        prediction = predictions_collection.find_one({'_id': prediction_id})
        
        if not prediction:
            return render_template('error.html', message="Prediction not found"), 404
            
        # Get user information
        user = get_user(prediction['user_id'])
        if not user:
            return render_template('error.html', message="User not found"), 404
            
        # Format prediction data for display
        formatted_prediction = {
            'disease': prediction['disease'].title(),
            'prediction': prediction['prediction'],
            'confidence_score': prediction['confidence_score'],
            'risk_factors': prediction['risk_factors'],
            'recommendations': prediction['recommendations'],
            'timestamp': prediction['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'inputs': prediction['inputs']
        }
        
        return render_template('prediction_detail.html', 
                             prediction=formatted_prediction,
                             user=user)
        
    except Exception as e:
        print(f"Error in prediction_detail: {str(e)}")
        return render_template('error.html', message=f"Error viewing prediction: {str(e)}"), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    
    # Send initial metrics
    socketio.emit('metrics_update', DASHBOARD_METRICS)
    
    # Send all predictions
    socketio.emit('predictions_update', get_user_predictions(session['user_id']))

@app.route('/chat', methods=['POST'])
def chatbot():
    data = request.json
    user_input = data.get("message", "")

    # Determine Intent
    intent = get_intent(user_input)
    
    if intent == "greeting":
        return jsonify({"reply": "Hello! How can I assist you today?"})
    elif intent == "farewell":
        return jsonify({"reply": "Goodbye! Take care!"})
    elif intent == "general_query":
        return jsonify({"reply": "I can help with symptom checking, disease prediction, and finding hospitals. What would you like to do?"})

    return jsonify({"reply": "I'm here to assist. How can I help?"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    disease = data.get("disease")
    user_inputs = data.get("inputs")

    # Generate input summary
    summary = summarize_inputs(disease, user_inputs)

    # Get ML model prediction
    prediction = predict_specific_disease(disease, user_inputs)

    return jsonify({"summary": summary, "prediction": prediction})

@app.route("/find_hospital", methods=["POST"])
def hospital_finder():
    data = request.json
    city = data.get("city")
    disease = data.get("disease")

    response = find_nearby_hospitals(city, disease)
    return jsonify({"reply": response})

def main():
    try:
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")

if __name__ == '__main__':
    main()
