# WerTigo Travel Recommendation System

A travel recommendation system with chat interface that uses a RoBERTa-based model to analyze user queries and recommend travel destinations.

## Features

- AI-powered travel recommendations based on user queries
- Conversational interface with follow-up suggestions
- Trip planning with customized itineraries
- User authentication and trip saving
- Interactive map with route planning
- Responsive UI with star ratings and clean design

## Prerequisites

- Python 3.7 or higher
- MySQL Server 5.7 or higher
- Web browser (Chrome, Firefox, Edge recommended)

## Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/your-repo/wertigo.git
cd wertigo
```

### 2. Configure MySQL

1. Install MySQL if you haven't already.
2. Create a MySQL user or use the root user.
3. Open `db_config.py` and update the configuration with your MySQL credentials:

```python
# Database connection settings
DB_CONFIG = {
    'user': 'your_mysql_username',
    'password': 'your_mysql_password',
    'host': 'localhost',
    'database': 'wertigo_db',
    'port': 3306,
    'raise_on_warnings': True
}
```

### 3. Run the installation script

#### On Windows:
```
install.bat
```

#### On Linux/Mac:
```
chmod +x install.sh
./install.sh
```

This script will:
- Install required Python packages
- Create the database and tables
- Import the destination data from CSV file
- Create a test user for demo purposes

### 4. Start the server

```
python app.py
```

### 5. Open the application

Open `index.html` in your web browser. You can use the application as a guest, or sign in using:

- Email: test@example.com
- Password: password123

## Using the Application

1. **Ask for recommendations**: Type queries like "I want to visit beaches in Tagaytay" or "Show me historical sites in Cavite"
2. **Create a trip**: Click the "create a new trip" button to start planning a full trip itinerary
3. **Save trips**: Sign in to save your planned trips for later
4. **View routes**: For planned trips, click "View Route on Map" to see the route on the map

## Architecture

- **Frontend**: HTML, CSS, JavaScript with Leaflet.js for maps
- **Backend**: Flask API server
- **Machine Learning**: RoBERTa-based model for query analysis
- **Database**: MySQL for storing destinations, users, trips, and preferences

## Troubleshooting

- **Database Connection Issues**: Check your MySQL credentials in `db_config.py`
- **Missing Packages**: Run `pip install -r requirements.txt` to make sure all required packages are installed
- **CORS Issues**: Make sure you're running the backend server (app.py) when using the frontend

## License

This project is licensed under the MIT License - see the LICENSE file for details. 