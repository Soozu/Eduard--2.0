@echo off
echo WerTigo Travel Recommendation System Setup
echo ==========================================

echo Installing Python requirements...
pip install -r requirements.txt

echo Setting up the database...
python setup_database.py

echo Setup complete!
echo.
echo To start the application, run: python app.py
echo Then open index.html in your browser
echo.
pause 