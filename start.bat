@echo off
echo Starting WerTigo Travel App

echo Starting Flask backend (port 5000)...
start cmd /k python app.py

echo Starting frontend server (port 8000)...
start cmd /k python server.py

echo WerTigo Travel App is running!
echo Backend: http://localhost:5000
echo Frontend: http://localhost:8000
echo.
echo Press Ctrl+C in the respective command windows to stop the servers 