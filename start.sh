#!/bin/bash
echo "Starting WerTigo Travel App"

echo "Starting Flask backend (port 5000)..."
python app.py &
FLASK_PID=$!

echo "Starting frontend server (port 8000)..."
python server.py &
SERVER_PID=$!

echo "WerTigo Travel App is running!"
echo "Backend: http://localhost:5000"
echo "Frontend: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all servers"

function cleanup {
    echo "Stopping servers..."
    kill $FLASK_PID
    kill $SERVER_PID
    exit
}

trap cleanup INT
wait 