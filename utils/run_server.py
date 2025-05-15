from app import app

if __name__ == "__main__":
    print("Starting WerTigo API server...")
    print("Database connection initialized.")
    app.run(host='0.0.0.0', port=5000, debug=True) 