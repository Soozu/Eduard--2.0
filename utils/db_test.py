import mysql.connector
from db_config import DB_CONFIG

def test_connection():
    try:
        # Test the database connection
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Try executing a simple query
        cursor.execute("SELECT 'Connection successful'")
        result = cursor.fetchone()
        print(f"Database test: {result[0]}")
        
        # Test accessing the destinations table
        cursor.execute("SELECT COUNT(*) FROM destinations")
        count = cursor.fetchone()[0]
        print(f"Found {count} destinations in the database")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Database connection error: {e}")
        return False

if __name__ == "__main__":
    test_connection() 