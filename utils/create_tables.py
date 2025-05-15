import mysql.connector
from db_config import DB_CONFIG

def create_tickets_table():
    """Create or update the tickets table structure"""
    print("Setting up tickets table...")
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SHOW TABLES LIKE 'tickets'")
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            # Create tickets table
            create_table_sql = """
            CREATE TABLE tickets (
                id INT AUTO_INCREMENT PRIMARY KEY,
                ticket_id VARCHAR(20) NOT NULL UNIQUE,
                email VARCHAR(255) NOT NULL,
                trip_id INT NULL,
                itinerary JSON NULL,
                status ENUM('active', 'completed', 'cancelled') NOT NULL DEFAULT 'active',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX (email),
                INDEX (trip_id),
                INDEX (status)
            ) ENGINE=InnoDB
            """
            cursor.execute(create_table_sql)
            print("Tickets table created.")
        else:
            # Check if the table has all required columns
            cursor.execute("DESCRIBE tickets")
            columns = cursor.fetchall()
            column_names = [col[0] for col in columns]
            
            # Check and add the itinerary column if missing
            if 'itinerary' not in column_names:
                cursor.execute("ALTER TABLE tickets ADD COLUMN itinerary JSON NULL AFTER trip_id")
                print("Added itinerary column to tickets table.")
            
            # Check and add proper indexes if missing
            cursor.execute("SHOW INDEX FROM tickets")
            indexes = cursor.fetchall()
            index_columns = [idx[4] for idx in indexes]  # Column_name is at index 4
            
            if 'ticket_id' not in index_columns:
                cursor.execute("ALTER TABLE tickets ADD UNIQUE INDEX (ticket_id)")
                print("Added index for ticket_id.")
                
            if 'email' not in index_columns:
                cursor.execute("ALTER TABLE tickets ADD INDEX (email)")
                print("Added index for email.")
                
            if 'status' not in index_columns:
                cursor.execute("ALTER TABLE tickets ADD INDEX (status)")
                print("Added index for status.")
            
            print("Tickets table structure verified.")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("Tickets table setup completed successfully.")
        return True
        
    except mysql.connector.Error as err:
        print(f"Error setting up tickets table: {err}")
        return False

if __name__ == "__main__":
    create_tickets_table() 