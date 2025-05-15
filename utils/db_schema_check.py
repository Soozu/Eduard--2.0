import mysql.connector
from db_config import DB_CONFIG

def check_database_schema():
    """Check the database schema for potential issues"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # Show all tables
        print("Database tables:")
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = list(table.values())[0]
            print(f"- {table_name}")
        
        # Check destinations table
        print("\nChecking destinations table:")
        cursor.execute("DESCRIBE destinations")
        columns = cursor.fetchall()
        
        has_fulltext_index = False
        
        # Check for required columns
        required_columns = ['id', 'name', 'city', 'description', 'category']
        missing_columns = []
        
        for req_col in required_columns:
            if not any(col['Field'] == req_col for col in columns):
                missing_columns.append(req_col)
        
        if missing_columns:
            print(f"WARNING: Missing required columns: {', '.join(missing_columns)}")
        else:
            print("All required columns are present.")
        
        # Check for indexes
        cursor.execute("SHOW INDEX FROM destinations")
        indexes = cursor.fetchall()
        
        for idx in indexes:
            if idx.get('Index_type') == 'FULLTEXT':
                has_fulltext_index = True
                print(f"FULLTEXT index found: {idx.get('Key_name')} on column(s): {idx.get('Column_name')}")
        
        if not has_fulltext_index:
            print("WARNING: No FULLTEXT index found on destinations table. This may affect search performance.")
        
        # Check tickets table
        print("\nChecking tickets table:")
        try:
            cursor.execute("DESCRIBE tickets")
            columns = cursor.fetchall()
            
            # Check for required columns
            required_columns = ['ticket_id', 'email', 'itinerary', 'status']
            missing_columns = []
            
            for req_col in required_columns:
                if not any(col['Field'] == req_col for col in columns):
                    missing_columns.append(req_col)
            
            if missing_columns:
                print(f"WARNING: Missing required columns: {', '.join(missing_columns)}")
            else:
                print("All required columns are present.")
                
            # Check for indexes
            cursor.execute("SHOW INDEX FROM tickets")
            indexes = cursor.fetchall()
            
            has_ticket_id_index = False
            has_email_index = False
            
            for idx in indexes:
                if idx.get('Column_name') == 'ticket_id':
                    has_ticket_id_index = True
                elif idx.get('Column_name') == 'email':
                    has_email_index = True
            
            if not has_ticket_id_index:
                print("WARNING: No index found on ticket_id column. This may affect lookup performance.")
            if not has_email_index:
                print("WARNING: No index found on email column. This may affect lookup performance.")
                
        except mysql.connector.Error as e:
            print(f"ERROR: Could not check tickets table: {e}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error checking database schema: {e}")

if __name__ == "__main__":
    check_database_schema() 