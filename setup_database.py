import mysql.connector
from mysql.connector import errorcode
import pandas as pd
import sys
import os
from db_config import DB_CONFIG

def create_database(cursor, db_name):
    try:
        cursor.execute(f"CREATE DATABASE {db_name} DEFAULT CHARACTER SET 'utf8'")
        print(f"Database {db_name} created successfully.")
    except mysql.connector.Error as err:
        print(f"Failed creating database: {err}")
        exit(1)

def connect_to_database():
    try:
        # First try to connect to the database
        cnx = mysql.connector.connect(**DB_CONFIG)
        print("Connected to database successfully")
        return cnx
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Access denied: Check your username and password")
            exit(1)
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            # Database doesn't exist, let's create it
            print("Database doesn't exist, creating it...")
            cnx = mysql.connector.connect(
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                host=DB_CONFIG['host']
            )
            cursor = cnx.cursor()
            create_database(cursor, DB_CONFIG['database'])
            cnx.database = DB_CONFIG['database']
            return cnx
        else:
            print(f"Database connection error: {err}")
            exit(1)

def create_tables(cursor):
    """Create the required tables"""
    
    TABLES = {}
    
    # Destinations table
    TABLES['destinations'] = (
        "CREATE TABLE IF NOT EXISTS `destinations` ("
        "  `id` int NOT NULL AUTO_INCREMENT,"
        "  `name` varchar(255) NOT NULL,"
        "  `city` varchar(255) NOT NULL,"
        "  `province` varchar(255) NOT NULL,"
        "  `description` text NOT NULL,"
        "  `category` varchar(255) NOT NULL,"
        "  `metadata` text,"
        "  `ratings` float,"
        "  `budget` varchar(255),"
        "  `latitude` float,"
        "  `longitude` float,"
        "  `operating_hours` varchar(255),"
        "  `contact_information` text,"
        "  PRIMARY KEY (`id`),"
        "  FULLTEXT KEY `ft_destination` (`name`, `city`, `description`, `category`, `metadata`)"
        ") ENGINE=InnoDB"
    )
    
    # Users table
    TABLES['users'] = (
        "CREATE TABLE IF NOT EXISTS `users` ("
        "  `id` int NOT NULL AUTO_INCREMENT,"
        "  `username` varchar(255) NOT NULL,"
        "  `email` varchar(255) NOT NULL,"
        "  `password` varchar(255) NOT NULL,"
        "  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  PRIMARY KEY (`id`),"
        "  UNIQUE KEY `email` (`email`)"
        ") ENGINE=InnoDB"
    )
    
    # Sessions table
    TABLES['sessions'] = (
        "CREATE TABLE IF NOT EXISTS `sessions` ("
        "  `id` varchar(36) NOT NULL,"
        "  `user_id` int DEFAULT NULL,"
        "  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  `last_activity` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,"
        "  PRIMARY KEY (`id`),"
        "  KEY `user_id` (`user_id`),"
        "  CONSTRAINT `sessions_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE"
        ") ENGINE=InnoDB"
    )
    
    # Trips table
    TABLES['trips'] = (
        "CREATE TABLE IF NOT EXISTS `trips` ("
        "  `id` int NOT NULL AUTO_INCREMENT,"
        "  `user_id` int NOT NULL,"
        "  `destination` varchar(255) NOT NULL,"
        "  `travel_dates` varchar(255) NOT NULL,"
        "  `travelers` int NOT NULL,"
        "  `budget` varchar(255),"
        "  `interests` text,"
        "  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  PRIMARY KEY (`id`),"
        "  KEY `user_id` (`user_id`),"
        "  CONSTRAINT `trips_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE"
        ") ENGINE=InnoDB"
    )
    
    # Trip Itineraries table
    TABLES['trip_itineraries'] = (
        "CREATE TABLE IF NOT EXISTS `trip_itineraries` ("
        "  `id` int NOT NULL AUTO_INCREMENT,"
        "  `trip_id` int NOT NULL,"
        "  `day` int NOT NULL,"
        "  `itinerary_data` json NOT NULL,"
        "  PRIMARY KEY (`id`),"
        "  KEY `trip_id` (`trip_id`),"
        "  CONSTRAINT `trip_itineraries_ibfk_1` FOREIGN KEY (`trip_id`) REFERENCES `trips` (`id`) ON DELETE CASCADE"
        ") ENGINE=InnoDB"
    )
    
    # User preferences table
    TABLES['preferences'] = (
        "CREATE TABLE IF NOT EXISTS `preferences` ("
        "  `id` int NOT NULL AUTO_INCREMENT,"
        "  `user_id` int NOT NULL,"
        "  `preference_type` varchar(50) NOT NULL,"
        "  `preference_value` varchar(255) NOT NULL,"
        "  `count` int NOT NULL DEFAULT 1,"
        "  `last_updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,"
        "  PRIMARY KEY (`id`),"
        "  UNIQUE KEY `user_preference` (`user_id`, `preference_type`, `preference_value`),"
        "  CONSTRAINT `preferences_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE"
        ") ENGINE=InnoDB"
    )
    
    # Conversation History table
    TABLES['conversations'] = (
        "CREATE TABLE IF NOT EXISTS `conversations` ("
        "  `id` int NOT NULL AUTO_INCREMENT,"
        "  `session_id` varchar(36) NOT NULL,"
        "  `user_id` int DEFAULT NULL,"
        "  `user_message` text NOT NULL,"
        "  `system_response` text NOT NULL,"
        "  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  PRIMARY KEY (`id`),"
        "  KEY `session_id` (`session_id`),"
        "  KEY `user_id` (`user_id`),"
        "  CONSTRAINT `conversations_ibfk_1` FOREIGN KEY (`session_id`) REFERENCES `sessions` (`id`) ON DELETE CASCADE,"
        "  CONSTRAINT `conversations_ibfk_2` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE SET NULL"
        ") ENGINE=InnoDB"
    )
    
    for table_name in TABLES:
        table_description = TABLES[table_name]
        try:
            print(f"Creating table {table_name}: ", end='')
            cursor.execute(table_description)
            print("OK")
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print("already exists.")
            else:
                print(err.msg)
        else:
            print(f"Table {table_name} created successfully")

def import_csv_data(cursor, cnx):
    """Import data from CSV file into the destinations table"""
    try:
        # Read CSV data
        csv_file = "newdataset.csv"
        
        # Check if file exists
        if not os.path.exists(csv_file):
            print(f"Error: CSV file '{csv_file}' not found in the current directory.")
            print(f"Current working directory: {os.getcwd()}")
            return
            
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            print("Please check if the file is valid and properly formatted.")
            return
        
        # Rename columns to match database structure (replace spaces with underscores)
        df.columns = [col.replace(' ', '_') for col in df.columns]
        
        # Clean NaN values
        df = df.fillna('')
        
        # Check required columns exist
        required_columns = ['name', 'city', 'province', 'description', 'category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: The following required columns are missing from the CSV: {', '.join(missing_columns)}")
            return
        
        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM destinations")
        count = cursor.fetchone()[0]
        
        if count > 0:
            choice = input("Data already exists in the destinations table. Do you want to replace it? (y/n): ")
            if choice.lower() != 'y':
                print("Skipping data import.")
                return
            else:
                cursor.execute("TRUNCATE TABLE destinations")
                print("Existing data cleared.")
        
        # Prepare the INSERT statement
        add_destination = (
            "INSERT INTO destinations "
            "(name, city, province, description, category, metadata, ratings, budget, "
            "latitude, longitude, operating_hours, contact_information) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        
        # Insert each row into the database
        success_count = 0
        error_count = 0
        
        for idx, row in df.iterrows():
            try:
                # Convert empty strings for numeric fields to None/NULL
                budget = row['budget'] if row['budget'] != '' else None  # Keep as string
                
                # Handle numeric conversions safely
                try:
                    ratings = float(row['ratings']) if row['ratings'] != '' else None
                except (ValueError, TypeError):
                    ratings = None
                    print(f"Warning: Invalid rating value '{row['ratings']}' in row {idx+1}, setting to NULL")
                
                try:
                    latitude = float(row['latitude']) if row['latitude'] != '' else None
                except (ValueError, TypeError):
                    latitude = None
                    print(f"Warning: Invalid latitude value '{row['latitude']}' in row {idx+1}, setting to NULL")
                
                try:
                    longitude = float(row['longitude']) if row['longitude'] != '' else None
                except (ValueError, TypeError):
                    longitude = None
                    print(f"Warning: Invalid longitude value '{row['longitude']}' in row {idx+1}, setting to NULL")
                
                data = (
                    row['name'],
                    row['city'],
                    row['province'],
                    row['description'],
                    row['category'],
                    row.get('metadata', ''),  # Use get() to handle missing columns
                    ratings,
                    budget,
                    latitude,
                    longitude,
                    row.get('operating_hours', ''),
                    row.get('contact_information', '')
                )
                
                cursor.execute(add_destination, data)
                success_count += 1
                
                # Commit in batches to avoid memory issues with large datasets
                if idx > 0 and idx % 100 == 0:
                    cnx.commit()
                    print(f"Processed {idx+1} rows...")
                
            except Exception as e:
                error_count += 1
                print(f"Error inserting row {idx+1}: {e}")
                print(f"Row data: {row}")
                # Continue with the next row
        
        # Final commit
        cnx.commit()
        
        if error_count > 0:
            print(f"Import completed with {error_count} errors. Successfully imported {success_count} destinations.")
        else:
            print(f"Successfully imported all {success_count} destinations into the database.")
        
    except Exception as e:
        print(f"Error during CSV import: {e}")
        # Don't re-raise the exception to allow the script to continue
        return False
    
    return True

def main():
    # Connect to the database
    cnx = connect_to_database()
    cursor = cnx.cursor()
    
    # Create tables
    create_tables(cursor)
    
    # Import data from CSV
    import_success = import_csv_data(cursor, cnx)
    
    if not import_success:
        print("\nNote: If you want to manually import the data later, you can:")
        print("1. Ensure 'newdataset.csv' is in the correct directory")
        print("2. Make sure the CSV file has the correct format")
        print("3. Run this script again or use a database management tool to import the data")
    
    # Create a test user for demonstration purposes
    try:
        print("Creating a test user...")
        add_user = (
            "INSERT INTO users (username, email, password) "
            "VALUES (%s, %s, %s)"
        )
        cursor.execute(add_user, ('testuser', 'test@example.com', 'password123'))
        cnx.commit()
        print("Test user created.")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_DUP_ENTRY:
            print("Test user already exists.")
        else:
            print(f"Error creating test user: {err}")
    
    # Close connections
    cursor.close()
    cnx.close()
    
    print("\nDatabase setup completed!")
    print("You can now run the application with 'python app.py'")
    print("Access the web interface by opening 'index.html' in your browser")

if __name__ == "__main__":
    main() 