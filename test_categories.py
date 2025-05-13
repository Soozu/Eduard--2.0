import database as db

def test_categories():
    print("Testing get_distinct_categories function:")
    categories = db.get_distinct_categories()
    
    if categories:
        print(f"Found {len(categories)} categories:")
        for i, category in enumerate(categories):
            print(f"{i+1}. {category}")
    else:
        print("No categories found or error occurred.")

if __name__ == "__main__":
    test_categories() 