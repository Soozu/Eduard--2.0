import database as db
import json

def test_fetch_functions():
    print("Testing database fetch functions...")
    
    # Test get_destinations
    print("\n1. Testing get_destinations():")
    destinations = db.get_destinations(limit=5)
    print(f"Retrieved {len(destinations)} destinations")
    if destinations:
        print(f"First destination: {destinations[0]['name']} in {destinations[0]['city']}")
    
    # Test search_destinations
    print("\n2. Testing search_destinations():")
    search_results = db.search_destinations("beach", limit=5)
    print(f"Search for 'beach' returned {len(search_results)} results")
    if search_results:
        print(f"Top result: {search_results[0]['name']} (relevance: {search_results[0].get('relevance', 'N/A')})")
    
    # Test get_distinct_cities
    print("\n3. Testing get_distinct_cities():")
    cities = db.get_distinct_cities()
    print(f"Found {len(cities)} distinct cities")
    if cities:
        print(f"First 5 cities: {', '.join(cities[:5])}")
    
    # Test get_distinct_categories
    print("\n4. Testing get_distinct_categories():")
    categories = db.get_distinct_categories()
    print(f"Found {len(categories)} distinct categories")
    if categories:
        print(f"Categories: {', '.join(categories[:5])}")
    
    # Test get_destination_by_id with the first destination's ID
    if destinations:
        first_id = destinations[0]['id']
        print(f"\n5. Testing get_destination_by_id({first_id}):")
        dest = db.get_destination_by_id(first_id)
        if dest:
            print(f"Retrieved: {dest['name']}")
        else:
            print("Failed to retrieve destination by ID")
    
    # Test ticket functions
    print("\n6. Testing ticket functions:")
    test_email = "test@example.com"
    tickets = db.get_tickets_by_email(test_email)
    print(f"Found {len(tickets)} tickets for email {test_email}")
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    test_fetch_functions() 