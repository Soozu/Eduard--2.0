import requests
import json

# Test the API
BASE_URL = "http://localhost:5000"

def test_recommendation():
    """Test the recommendation endpoint"""
    print("=== Testing Recommendation API ===")
    
    # Create a session
    session_response = requests.post(f"{BASE_URL}/api/create_session")
    if session_response.status_code != 201:
        print(f"Failed to create session: {session_response.text}")
        return
    
    session_id = session_response.json().get("session_id")
    print(f"Created session: {session_id}")
    
    # Test a basic query
    query = "I want to visit beaches in Boracay"
    print(f"\nTesting query: '{query}'")
    
    response = requests.post(
        f"{BASE_URL}/api/recommend",
        json={"query": query, "session_id": session_id}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Status: Success")
        print(f"Detected city: {data.get('detected_city')}")
        print(f"Detected category: {data.get('detected_category')}")
        
        if 'recommendations' in data:
            print(f"Received {len(data['recommendations'])} recommendations")
            for i, rec in enumerate(data['recommendations']):
                print(f"  {i+1}. {rec.get('name')} ({rec.get('city')}) - {rec.get('category')}")
        else:
            print("No recommendations in response")
            print(data)
    else:
        print(f"Failed: {response.status_code}")
        print(response.text)

def test_trip_creation():
    """Test the trip creation endpoint"""
    print("\n=== Testing Trip Creation API ===")
    
    # Create a session
    session_response = requests.post(f"{BASE_URL}/api/create_session")
    session_id = session_response.json().get("session_id")
    
    # Test trip creation
    trip_data = {
        "destination": "Boracay",
        "travel_dates": "2023-06-01 to 2023-06-05",
        "travelers": 2,
        "budget": "mid-range",
        "interests": ["beach", "diving", "nightlife"]
    }
    
    print(f"Creating trip to {trip_data['destination']}")
    
    response = requests.post(
        f"{BASE_URL}/api/create_trip?session_id={session_id}",
        json=trip_data
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Status: Success")
        if 'trip' in data:
            trip = data['trip']
            print(f"Created trip to {trip.get('destination')}")
            print(f"Travel dates: {trip.get('travel_dates')}")
            print(f"Number of days in itinerary: {len(trip.get('itinerary', []))}")
            
            # Print first day of itinerary
            if trip.get('itinerary'):
                day = trip['itinerary'][0]
                print("\nFirst day includes:")
                for place in day.get('places', []):
                    print(f"  - {place.get('name')} ({place.get('category')})")
        else:
            print("No trip in response")
            print(data)
    else:
        print(f"Failed: {response.status_code}")
        print(response.text)

def test_special_query():
    """Test how the API responds to 'I want to plan dates'"""
    print("\n=== Testing Special Query ===")
    
    # Create a session
    session_response = requests.post(f"{BASE_URL}/api/create_session")
    session_id = session_response.json().get("session_id")
    print(f"Created session: {session_id}")
    
    # Test the query
    query = "I want to plan dates"
    print(f"\nTesting query: '{query}'")
    
    response = requests.post(
        f"{BASE_URL}/api/recommend",
        json={"query": query, "session_id": session_id}
    )
    
    print(f"Response status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("Running API tests...")
    
    # Run recommendation test
    test_recommendation()
    
    # Run trip creation test
    test_trip_creation()
    
    # Test the specific query causing problems
    test_special_query() 