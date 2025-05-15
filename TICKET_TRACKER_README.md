# WerTigo Ticket Tracking System

This document explains how to set up and use the ticket tracking system in the WerTigo travel recommendation application.

## Overview

The ticket tracking system allows users to:

1. Create a tracking ticket for their travel itinerary
2. Look up their itinerary using a ticket ID and email
3. Track the status of their travel plans

## Database Setup

Before using the ticket tracking system, you need to set up the tickets table in your database.

1. Make sure your MySQL server is running
2. Run the `create_tickets_table.sql` script:

```bash
mysql -u root -p wertigo_db < create_tickets_table.sql
```

## Key Files

- `app.py`: Contains the API endpoints for ticket creation and retrieval
- `database.py`: Contains the database functions for ticket operations
- `tracker.html`: The frontend for looking up tickets
- `tracker.css`: Styles for the ticket tracker interface
- `index.html`: Updated to allow ticket creation when an itinerary is created

## API Endpoints

### Create Ticket

```
POST /api/create_ticket
```

Request body:
```json
{
  "email": "user@example.com",
  "trip_id": "12345",  // Optional
  "itinerary": { ... } // Optional, full itinerary object
}
```

Response:
```json
{
  "ticket_id": "TKT-AB123",
  "status": "created",
  "message": "Your ticket has been created. You can use this ticket ID to track your itinerary."
}
```

### Get Ticket

```
GET /api/tickets/{ticket_id}?email={email}
```

Response:
```json
{
  "ticket": {
    "ticket_id": "TKT-AB123",
    "status": "active",
    "created_at": "2023-05-15T10:30:00",
    "updated_at": "2023-05-15T10:30:00",
    "trip": { ... } // Trip data if available
  }
}
```

### Get Tickets by Email

```
GET /api/tickets?email={email}
```

Response:
```json
{
  "tickets": [
    {
      "ticket_id": "TKT-AB123",
      "status": "active",
      "created_at": "2023-05-15T10:30:00",
      "updated_at": "2023-05-15T10:30:00"
    },
    // More tickets...
  ]
}
```

## How to Use

### Creating a Ticket

1. When a user creates a travel itinerary in the AI chat interface, they will be prompted to enter their email to create a tracking ticket
2. After entering their email and clicking "Create Ticket", a ticket ID will be generated
3. The ticket ID and email are saved in localStorage for easy access

### Looking Up a Ticket

1. Navigate to `tracker.html`
2. Enter the ticket ID and email used when creating the ticket
3. Click "Look Up Ticket" to view the details of the itinerary

## Troubleshooting

If you encounter issues with the ticket tracking system:

1. Check that the MySQL server is running
2. Verify that the `tickets` table exists in the database
3. Check the browser console for any JavaScript errors
4. Check the server logs for any backend errors

## Future Enhancements

Planned future enhancements for the ticket tracking system:

1. Email notifications for ticket status changes
2. QR code generation for tickets
3. Integration with third-party travel services
4. Mobile app support 