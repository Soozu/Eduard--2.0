-- Create tickets table if it doesn't exist
CREATE TABLE IF NOT EXISTS tickets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticket_id VARCHAR(20) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL,
    trip_id VARCHAR(36) NULL,
    itinerary MEDIUMTEXT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at VARCHAR(30) NOT NULL,
    updated_at VARCHAR(30) NOT NULL,
    INDEX idx_ticket_id (ticket_id),
    INDEX idx_email (email),
    INDEX idx_trip_id (trip_id),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Comment explaining the table structure
-- ticket_id: A unique identifier for the ticket (e.g., TKT-12345)
-- email: Email address of the user who created the ticket
-- trip_id: Optional reference to a trip in the trips table
-- itinerary: JSON-encoded itinerary data
-- status: Status of the ticket (active, completed, cancelled)
-- created_at: ISO-formatted datetime when the ticket was created
-- updated_at: ISO-formatted datetime when the ticket was last updated 