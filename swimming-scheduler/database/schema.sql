-- Swimming Scheduler Database Schema
-- Create the database
CREATE DATABASE IF NOT EXISTS swimming_scheduler;
USE swimming_scheduler;

-- Users table to store friend information
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    phone VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Swimming events table
CREATE TABLE events (
    id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    location VARCHAR(200),
    created_by INT NOT NULL,
    event_date DATE NOT NULL,
    start_time TIME,
    end_time TIME,
    status ENUM('planning', 'confirmed', 'cancelled') DEFAULT 'planning',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE CASCADE
);

-- Proposed dates for events (when creating an event with multiple date options)
CREATE TABLE event_proposed_dates (
    id INT PRIMARY KEY AUTO_INCREMENT,
    event_id INT NOT NULL,
    proposed_date DATE NOT NULL,
    start_time TIME,
    end_time TIME,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);

-- User availability for proposed dates
CREATE TABLE user_availability (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    event_id INT NOT NULL,
    proposed_date_id INT NOT NULL,
    is_available BOOLEAN NOT NULL DEFAULT FALSE,
    response_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
    FOREIGN KEY (proposed_date_id) REFERENCES event_proposed_dates(id) ON DELETE CASCADE,
    UNIQUE KEY unique_user_date_response (user_id, proposed_date_id)
);

-- Event participants (who is invited to the event)
CREATE TABLE event_participants (
    id INT PRIMARY KEY AUTO_INCREMENT,
    event_id INT NOT NULL,
    user_id INT NOT NULL,
    invitation_status ENUM('pending', 'accepted', 'declined') DEFAULT 'pending',
    invited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    responded_at TIMESTAMP NULL,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY unique_event_participant (event_id, user_id)
);

-- Insert sample users
INSERT INTO users (name, email, phone) VALUES
('Alice Johnson', 'alice@example.com', '555-0101'),
('Bob Smith', 'bob@example.com', '555-0102'),
('Carol Davis', 'carol@example.com', '555-0103'),
('David Wilson', 'david@example.com', '555-0104'),
('Eva Brown', 'eva@example.com', '555-0105');

-- Create indexes for better performance
CREATE INDEX idx_events_date ON events(event_date);
CREATE INDEX idx_events_status ON events(status);
CREATE INDEX idx_availability_user ON user_availability(user_id);
CREATE INDEX idx_availability_event ON user_availability(event_id);
CREATE INDEX idx_participants_event ON event_participants(event_id);
CREATE INDEX idx_participants_user ON event_participants(user_id);