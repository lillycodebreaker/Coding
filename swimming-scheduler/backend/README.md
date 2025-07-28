# Swimming Scheduler Backend

Python Flask REST API for the Swimming Scheduler application.

## Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure Database**
```bash
cp .env.example .env
# Edit .env with your MySQL credentials
```

3. **Setup Database**
```bash
mysql -u root -p < ../database/schema.sql
```

4. **Run the Server**
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Documentation

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### Users
- `GET /users` - Get all users
- `POST /users` - Create new user
- `GET /users/{id}` - Get user by ID
- `PUT /users/{id}` - Update user
- `DELETE /users/{id}` - Delete user

#### Events
- `POST /events` - Create new event with proposed dates
- `GET /events/{id}` - Get event details with proposed dates
- `GET /users/{userId}/events` - Get events for a user (as participant)
- `GET /users/{userId}/created-events` - Get events created by a user

#### Availability
- `GET /events/{id}/availability-report` - Get availability report
- `POST /events/{id}/availability` - Update user availability
- `GET /events/{id}/users/{userId}/availability` - Get user's availability

### Example Requests

#### Create User
```bash
curl -X POST http://localhost:5000/api/users \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe", "email": "john@example.com", "phone": "555-0123"}'
```

#### Create Event
```bash
curl -X POST http://localhost:5000/api/events \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Weekend Pool Session",
    "description": "Fun swimming session",
    "location": "Community Pool",
    "created_by": 1,
    "proposed_dates": [
      {"date": "2024-01-15", "start_time": "10:00", "end_time": "12:00"},
      {"date": "2024-01-16", "start_time": "14:00", "end_time": "16:00"}
    ],
    "participant_ids": [1, 2, 3]
  }'
```

#### Update Availability
```bash
curl -X POST http://localhost:5000/api/events/1/availability \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "proposed_date_id": 1,
    "is_available": true,
    "notes": "Looking forward to it!"
  }'
```

## Development

### Project Structure
```
backend/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
├── config/
│   └── database.py    # Database configuration
├── models/
│   ├── user.py        # User model
│   └── event.py       # Event model
└── routes/
    ├── users.py       # User routes
    └── events.py      # Event routes
```

### Environment Variables
```bash
DB_HOST=localhost
DB_PORT=3306
DB_NAME=swimming_scheduler
DB_USER=root
DB_PASSWORD=your_password
FLASK_ENV=development
FLASK_DEBUG=True
```

### Error Handling
All endpoints return consistent JSON responses:
```json
{
  "success": true,
  "data": {...}
}
```

Or for errors:
```json
{
  "success": false,
  "error": "Error message"
}
```