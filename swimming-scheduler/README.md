# Swimming Scheduler

A comprehensive web application for scheduling swimming events with friends. Built with Node.js frontend, Python Flask backend, and MySQL database.

## Features

- 🏊‍♀️ **Event Creation**: Create swimming events with multiple proposed dates and times
- 👥 **User Management**: Add and manage friends who will participate in events
- 📅 **Availability Tracking**: Users can mark their availability for proposed dates
- 📊 **Smart Reports**: View availability reports to find the best time for everyone
- 📱 **Responsive Design**: Beautiful, mobile-friendly interface
- 🔄 **Real-time Updates**: See availability changes as they happen

## Architecture

- **Frontend**: Node.js with Express and EJS templates
- **Backend**: Python Flask with RESTful API
- **Database**: MySQL with comprehensive schema
- **Styling**: Bootstrap 5 with custom CSS

## Project Structure

```
swimming-scheduler/
├── frontend/           # Node.js frontend application
│   ├── server.js      # Express server
│   ├── package.json   # Node.js dependencies
│   ├── src/           # EJS templates
│   └── public/        # Static assets (CSS, JS)
├── backend/           # Python Flask backend
│   ├── app.py         # Main Flask application
│   ├── requirements.txt
│   ├── models/        # Database models
│   ├── routes/        # API routes
│   └── config/        # Database configuration
└── database/          # Database schema and queries
    ├── schema.sql     # Database creation script
    └── queries.sql    # Common SQL queries
```

## Quick Start

### Prerequisites

- Node.js (v14 or higher)
- Python 3.8+
- MySQL 8.0+

### 1. Database Setup

```bash
# Start MySQL and create the database
mysql -u root -p < database/schema.sql
```

### 2. Backend Setup (Python Flask)

```bash
cd backend
pip install -r requirements.txt

# Configure database connection in .env file
cp .env.example .env
# Edit .env with your MySQL credentials

# Start the backend server
python app.py
```

The backend will run on `http://localhost:5000`

### 3. Frontend Setup (Node.js)

```bash
cd frontend
npm install

# Configure API connection
cp .env.example .env
# Edit .env if needed (default points to localhost:5000)

# Start the frontend server
npm start
```

The frontend will run on `http://localhost:3000`

## Usage

### 1. Add Users
1. Go to **Users** page
2. Add friends who will participate in swimming events
3. Each user needs a name and email (phone is optional)

### 2. Create Events
1. Go to **Create Event** page
2. Fill in event details (title, location, description)
3. Add multiple proposed dates and times
4. Select participants from your friends list
5. Submit to create the event

### 3. Set Availability
1. Share the event link with participants
2. Each person visits the event page
3. Click "Set Availability" and select their user
4. Mark availability for each proposed date
5. Add optional notes for each date

### 4. View Reports
1. Go to the event details page
2. View the availability report showing:
   - How many people are available for each date
   - Percentage availability
   - Names of available/unavailable participants
3. Use this data to choose the best date for everyone

## API Endpoints

### Users
- `GET /api/users` - Get all users
- `POST /api/users` - Create new user
- `GET /api/users/{id}` - Get user by ID
- `PUT /api/users/{id}` - Update user
- `DELETE /api/users/{id}` - Delete user

### Events
- `POST /api/events` - Create new event
- `GET /api/events/{id}` - Get event details
- `GET /api/events/{id}/availability-report` - Get availability report
- `POST /api/events/{id}/availability` - Update user availability
- `GET /api/events/{id}/users/{userId}/availability` - Get user's availability

## Database Schema

The application uses a normalized MySQL schema with the following main tables:

- **users**: Friend information
- **events**: Swimming event details
- **event_proposed_dates**: Multiple date options for each event
- **event_participants**: Who is invited to each event
- **user_availability**: Individual availability responses

## Configuration

### Backend (.env)
```
DB_HOST=localhost
DB_PORT=3306
DB_NAME=swimming_scheduler
DB_USER=root
DB_PASSWORD=your_password
FLASK_ENV=development
```

### Frontend (.env)
```
PORT=3000
API_BASE_URL=http://localhost:5000/api
```

## Development

### Backend Development
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev  # Uses nodemon for auto-restart
```

## Production Deployment

### Backend
- Use a production WSGI server like Gunicorn
- Set up proper environment variables
- Configure MySQL with production settings

### Frontend
- Set `NODE_ENV=production`
- Use PM2 or similar for process management
- Configure reverse proxy (nginx)

### Database
- Use MySQL 8.0 in production
- Set up proper backup procedures
- Configure replication if needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For questions or issues, please create an issue in the repository or contact the development team.

---

Happy swimming! 🏊‍♀️🏊‍♂️