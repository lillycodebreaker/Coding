# Swimming Scheduler Frontend

Node.js frontend application with Express server and EJS templates.

## Setup

1. **Install Dependencies**
```bash
npm install
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env if needed (default settings should work)
```

3. **Start the Server**
```bash
npm start
```

For development with auto-restart:
```bash
npm run dev
```

The application will be available at `http://localhost:3000`

## Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Modern UI**: Built with Bootstrap 5 and custom CSS
- **Interactive Forms**: Dynamic form controls and validation
- **Real-time Updates**: Automatic data refresh and notifications
- **User-friendly**: Intuitive navigation and clear feedback

## Pages

### Home (`/`)
- Dashboard with statistics
- Quick access to main features
- Recent events overview

### Users (`/users`)
- Manage friend list
- Add new users
- Edit/delete existing users

### Events (`/events`)
- View all events
- Filter by user or status
- Quick access to create new events

### Create Event (`/create-event`)
- Interactive form for new events
- Dynamic date/time inputs
- User selection with checkboxes

### Event Details (`/event/{id}`)
- Complete event information
- Availability report with visual indicators
- Quick actions for setting availability

### Set Availability (`/availability/{eventId}/{userId}`)
- User-friendly availability form
- Radio button selection
- Optional notes for each date

## Technical Details

### Project Structure
```
frontend/
├── server.js          # Express server
├── package.json       # Dependencies and scripts
├── .env              # Environment configuration
├── src/
│   ├── components/
│   │   └── layout.ejs    # Main layout template
│   └── pages/
│       ├── index.ejs           # Home page
│       ├── users.ejs           # User management
│       ├── events.ejs          # Events list
│       ├── create-event.ejs    # Create event form
│       ├── event-detail.ejs    # Event details
│       └── availability.ejs    # Set availability
└── public/
    └── css/
        └── style.css   # Custom styles
```

### Dependencies
- **express**: Web server framework
- **ejs**: Template engine
- **axios**: HTTP client for API calls
- **body-parser**: Request parsing middleware
- **cors**: Cross-origin request handling

### API Integration
The frontend communicates with the Python backend via REST API:
- Base URL: `http://localhost:5000/api`
- All requests use Axios for HTTP communication
- Consistent error handling and user feedback

### Styling
- **Bootstrap 5**: Responsive component framework
- **Font Awesome**: Icon library
- **Custom CSS**: Enhanced styling and animations
- **CSS Variables**: Consistent color scheme

### JavaScript Features
- **ES6+**: Modern JavaScript syntax
- **Async/Await**: Clean asynchronous code
- **Error Handling**: Comprehensive error catching
- **Dynamic Content**: Client-side DOM manipulation
- **Form Validation**: Real-time input validation

## Development

### Environment Variables
```bash
PORT=3000
API_BASE_URL=http://localhost:5000/api
```

### Scripts
```bash
npm start      # Start production server
npm run dev    # Start with nodemon (auto-restart)
```

### Adding New Pages
1. Create EJS template in `src/pages/`
2. Add route in `server.js`
3. Include necessary JavaScript for API calls
4. Test functionality and responsive design

### Customizing Styles
Edit `public/css/style.css` to modify:
- Color scheme (CSS variables)
- Component styling
- Responsive breakpoints
- Animations and transitions