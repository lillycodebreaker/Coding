const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

// Set EJS as view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'src'));

// API base URL
const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:5000/api';

// Routes
app.get('/', (req, res) => {
    res.render('pages/index', { 
        title: 'Swimming Scheduler',
        apiUrl: API_BASE_URL
    });
});

app.get('/events', (req, res) => {
    res.render('pages/events', { 
        title: 'Events - Swimming Scheduler',
        apiUrl: API_BASE_URL
    });
});

app.get('/create-event', (req, res) => {
    res.render('pages/create-event', { 
        title: 'Create Event - Swimming Scheduler',
        apiUrl: API_BASE_URL
    });
});

app.get('/event/:id', (req, res) => {
    res.render('pages/event-detail', { 
        title: 'Event Details - Swimming Scheduler',
        apiUrl: API_BASE_URL,
        eventId: req.params.id
    });
});

app.get('/availability/:eventId/:userId', (req, res) => {
    res.render('pages/availability', { 
        title: 'Set Availability - Swimming Scheduler',
        apiUrl: API_BASE_URL,
        eventId: req.params.eventId,
        userId: req.params.userId
    });
});

app.get('/users', (req, res) => {
    res.render('pages/users', { 
        title: 'Users - Swimming Scheduler',
        apiUrl: API_BASE_URL
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`Swimming Scheduler Frontend running on http://localhost:${PORT}`);
    console.log(`API Backend: ${API_BASE_URL}`);
});