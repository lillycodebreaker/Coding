#!/bin/bash

# Swimming Scheduler Startup Script

echo "🏊‍♀️ Starting Swimming Scheduler..."

# Check if MySQL is running
if ! pgrep -x "mysqld" > /dev/null; then
    echo "❌ MySQL is not running. Please start MySQL first."
    exit 1
fi

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null; then
        echo "⚠️  Port $1 is already in use"
        return 1
    fi
    return 0
}

# Check if ports are available
echo "🔍 Checking ports..."
check_port 5000 || { echo "Backend port 5000 is occupied"; exit 1; }
check_port 3000 || { echo "Frontend port 3000 is occupied"; exit 1; }

# Start backend
echo "🐍 Starting Python backend..."
cd backend
if [ ! -f ".env" ]; then
    echo "⚠️  Backend .env file not found. Creating from template..."
    cp .env .env.example 2>/dev/null || echo "Please create .env file manually"
fi

# Install Python dependencies if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate 2>/dev/null || echo "⚠️  Could not activate virtual environment"
pip install -r requirements.txt > /dev/null 2>&1

# Start backend in background
echo "🚀 Starting Flask server on port 5000..."
python app.py &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Start frontend
echo "🌐 Starting Node.js frontend..."
cd ../frontend

if [ ! -f ".env" ]; then
    echo "⚠️  Frontend .env file not found. Creating from template..."
    cp .env .env.example 2>/dev/null || echo "Please create .env file manually"
fi

# Install Node dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Start frontend in background
echo "🚀 Starting Express server on port 3000..."
npm start &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait for frontend to start
sleep 3

echo ""
echo "✅ Swimming Scheduler is now running!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:5000"
echo "📊 API Health: http://localhost:5000/api/health"
echo ""
echo "📝 To stop the services:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "💡 Tips:"
echo "   - First add some users at http://localhost:3000/users"
echo "   - Then create events at http://localhost:3000/create-event"
echo "   - Share event links with friends to set availability"
echo ""
echo "Press Ctrl+C to stop all services..."

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "👋 Swimming Scheduler stopped. Have a great day!"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup INT TERM

# Keep script running
wait