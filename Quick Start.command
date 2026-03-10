#!/bin/bash

# Quick Start Script for 4Demension
# This is the most reliable way to start the app

echo "🚀 Starting 4Demension..."

# Navigate to webapp directory
cd "/Users/macos/Desktop/Developer/4Demension/webapp"

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the server
echo "🌐 Starting web server..."
npm run dev

echo "✅ 4Demension stopped"
