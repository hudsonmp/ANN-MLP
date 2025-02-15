#!/bin/bash

# Start Django backend
echo "Starting Django backend..."
python manage.py runserver &

# Start Next.js frontend
echo "Starting Next.js frontend..."
cd frontend && npm run dev 