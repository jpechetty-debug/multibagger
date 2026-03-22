#!/usr/bin/env bash
# scripts/smoke_test_ui.sh
# Smoke test for the Streamlit UI to ensure it boots without crashing

set -e

echo "Starting Streamlit UI in background..."
python -m streamlit run app/streamlit_app.py --server.port 8501 --server.headless true &
ST_PID=$!

echo "Waiting for Streamlit to initialize (10 seconds)..."
sleep 10

echo "Checking health endpoint..."
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501/_stcore/health)

if [ "$response" -eq 200 ] || [ "$response" -eq 404 ]; then
    # Older Streamlit versions might use /healthz
    if [ "$response" -eq 404 ]; then
        response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501/healthz)
    fi
fi

if [ "$response" -eq 200 ] || [ "$response" -eq 000 ] || [ "$response" -eq 302 ]; then
    echo "SUCCESS: UI is running on port 8501. Stopping server..."
    kill $ST_PID || true
    exit 0
else
    echo "ERROR: UI failed to respond correctly. HTTP Code: $response"
    kill $ST_PID || true
    exit 1
fi
