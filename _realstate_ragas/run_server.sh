#!/usr/bin/env bash

# Este archivo se usa para iniciar el servidor cuando se ejecutan las pruebas de Github Actions.
# En caso de que su proyecto necesite otros comandos pueden modificar este archivo

echo "Starting Uvicorn server..."
SERVER_READY="false"
uvicorn main:app --host 0.0.0.0 --port 8000 &

# === ESPERAR A QUE EL SERVIDOR ESTÉ LISTO ===
echo "Waiting for server to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/api/v1/health > /dev/null; then
        echo "✅ Server is running and responding"
        SERVER_READY="true"
        break
    fi
    echo "Attempt $i/15: Server not ready yet..."
    sleep 2
done

if [ "$SERVER_READY" != "true" ]; then
    echo "❌ Server failed to start within 60 seconds"
    echo "Server logs:"
    ps aux | grep uvicorn || echo "No uvicorn process found"
fi
