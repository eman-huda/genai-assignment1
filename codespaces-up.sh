#!/bin/sh
# Start the app inside GitHub Codespaces (see docker-compose.codespaces.yml).
docker compose -f docker-compose.yml -f docker-compose.codespaces.yml up "$@"
