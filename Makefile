.PHONY: up down logs status

ENV_FILE   := .env
RUN_DIR    := .run
PID_FILE   := $(RUN_DIR)/api.pid
LOG_FILE   := $(RUN_DIR)/api.log
API_HOST   := 127.0.0.1
API_PORT   := 8000

# Starts Postgres+pgvector (docker compose), applies migrations, then starts
# the FastAPI app in the background. Safe to re-run: skips steps that are
# already satisfied instead of erroring.
up:
	@mkdir -p $(RUN_DIR)
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "No $(ENV_FILE) found — copying from .env.example."; \
		echo "Fill in OPENROUTER_API_KEY, FMP_API_KEY, VOYAGE_API_KEY before making a real forecast."; \
		cp .env.example $(ENV_FILE); \
	fi
	@echo "Starting Postgres+pgvector..."
	@docker compose up -d
	@echo "Waiting for Postgres to be ready..."
	@until docker compose exec -T postgres pg_isready -U equity_ensemble >/dev/null 2>&1; do sleep 1; done
	@echo "Applying migrations..."
	@uv run --env-file $(ENV_FILE) python -m equity_ensemble.persistence.migrate
	@if [ -f $(PID_FILE) ] && kill -0 "$$(cat $(PID_FILE))" 2>/dev/null; then \
		echo "API already running (pid $$(cat $(PID_FILE))) at http://$(API_HOST):$(API_PORT)"; \
	else \
		echo "Starting API at http://$(API_HOST):$(API_PORT) (logs: $(LOG_FILE))..."; \
		nohup uv run --env-file $(ENV_FILE) uvicorn equity_ensemble.api.main:app \
			--host $(API_HOST) --port $(API_PORT) \
			> $(LOG_FILE) 2>&1 < /dev/null & echo $$! > $(PID_FILE); \
		sleep 1; \
		if kill -0 "$$(cat $(PID_FILE))" 2>/dev/null; then \
			echo "API started (pid $$(cat $(PID_FILE)))."; \
		else \
			echo "API failed to start — check $(LOG_FILE)"; \
			rm -f $(PID_FILE); \
			exit 1; \
		fi; \
	fi
	@echo ""
	@echo "Up. Try:"
	@echo "  curl -X POST http://$(API_HOST):$(API_PORT)/forecast -H 'content-type: application/json' -d '{\"ticker\": \"AAPL\"}'"
	@echo "  make logs    # tail the API log"
	@echo "  make down    # stop everything"

# Stops the API process (if running) and the docker containers. Leaves the
# Postgres data volume intact — use `docker compose down -v` to wipe it.
down:
	@if [ -f $(PID_FILE) ]; then \
		PID=$$(cat $(PID_FILE)); \
		if kill -0 "$$PID" 2>/dev/null; then \
			echo "Stopping API (pid $$PID)..."; \
			kill "$$PID" 2>/dev/null || true; \
			for i in 1 2 3 4 5; do kill -0 "$$PID" 2>/dev/null || break; sleep 1; done; \
			pkill -9 -P "$$PID" 2>/dev/null || true; \
			kill -9 "$$PID" 2>/dev/null || true; \
		fi; \
		rm -f $(PID_FILE); \
	else \
		echo "No API pid file — nothing to stop."; \
	fi
	@echo "Stopping Postgres..."
	@docker compose down
	@echo "Down."

# Tails the backgrounded API's log file.
logs:
	@tail -f $(LOG_FILE)

# Quick status check for both the API process and the docker containers.
status:
	@if [ -f $(PID_FILE) ] && kill -0 "$$(cat $(PID_FILE))" 2>/dev/null; then \
		echo "API: running (pid $$(cat $(PID_FILE))) at http://$(API_HOST):$(API_PORT)"; \
	else \
		echo "API: not running"; \
	fi
	@docker compose ps
