.PHONY: up down logs logs-web status

ENV_FILE     := .env
RUN_DIR      := .run
PID_FILE     := $(RUN_DIR)/api.pid
LOG_FILE     := $(RUN_DIR)/api.log
API_HOST     := 127.0.0.1
API_PORT     := 8000

WEB_DIR      := web
WEB_ENV_FILE := $(WEB_DIR)/.env.local
WEB_PID_FILE := $(RUN_DIR)/web.pid
WEB_LOG_FILE := $(RUN_DIR)/web.log
WEB_PORT     := 3000

# Starts Postgres+pgvector (docker compose), applies migrations, starts the
# FastAPI app, then starts the Next.js web UI — all in the background. Safe
# to re-run: skips steps that are already satisfied instead of erroring.
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
	@if [ ! -d $(WEB_DIR)/node_modules ]; then \
		echo "Installing web/ dependencies (first run)..."; \
		npm --prefix $(WEB_DIR) install; \
	fi
	@if [ ! -f $(WEB_ENV_FILE) ]; then \
		echo "No $(WEB_ENV_FILE) found — copying from .env.local.example."; \
		cp $(WEB_DIR)/.env.local.example $(WEB_ENV_FILE); \
	fi
	@if [ -f $(WEB_PID_FILE) ] && kill -0 "$$(cat $(WEB_PID_FILE))" 2>/dev/null; then \
		echo "Web UI already running (pid $$(cat $(WEB_PID_FILE))) at http://localhost:$(WEB_PORT)"; \
	else \
		echo "Starting web UI at http://localhost:$(WEB_PORT) (logs: $(WEB_LOG_FILE))..."; \
		nohup npm --prefix $(WEB_DIR) run dev -- --port $(WEB_PORT) \
			> $(WEB_LOG_FILE) 2>&1 < /dev/null & echo $$! > $(WEB_PID_FILE); \
		sleep 2; \
		if kill -0 "$$(cat $(WEB_PID_FILE))" 2>/dev/null; then \
			echo "Web UI started (pid $$(cat $(WEB_PID_FILE)))."; \
		else \
			echo "Web UI failed to start — check $(WEB_LOG_FILE)"; \
			rm -f $(WEB_PID_FILE); \
			exit 1; \
		fi; \
	fi
	@echo ""
	@echo "Up. Try:"
	@echo "  open http://localhost:$(WEB_PORT)"
	@echo "  curl -X POST http://$(API_HOST):$(API_PORT)/forecast -H 'content-type: application/json' -d '{\"ticker\": \"AAPL\"}'"
	@echo "  make logs      # tail the API log"
	@echo "  make logs-web  # tail the web UI log"
	@echo "  make down      # stop everything"

# Stops the web UI, the API process, and the docker containers. Leaves the
# Postgres data volume intact — use `docker compose down -v` to wipe it.
down:
	@if [ -f $(WEB_PID_FILE) ]; then \
		PID=$$(cat $(WEB_PID_FILE)); \
		if kill -0 "$$PID" 2>/dev/null; then \
			echo "Stopping web UI (pid $$PID)..."; \
			pkill -P "$$PID" 2>/dev/null || true; \
			kill "$$PID" 2>/dev/null || true; \
			for i in 1 2 3 4 5; do kill -0 "$$PID" 2>/dev/null || break; sleep 1; done; \
			pkill -9 -P "$$PID" 2>/dev/null || true; \
			kill -9 "$$PID" 2>/dev/null || true; \
		fi; \
		rm -f $(WEB_PID_FILE); \
	else \
		echo "No web UI pid file — nothing to stop."; \
	fi
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

# Tails the backgrounded web UI's log file.
logs-web:
	@tail -f $(WEB_LOG_FILE)

# Quick status check for the web UI, the API process, and the docker containers.
status:
	@if [ -f $(WEB_PID_FILE) ] && kill -0 "$$(cat $(WEB_PID_FILE))" 2>/dev/null; then \
		echo "Web UI: running (pid $$(cat $(WEB_PID_FILE))) at http://localhost:$(WEB_PORT)"; \
	else \
		echo "Web UI: not running"; \
	fi
	@if [ -f $(PID_FILE) ] && kill -0 "$$(cat $(PID_FILE))" 2>/dev/null; then \
		echo "API: running (pid $$(cat $(PID_FILE))) at http://$(API_HOST):$(API_PORT)"; \
	else \
		echo "API: not running"; \
	fi
	@docker compose ps
