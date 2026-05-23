#!/usr/bin/env bash
# TurboQuant Full Stack Launcher — starts API + Dashboard + TQ Pipeline
# Usage: ./bin/tq_stack.sh [start|stop|restart|status]

set -e

API_PORT=8765
DASH_PORT=8080
LOG_DIR="/data/data/com.termux/files/home/corpus-control-plane/logs"
PID_DIR="/data/data/com.termux/files/home/corpus-control-plane/.pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

API_PID_FILE="$PID_DIR/tq_api.pid"
DASH_PID_FILE="$PID_DIR/tq_dash.pid"

log() { echo "[TQ-$(date +%H:%M:%S)] $1"; }

start_api() {
  if [ -f "$API_PID_FILE" ] && kill -0 "$(cat $API_PID_FILE)" 2>/dev/null; then
    log "API already running (PID $(cat $API_PID_FILE))"
    return 1
  fi
  log "Starting TQ API on port $API_PORT..."
  cd /data/data/com.termux/files/home/corpus-control-plane
  python3 bin/turboquant_backend_api.py >> "$LOG_DIR/tq_api.log" 2>&1 &
  echo $! > "$API_PID_FILE"
  sleep 2
  if kill -0 "$(cat $API_PID_FILE)" 2>/dev/null; then
    log "API started (PID $(cat $API_PID_FILE))"
  else
    log "API failed to start - check $LOG_DIR/tq_api.log"
  fi
}

start_dash() {
  if [ -f "$DASH_PID_FILE" ] && kill -0 "$(cat $DASH_PID_FILE)" 2>/dev/null; then
    log "Dashboard already running (PID $(cat $DASH_PID_FILE))"
    return 1
  fi
  log "Starting TQ Dashboard on port $DASH_PORT..."
  cd /data/data/com.termux/files/home/corpus-control-plane
  python3 bin/turboquant_dashboard_server.py >> "$LOG_DIR/tq_dash.log" 2>&1 &
  echo $! > "$DASH_PID_FILE"
  sleep 1
  if kill -0 "$(cat $DASH_PID_FILE)" 2>/dev/null; then
    log "Dashboard started (PID $(cat $DASH_PID_FILE))"
  else
    log "Dashboard failed to start - check $LOG_DIR/tq_dash.log"
  fi
}

stop_api() {
  if [ -f "$API_PID_FILE" ]; then
    PID=$(cat $API_PID_FILE)
    if kill -0 "$PID" 2>/dev/null; then
      log "Stopping API (PID $PID)..."
      kill "$PID" && rm "$API_PID_FILE" && log "API stopped"
    else
      rm "$API_PID_FILE"
    fi
  fi
}

stop_dash() {
  if [ -f "$DASH_PID_FILE" ]; then
    PID=$(cat $DASH_PID_FILE)
    if kill -0 "$PID" 2>/dev/null; then
      log "Stopping Dashboard (PID $PID)..."
      kill "$PID" && rm "$DASH_PID_FILE" && log "Dashboard stopped"
    else
      rm "$DASH_PID_FILE"
    fi
  fi
}

status() {
  echo "=== TQ Stack Status ==="
  echo "API: $([ -f $API_PID_FILE ] && kill -0 "$(cat $API_PID_FILE)" 2>/dev/null && echo "RUNNING ($(cat $API_PID_FILE))" || echo "STOPPED")"
  echo "Dashboard: $([ -f $DASH_PID_FILE ] && kill -0 "$(cat $DASH_PID_FILE)" 2>/dev/null && echo "RUNNING ($(cat $DASH_PID_FILE))" || echo "STOPPED")"
  echo ""
  echo "Ports:"
  echo "  API:    http://localhost:$API_PORT"
  echo "  Dashboard: http://localhost:$DASH_PORT"
  echo ""
  echo "Logs: $LOG_DIR/"
}

case "${1:-start}" in
  start)
    start_api
    start_dash
    status
    ;;
  stop)
    stop_dash
    stop_api
    log "All stopped"
    ;;
  restart)
    stop_dash; stop_api
    start_api
    start_dash
    status
    ;;
  status)
    status
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac