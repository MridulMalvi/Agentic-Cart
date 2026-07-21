"""
scripts/start_dev.py

Single-command dev launcher for the AI Shopping Assistant.

Starts (in order):
  1. Flask API server (port 5001)
  2. Streamlit dashboard (port 8501)

Handles Ctrl+C gracefully — kills both processes cleanly.

Usage:
    python scripts/start_dev.py

Requirements: Both processes must be started from the project root.
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

processes = []


def stop_all(signum=None, frame=None):
    print("\n\n[Launcher] Shutting down...")
    for name, proc in processes:
        try:
            proc.terminate()
            print(f"[Launcher] Stopped {name} (PID {proc.pid})")
        except Exception:
            pass
    # Give processes time to exit cleanly
    time.sleep(1)
    for name, proc in processes:
        try:
            proc.kill()
        except Exception:
            pass
    print("[Launcher] All processes stopped. Goodbye!")
    sys.exit(0)


signal.signal(signal.SIGINT,  stop_all)
signal.signal(signal.SIGTERM, stop_all)


def start_flask():
    print("[Launcher] Starting Flask API on port 5001...")
    proc = subprocess.Popen(
        [PYTHON, str(ROOT / "api.py")],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    processes.append(("Flask API", proc))
    return proc


def start_streamlit():
    print("[Launcher] Starting Streamlit dashboard on port 8501...")
    proc = subprocess.Popen(
        [
            PYTHON, "-m", "streamlit", "run",
            str(ROOT / "app.py"),
            "--server.port", "8501",
            "--server.headless", "true",
            "--server.fileWatcherType", "none",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    processes.append(("Streamlit", proc))
    return proc


def tail_output(proc, name: str):
    """Stream a process's stdout to the terminal with a prefix."""
    import threading

    def _read():
        for line in proc.stdout:
            print(f"[{name}] {line}", end="")

    t = threading.Thread(target=_read, daemon=True)
    t.start()


def main():
    print("=" * 60)
    print("  AI Shopping Assistant — Dev Launcher")
    print("  AMD Ryzen AI 7 350 — NLP workers: 6 processes")
    print("=" * 60)

    # Check .env exists
    env_file = ROOT / ".env"
    if not env_file.exists():
        print("\n[ERROR] .env file not found!")
        print(f"  Run: copy {ROOT}\\.env.template {ROOT}\\.env")
        print("  Then fill in your API keys and DB credentials.\n")
        sys.exit(1)

    # Start Flask
    flask_proc = start_flask()
    tail_output(flask_proc, "Flask")
    time.sleep(2)  # Give Flask time to bind

    # Check Flask started OK
    if flask_proc.poll() is not None:
        print("[ERROR] Flask failed to start. Check your .env and DB connections.")
        stop_all()

    # Start Streamlit
    st_proc = start_streamlit()
    tail_output(st_proc, "Streamlit")
    time.sleep(2)

    print("\n" + "=" * 60)
    print("  [OK] Both services are running!")
    print()
    print("  Dashboard    -> http://localhost:8501")
    print("  Flask API    -> http://localhost:5001/api/health")
    print()
    print("  Press Ctrl+C to stop both servers")
    print("=" * 60 + "\n")

    # Keep alive — wait for either process to die
    while True:
        time.sleep(2)
        for name, proc in processes:
            if proc.poll() is not None:
                print(f"\n[Launcher] {name} exited unexpectedly (code {proc.returncode}).")
                stop_all()


if __name__ == "__main__":
    main()
