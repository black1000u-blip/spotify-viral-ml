"""
colab_run.py — Run this single file in Google Colab to launch the dashboard.

Usage (in a Colab cell):
    !pip install -q pyngrok
    exec(open('colab_run.py').read())

Or just paste the contents of this file directly into a Colab cell and run it.
"""

import os
import subprocess
import threading
import time

# ── 1. Install dependencies ────────────────────────────────────────────────────
print("📦 Installing dependencies...")
os.system("pip install -q -r requirements.txt")
os.system("pip install -q pyngrok")
print("✅ Dependencies installed.\n")

# ── 2. Start ngrok tunnel ──────────────────────────────────────────────────────
from pyngrok import ngrok, conf

NGROK_TOKEN = "3AbFUwuUiWdVY3KG97gvOnqiR18_7S56jZJzefkv21hPXGqar"
conf.get_default().auth_token = NGROK_TOKEN

# Kill any existing tunnels
ngrok.kill()
time.sleep(1)

tunnel = ngrok.connect(8501, bind_tls=True)
print("=" * 60)
print(f"🌐  DASHBOARD URL:  {tunnel.public_url}")
print("=" * 60)
print("Open the URL above in your browser.")
print("Keep this cell running — closing it stops the server.\n")

# ── 3. Launch Streamlit ────────────────────────────────────────────────────────
print("🚀 Starting Streamlit server on port 8501...")
os.system(
    "python -m streamlit run frontend/app.py "
    "--server.port 8501 "
    "--server.headless true "
    "--server.enableCORS false "
    "--server.enableXsrfProtection false"
)
