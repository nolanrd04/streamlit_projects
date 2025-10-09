import os
import subprocess
import sys
from pathlib import Path
import time
import requests

import streamlit as st
from utils import check_app_ready

ROOT = Path(__file__).parent
PROJECTS = [f"Project{i}" for i in range(1, 11)]

BASE_PORT = 8601  # avoid clashing with the dashboard's own port

# Cloud mode: if APP_URLS provided in secrets, we link to deployed apps instead of spawning processes
try:
    APP_URLS = dict(st.secrets.get("APP_URLS", {}))
except Exception:
    APP_URLS = {}
CLOUD_MODE = bool(APP_URLS)

# Only add AIT-204-NLP in local mode (too large for cloud)
if not CLOUD_MODE:
    PROJECTS.append("AIT-204-NLP")

if "processes" not in st.session_state:
    st.session_state.processes = {}

st.set_page_config(page_title="Apps Dashboard", page_icon="🗂️", layout="centered")
st.title("Apps Dashboard")

if CLOUD_MODE:
    st.caption("Cloud mode: open deployed apps via links.")
    for name in PROJECTS:
        url = APP_URLS.get(name)
        col1, col2 = st.columns([1, 2])
        with col1:
            if url:
                if hasattr(st, "link_button"):
                    st.link_button(name, url)
                else:
                    st.markdown(f"[{name}]({url})")
            else:
                st.button(name, disabled=True)
        with col2:
            if url:
                st.write(url)
            else:
                st.warning("URL not configured in secrets.")
    st.divider()
    st.caption("Configure [APP_URLS] in Streamlit secrets to enable links.")
else:
    st.caption("Local mode: launch standalone apps on their own ports.")

    python_cmd = sys.executable or "python3"
    st.write(f"WARNING: Projects take a long time to load. If you run into a connection issue, wait a little longer and try again.")

    for idx, name in enumerate(PROJECTS, start=0):
        port = BASE_PORT + idx
        app_path = ROOT / name / "app.py"

        if not app_path.exists():
            continue

        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button(name, key=f"btn-{name}"):
                proc = st.session_state.processes.get(name)
                if proc is None or proc.poll() is not None:
                    # Convert to absolute path
                    abs_app_path = os.path.abspath(str(app_path))
                    
                    cmd = [
                        python_cmd,
                        "-m",
                        "streamlit",
                        "run",
                        abs_app_path,
                        "--server.port",
                        str(port),
                        "--server.headless",
                        "true",
                        "--server.address",
                        "localhost",
                    ]
                    env = os.environ.copy()
                    env.setdefault("PYTHONUNBUFFERED", "1")
                    # Use the root directory as working directory
                    work_dir = str(ROOT)
                    
                    st.session_state.processes[name] = subprocess.Popen(
                        cmd,
                        cwd=work_dir,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    # Create placeholder for status message
                    status_msg = st.empty()
                    status_msg.info(f"Starting {name}...")

                    # Start checking if app is ready
                    url = f"http://localhost:{port}"
                    
                    # Check for immediate startup errors
                    time.sleep(2)  # Give it a moment to start
                    
                    if st.session_state.processes[name].poll() is not None:
                        # Process failed to start - clean up
                        status_msg.error(f"{name} failed to start. Cleaning up and retrying...")
                        
                        # Remove failed process from session state
                        if name in st.session_state.processes:
                            del st.session_state.processes[name]
                        
                        # Give user option to retry
                        status_msg.warning(f"⚠️ {name} failed to start. Check the terminal for errors, then click the button again to retry.")
                        st.stop()  # Stop execution to allow retry
                        
                    # Wait for app to be ready
                    ready = check_app_ready(url)
                    
                    # Double-check process is still running after ready check
                    if st.session_state.processes[name].poll() is not None:
                        status_msg.error(f"⚠️ {name} started but then crashed. Check the terminal for errors.")
                        if name in st.session_state.processes:
                            del st.session_state.processes[name]
                        st.stop()
                    
                    # Show appropriate status message
                    if ready:
                        status_msg.success(f"{name} is ready! Click the link to open →")
                    else:
                        status_msg.warning(f"{name} is taking longer than usual to start. The link will work once it's ready.")
        with col2:
            url = f"http://localhost:{port}"
            st.markdown(f"[Open {name}]({url})")

    st.divider()
    st.caption("Tip: Each project is a standalone Streamlit app inside its folder.")
