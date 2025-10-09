import requests
import time

def check_app_ready(url, timeout=30):
    """Check if a Streamlit app is ready by polling the URL."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    return False