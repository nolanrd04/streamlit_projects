"""
Project5 - Time Series Analysis
Auto-generated page that imports and runs the project
"""
import streamlit as st
import sys
from pathlib import Path
import importlib.util

# Configure page
st.set_page_config(
    page_title="Project5",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Project5")
st.markdown("---")

# Get project path
ROOT = Path(__file__).parent.parent
project_path = ROOT / "Project5"
app_file = project_path / "app.py"

if not app_file.exists():
    st.error(f"""
    ❌ Project file not found: `Project5/app.py`
    
    Please make sure the project folder and app.py exist.
    """)
    st.stop()

try:
    # Use importlib to load the module with a unique name to avoid caching issues
    spec = importlib.util.spec_from_file_location("project5_app_module", app_file)
    project_module = importlib.util.module_from_spec(spec)
    
    # Temporarily add project path to sys.path for relative imports
    original_path = sys.path.copy()
    sys.path.insert(0, str(project_path))
    
    try:
        # Execute the module
        spec.loader.exec_module(project_module)
        
        # If the app has a main() function, call it
        if hasattr(project_module, 'main'):
            project_module.main()
            
    finally:
        # Restore original sys.path
        sys.path = original_path
    
except ImportError as e:
    st.error(f"""
    ❌ Could not import `Project5/app.py`
    
    **Error:** {e}
    
    **Checklist:**
    - [ ] Does `Project5/app.py` exist?
    - [ ] Are all dependencies installed?
    - [ ] Check for syntax errors in the app.py file
    """)
    
except Exception as e:
    st.error(f"""
    ❌ Error running Project5
    
    **Error:** {e}
    
    **Tip:** Check the terminal for detailed error messages.
    """)
    
    # Show the full traceback for debugging
    import traceback
    with st.expander("🐛 Show full error details"):
        st.code(traceback.format_exc())
