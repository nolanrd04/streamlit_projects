"""
Script to generate page files for all projects
Run this once to create pages/X_ProjectN.py files
"""
from pathlib import Path

ROOT = Path(__file__).parent
PAGES_DIR = ROOT / "pages"
PAGES_DIR.mkdir(exist_ok=True)

# Project configurations
PROJECTS = [
    (1, "🎬", "Project1", "Sentiment Analysis"),
    (2, "📊", "Project2", "Data Visualization"),
    (3, "🔍", "Project3", "Text Classification"),
    (4, "🖼️", "Project4", "Image Processing"),
    (5, "📈", "Project5", "Time Series Analysis"),
    (6, "⭐", "Project6", "Recommendation System"),
    (7, "🎯", "Project7", "Clustering Analysis"),
    (8, "📉", "Project8", "Regression Models"),
    (9, "🧠", "Project9", "Neural Networks"),
    (10, "💬", "Project10", "NLP Pipeline"),
    (11, "🏷️", "NER", "Named Entity Recognition"),
]

TEMPLATE = '''"""
{project_name} - {description}
Auto-generated page that imports and runs the project
"""
import streamlit as st
import sys
from pathlib import Path
import importlib.util

# Configure page
st.set_page_config(
    page_title="{project_name}",
    page_icon="{icon}",
    layout="wide"
)

# Title
st.title("{icon} {project_name}")
st.markdown("---")

# Get project path
ROOT = Path(__file__).parent.parent
project_path = ROOT / "{project_folder}"
app_file = project_path / "app.py"

if not app_file.exists():
    st.error(f"""
    ❌ Project file not found: `{project_folder}/app.py`
    
    Please make sure the project folder and app.py exist.
    """)
    st.stop()

try:
    # Use importlib to load the module with a unique name to avoid caching issues
    spec = importlib.util.spec_from_file_location("{module_name}", app_file)
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
    ❌ Could not import `{project_folder}/app.py`
    
    **Error:** {{e}}
    
    **Checklist:**
    - [ ] Does `{project_folder}/app.py` exist?
    - [ ] Are all dependencies installed?
    - [ ] Check for syntax errors in the app.py file
    """)
    
except Exception as e:
    st.error(f"""
    ❌ Error running {project_name}
    
    **Error:** {{e}}
    
    **Tip:** Check the terminal for detailed error messages.
    """)
    
    # Show the full traceback for debugging
    import traceback
    with st.expander("🐛 Show full error details"):
        st.code(traceback.format_exc())
'''

def generate_page_files():
    """Generate all page files"""
    print("Generating page files...")
    
    for num, icon, folder, description in PROJECTS:
        filename = f"{num}_{icon}_{folder}.py"
        filepath = PAGES_DIR / filename
        
        # Create a unique module name for each project
        module_name = f"{folder.lower()}_app_module"
        
        content = TEMPLATE.format(
            project_name=folder,
            description=description,
            icon=icon,
            project_folder=folder,
            module_name=module_name
        )
        
        filepath.write_text(content, encoding='utf-8')
        print(f"✅ Created: {filename}")
    
    print(f"\n🎉 Done! Created {len(PROJECTS)} page files in pages/")
    print("\nRun your app with:")
    print("  streamlit run dashboard.py")

if __name__ == "__main__":
    generate_page_files()
