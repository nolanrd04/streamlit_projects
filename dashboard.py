"""
Multi-Project Streamlit Dashboard
All projects run in the same Streamlit app - just switch pages in the sidebar!
"""
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent

st.set_page_config(
    page_title="Multi-Project Dashboard",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🗂️ Multi-Project Dashboard")
st.markdown("---")

st.markdown("""
## Welcome! 👋

This dashboard contains **10+ projects** all running in a single Streamlit app.

### How to Use:
1. **Open the sidebar** (←) to see all projects
2. **Click any project** to view it
3. **Switch between projects** instantly - no waiting!

### 📂 Available Projects:
    Access them through the sidebar.
""")

# Show project list
projects = [
    ("Project 1", "none"),
    ("Project 2", "ANN: Basketball Team Maker"),
    ("Project 3", "CNN: Image Prediction"),
    ("Project 4", "NLP: Sentiment Analysis"),
    ("Project 5", "none"),
    ("Project 6", "none"),
    ("Project 7", "none"),
    ("Project 8", "none"),
    ("Project 9", "none"),
    ("Project 10", "none"),
    ("NER", "Named Entity Recognition (Nolan's personal)"),
]

cols = st.columns(3)
for idx, (name, desc) in enumerate(projects):
    with cols[idx % 3]:
        st.info(f"**{name}**\n\n{desc}")

st.markdown("---")
st.caption("💡 Tip: Use the sidebar to navigate between projects. Each page is self-contained!")

st.markdown("""
### 🚀 Quick Links:
- All projects are in the **sidebar** ←
- Each project has its own **dedicated page**
- **No loading times** - everything runs instantly!
""")
