# Streamlit Apps Dashboard

A simple dashboard that launches 10 standalone Streamlit apps, each living in its own folder.

- Dashboard file: `dashboard.py`
- App folders: `Project1` … `Project10`, each with its own `app.py` and README.

## Quick Start (Local)

- macOS, zsh assumed
- Python 3.9+ recommended

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Run the dashboard

```bash
streamlit run dashboard.py --server.port 8501
```

- Clicking a button starts an app on its own port (8601–8610) and shows its URL.

### 4) Stop apps

- Stop the dashboard with Ctrl+C. If a port remains busy:

```bash
lsof -i :8601
kill -9 <PID>
```

## Structure

```
Streamlit-App-Dashboard/
├── dashboard.py
├── requirements.txt
├── secrets.example.toml
├── Project1/
│   ├── app.py
│   └── README.md
├── Project2/
│   ├── app.py
│   └── README.md
... (up to Project10)
```

Each project is fully standalone; you can also run any one directly:

```bash
cd Project3
streamlit run app.py --server.port 8603
```
