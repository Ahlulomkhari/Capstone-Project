# Vehicle Loan Default Prediction Dashboard

An interactive dashboard for exploring and predicting vehicle loan defaults, built on the L&T/LTFS vehicle loan dataset (233,154 loan records).

## Prerequisites

- **Python 3.9 or later** — download from [python.org](https://www.python.org/downloads/)
  - On Windows, make sure to tick **"Add Python to PATH"** during installation.

## Installation

1. Download or clone this project folder.
2. Open a terminal (Command Prompt, PowerShell, or Terminal) **in this folder**.
3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Dashboard

Start the app with:

```bash
python main.py
```

The dashboard will open automatically in your browser at **http://localhost:8501**. If it doesn't open automatically, copy that address into your browser manually.

### Other ways to run it

| Command | What it does |
|---|---|
| `python main.py` | Normal startup — loads the dashboard with existing/cached data and models |
| `python main.py --retrain` | Retrains the underlying model(s) from scratch before launching the dashboard |
| `streamlit run dashboard.py` | Launches the dashboard directly via Streamlit, bypassing `main.py` |

## Data

- The full dataset (**233,154 loans**) is already included at:
  ```
  data/full/full_dataset.csv
  ```
  This powers the **Overview** tab by default — no setup needed.
- To analyze a different file, use the **upload control in the sidebar** to load your own CSV instead.
- **First load takes about 30-60 seconds** while the data is read and processed. After that, it's cached, so subsequent loads and tab switches are fast.

## Troubleshooting

- **"python is not recognized" / command not found** — Python isn't on your PATH. Reinstall Python and make sure to check "Add Python to PATH" (Windows), or use `python3` instead of `python` (macOS/Linux).
- **Port 8501 already in use** — another Streamlit app is probably still running. Close it, or run `streamlit run dashboard.py --server.port 8502` to use a different port.
- **Slow first load** — expected behavior with the full 233,154-row dataset; subsequent loads will be fast thanks to caching.
- **Dependency errors during `pip install`** — make sure you're using Python 3.9+ (`python --version`), and consider using a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate      # macOS/Linux
  venv\Scripts\activate         # Windows
  pip install -r requirements.txt
  ```
