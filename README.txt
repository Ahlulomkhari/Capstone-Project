1. Install Python 3.9+ from python.org (tick "Add to PATH" on Windows).
2. Open a terminal in this folder and run:   pip install -r requirements.txt
3. Run:   python main.py
   The dashboard opens in your browser (http://localhost:8501).
Other options:  python main.py --retrain   |   streamlit run dashboard.py

FULL DATASET (233,154 loans) for the Overview tab:
  The full 233,154-loan file is already included in data/full/full_dataset.csv (or upload another in the sidebar).
  The first load takes about 30-60 seconds; after that it is cached.
