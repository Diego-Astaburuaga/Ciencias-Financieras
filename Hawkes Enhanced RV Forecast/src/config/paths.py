from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

DATA_DIR = ROOT_DIR / "data"
INTERIM_DATA_DIR = DATA_DIR / "interim"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
DOCUMENTS_DIR = ROOT_DIR / "documents"
TEMP_DIR = DATA_DIR / "temp"

INTERIM_DAILY_RV_DIR = INTERIM_DATA_DIR / "daily_rv"
INTERIM_EVENT_5_MIN_DIR = INTERIM_DATA_DIR / "event_5_min"
# Deprecated: Hawkes features are now aligned in event_5_min files.
INTERIM_HAWKES_DAILY_DIR = INTERIM_EVENT_5_MIN_DIR
PROCESSED_HAR_DIR = PROCESSED_DATA_DIR / "HAR"
PROCESSED_LSTM_DIR = PROCESSED_DATA_DIR / "LSTM"
TEMP_HAWKES_DIR = TEMP_DIR / "hawkes"

# Short aliases used by some scripts.
RAW_DATA = RAW_DATA_DIR
INTERIM_DATA = INTERIM_DATA_DIR
PROCESSED_DATA = PROCESSED_DATA_DIR

MODELS_DIR = ROOT_DIR / "models"