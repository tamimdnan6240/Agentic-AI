# =========================
# 0) Install (RELIABLE)
# =========================
!pip -q uninstall -y autogen pyautogen ag2 >/dev/null 2>&1
!pip -q install -U "ag2[openai]" python-dotenv

!pip show ag2
!python -c "import autogen; print('autogen version:', getattr(autogen,'__version__', 'unknown'))"

# ============================================================
# 1) Env (DO NOT hardcode keys)
#   Colab: Runtime > Secrets > add OPENAI_API_KEY
# ============================================================
import os, json, time, re, hashlib, sqlite3, sys
from dotenv import load_dotenv
load_dotenv()

# Read from Colab Secrets / .env
os.environ["OPENAI_API_KEY"] =   # set via Secrets
os.environ["OPENAI_MODEL_NAME"] = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")

print("OPENAI_MODEL_NAME:", os.environ["OPENAI_MODEL_NAME"])
print("OPENAI_API_KEY present?", bool(os.getenv("OPENAI_API_KEY")))

# ============================================================
# 2) Mount Drive + File Path(s)
# ============================================================
from google.colab import drive

FILE_PATH = "/content/drive/MyDrive/Network Maintenace - With & without maintenace/Original-data/2017-2019 (no main).csv"
csv_paths = [FILE_PATH]

if not os.path.exists("/content/drive/MyDrive"):
    drive.mount("/content/drive", force_remount=True)

for p in csv_paths:
    if not os.path.exists(p):
        raise FileNotFoundError(f"CSV path not found:\n{p}")

print("Mounted?", os.path.exists("/content/drive/MyDrive"))
print("CSV paths exist?", all(os.path.exists(p) for p in csv_paths))
