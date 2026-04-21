import json
import os
from datetime import datetime

DB_FILE = "cases_db.json"


def load_cases():
    if not os.path.exists(DB_FILE):
        return []

    with open(DB_FILE, "r") as f:
        return json.load(f)


def save_case(patient_id, data):
    cases = load_cases()

    case = {
        "patient_id": patient_id,
        "tumor": data["tumor"],
        "confidence": data["confidence"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    cases.append(case)

    with open(DB_FILE, "w") as f:
        json.dump(cases, f, indent=4)


def delete_case(index):
    cases = load_cases()
    cases.pop(index)

    with open(DB_FILE, "w") as f:
        json.dump(cases, f, indent=4)
