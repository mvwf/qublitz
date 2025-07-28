from fastapi import FastAPI, HTTPException, Security, UploadFile, File
from fastapi.security.api_key import APIKeyHeader
import sqlite3
import pandas as pd
import io
import json

app = FastAPI()
DB_PATH = "students.db"
API_KEYS_FILE = "keys.json"

# Load API keys and roles from JSON
with open(API_KEYS_FILE, "r") as f:
    api_keys_db = json.load(f)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header is None:
        raise HTTPException(status_code=401, detail="API key missing")
    if api_key_header not in api_keys_db:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key_header

def check_permission(api_key: str, required_role: str):
    user_role = api_keys_db[api_key]["role"]
    # write permission implies read permission
    if required_role == "read" and user_role in ("read", "write"):
        return True
    if required_role == "write" and user_role == "write":
        return True
    raise HTTPException(status_code=403, detail=f"Permission '{required_role}' required")

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...), api_key: str = Security(get_api_key)):
    check_permission(api_key, "write")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")

    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql("students", conn, if_exists="replace", index=False)
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"DB update failed: {e}")
    conn.close()
    return {"message": "Database updated successfully."}

@app.get("/student/{student_id}")
def get_student(student_id: str, api_key: str = Security(get_api_key)):
    check_permission(api_key, "read")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
    record = cursor.fetchone()
    conn.close()

    if not record:
        raise HTTPException(status_code=404, detail="Student not found")

    columns = ['student_id', 'student_name', 't1_time', 'frequency']
    data = dict(zip(columns, record))
    return data

@app.get("/me")
def get_me(api_key: str = Security(get_api_key)):
    user_info = api_keys_db[api_key]
    return {"user": user_info["user"], "role": user_info["role"]}