from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
import google.generativeai as genai
import pandas as pd
import requests
import faiss
import numpy as np
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

genai.configure(api_key="AIzaSyBWb3sptyhzHzqTWJSsiqTblR-g2v6pKU4")
OPENWEATHER_API_KEY = "8f46a8c3953b77fc1c3e7a582f1ff965"
YOUTUBE_API_KEY = "AIzaSyDWF3ON1BGMw4UPZAxxik8lblkNZFr7PWw"
gen_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

MYSQL_USER = "root"
MYSQL_PASSWORD = "root"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DB = "attendance_db"
DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
engine = create_engine(DATABASE_URL)

CSV_PATH = "/Users/supriya/Desktop/llm-mysql/attendance.csv"
df = pd.read_csv(CSV_PATH)
documents = df.apply(lambda row: f"{row['name']} was {row['status']} on {row['date']}", axis=1).tolist() 

embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # RAG based vector search
embeddings = embedding_model.encode(documents)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
text_store = documents

def search_similar_docs(query: str, k: int = 3):
    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return [text_store[i] for i in I[0]]

## Related to static DB
@app.get("/attendance")
def get_attendance(name: str = Query(...), month: str = Query(...)):
    query = text("""
        SELECT date FROM employee_attendance
        WHERE name = :name AND status = 'Absent' AND MONTH(date) = :month
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"name": name, "month": int(month)})
        dates = [str(row[0]) for row in result.fetchall()]
    context = f"{name} was absent on the following days in month {month}: {', '.join(dates)}"
    prompt = f"Context: {context}\nUser Question: What does this data mean?"
    response = gen_model.generate_content(prompt)
    return {"answer": response.text, "dates": dates}


def get_db_schema():
    with engine.connect() as conn:
        tables = conn.execute(text("SHOW TABLES")).fetchall()
        schema = {}
        for (table_name,) in tables:
            columns = conn.execute(text(f"DESCRIBE {table_name}")).fetchall()
            schema[table_name] = [col[0] for col in columns]
    return schema

def format_schema(schema):
    return "\n".join(f"Table `{table}`: columns = {', '.join(cols)}" for table, cols in schema.items())

def ask_gemini_for_sql(question: str):
    schema = format_schema(get_db_schema())
    prompt = f"""
You are an expert MySQL assistant. Based on the schema and question, write a safe SELECT SQL query only. Do not include any explanation.

{schema}

Question: {question}
SQL:
"""
    print("🧠 Prompt sent to Gemini:\n", prompt)  
    response = gen_model.generate_content(prompt)
    print("🧠 Gemini Response:\n", response.text)
     
    sql = response.text.strip()
    if sql.startswith("```sql"):
        sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql.split(";")[0] + ";"

def run_sql_query(sql: str):
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        columns = result.keys()
        return [dict(zip(columns, row)) for row in result]

# Weather Logic
def is_weather_query(q: str) -> bool:
    return any(word in q.lower() for word in ["weather", "temperature", "forecast", "climate"])

def call_weather_api(city: str):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json()
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"The current temperature in {city} is {temp}°C with {desc}."
    return "Couldn't fetch weather data."

# Youtube logic
def is_youtube_query(q: str) -> bool:
    return "youtube" in q.lower() or "video" in q.lower()

def call_youtube_api(query: str, max_results=3):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "key": YOUTUBE_API_KEY,
        "type": "video",
        "maxResults": max_results
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return [
            {
                "title": item["snippet"]["title"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            } for item in response.json()["items"]
        ]
    return [{"error": "Failed to fetch videos"}]

# UI
@app.get("/", response_class=HTMLResponse)
def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Ask
@app.get("/ask")
def query_with_natural_language(q: str = Query(...)):
    try:
        if is_weather_query(q):
            city_prompt = f"Extract the city name from this question: {q}"
            city = gen_model.generate_content(city_prompt).text.strip()
            return {"answer": call_weather_api(city)}

        elif is_youtube_query(q):
            topic_prompt = f"Extract the topic for YouTube search from this question: {q}"
            topic = gen_model.generate_content(topic_prompt).text.strip()
            return {"answer": call_youtube_api(topic)}

        else:
            try:
                sql = ask_gemini_for_sql(q)
                results = run_sql_query(sql)
                return {"answer": results}
            except Exception as sql_err:
                print("⚠️ SQL failed, using RAG. Error:", sql_err)
                similar = search_similar_docs(q)
                return {"answer": "\n".join(similar)}

    except Exception as e:
        print("❌ Global Error:", e)
        return {"error": "Sorry, could not process your query."}