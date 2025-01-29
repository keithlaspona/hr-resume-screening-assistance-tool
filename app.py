import os
import sqlite3
import logging
from flask import Flask, render_template, request, session, redirect, url_for
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_fallback_secret")

UPLOAD_FOLDER = 'static/resumes'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Hardcoded credentials for login
USERNAME = "admin"
PASSWORD = "admin"

# Set up Google API key for ChatGoogleGenerativeAI
google_api_key = os.getenv("GOOGLE_API_KEY")

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
            text = "".join(page.extract_text() or "" for page in pdf.pages)
        return text.strip()
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        return f"Error reading PDF: {e}"

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        documents = [job_description] + resumes
        vectorizer = TfidfVectorizer().fit_transform(documents)
        vectors = vectorizer.toarray()
        job_description_vector = vectors[0]
        resume_vectors = vectors[1:]
        return cosine_similarity([job_description_vector], resume_vectors).flatten()
    except Exception as e:
        logging.error(f"Error in ranking resumes: {e}")
        return []

# Function to summarize text using ChatGoogleGenerativeAI
def summarize_text(text):
    try:
        google_api_key = os.getenv("google_api_key")
        if not google_api_key:
            return "Error: Google API key is missing."
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

        prompt = f"Summarize the following pdf resumes: {text[:1500]}"  # Limit text length
        response = llm.invoke(prompt)
        
        return response.content.strip()
    except Exception as e:
        logging.error(f"Error in summarization: {e}")
        return f"Error in summarization: {e}"

# Function to store data in SQLite3
def store_data_in_db(job_description, resumes, scores, summaries):
    try:
        conn = sqlite3.connect('resumes.db')
        c = conn.cursor()
        c.execute(''' 
        CREATE TABLE IF NOT EXISTS resume_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            job_description TEXT, 
            resume_name TEXT, 
            resume_text TEXT, 
            score REAL, 
            summary TEXT
        ) ''')

        for i, resume_text in enumerate(resumes):
            c.execute('''
            INSERT INTO resume_data (job_description, resume_name, resume_text, score, summary)
            VALUES (?, ?, ?, ?, ?) 
            ''', (job_description, f"resume_{i + 1}", resume_text, scores[i], summaries[i]))

        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    finally:
        conn.close()

# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == USERNAME and password == PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("main"))
        else:
            return render_template("index.html", error="Invalid username or password")

    return render_template("index.html")

# Protected route for main functionality
@app.route("/main", methods=["GET", "POST"])
def main():
    if not session.get("logged_in"):
        logging.info("User not logged in. Redirecting to login.")
        return redirect(url_for("login"))

    results, top_match = [], None

    if request.method == "POST":
        job_description = request.form.get("job_description", "").strip()
        uploaded_files = request.files.getlist("resumes")

        if not job_description:
            return render_template("main.html", results=[], top_match=None, error="Job description is required.")
        if not uploaded_files:
            return render_template("main.html", results=[], top_match=None, error="Please upload at least one resume.")

        resumes, summaries = [], []

        for file in uploaded_files:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            resume_text = extract_text_from_pdf(filepath)
            if resume_text.startswith("Error"):
                continue

            resumes.append(resume_text)
            summaries.append(summarize_text(resume_text))

        scores = rank_resumes(job_description, resumes)
        store_data_in_db(job_description, resumes, scores, summaries)

        results = list(zip([file.filename for file in uploaded_files], [round(score, 2) for score in scores], summaries))
        results.sort(key=lambda x: x[1], reverse=True)

        if results:
            top_match = results[0]

    return render_template("main.html", results=results, top_match=top_match, error=None)

# Logout route
@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    logging.info("User logged out. Redirecting to login.")
    return redirect(url_for("login"))

# Root route
@app.route("/")
def index():
    return redirect(url_for("main")) if session.get("logged_in") else redirect(url_for("login"))

# Run the app on Render
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)