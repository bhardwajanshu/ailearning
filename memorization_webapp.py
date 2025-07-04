#!/usr/bin/env python3
"""
AI Memorization Web App - Exact Port of memorization_gui.py
All features converted to web-based interface using Flask

🔑 Features:
- Upload text/image content with OCR
- AI-powered short notes conversion
- Interactive quizzes and flashcards
- Speech practice with real-time feedback
- Progress tracking
- Voice reading
- Modern responsive web interface
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import sqlite3
import os
import base64
import io
import json
import random
import tempfile
import uuid
from datetime import datetime
from difflib import SequenceMatcher
from PIL import Image
import threading
import time

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

class MemorizationWebApp:
    def __init__(self):
        # 🔑 EMBED YOUR OPENAI API KEY HERE
        
        self.db_path = 'upsc_learning.db'
        self.client = None
        
        # Initialize database
        self.init_db()
        
    def get_openai_client(self):
        """Get OpenAI client instance"""
        if not OPENAI_AVAILABLE:
            return None
            
        if self.client is None:
            try:
                self.client = OpenAI(api_key=self.embedded_api_key)
                # Test the connection
                self.client.models.list()
            except Exception as e:
                print(f"OpenAI initialization error: {e}")
                return None
        return self.client
        
    def get_db_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
        
    def init_db(self):
        """Initialize database with all required tables"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        # Content table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content_type TEXT DEFAULT 'text',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Content lines table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id INTEGER NOT NULL,
                line_number INTEGER NOT NULL,
                line_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (content_id) REFERENCES content (id) ON DELETE CASCADE
            )
        ''')
        
        # Line memorization tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS line_memorization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line_id INTEGER NOT NULL,
                pronunciation_count INTEGER DEFAULT 0,
                accuracy_score REAL DEFAULT 0.0,
                is_memorized BOOLEAN DEFAULT FALSE,
                last_practiced TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (line_id) REFERENCES content_lines (id) ON DELETE CASCADE
            )
        ''')
        
        # Quiz results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quiz_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                user_answer TEXT,
                is_correct BOOLEAN DEFAULT FALSE,
                taken_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (line_id) REFERENCES content_lines (id) ON DELETE CASCADE
            )
        ''')
        
        # Explanations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line_id INTEGER NOT NULL,
                explanation TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (line_id) REFERENCES content_lines (id) ON DELETE CASCADE
            )
        ''')
        
        # Flashcard results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS flashcard_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line_id INTEGER NOT NULL,
                front_text TEXT NOT NULL,
                back_text TEXT NOT NULL,
                difficulty_rating INTEGER DEFAULT 2,
                review_count INTEGER DEFAULT 0,
                next_review TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (line_id) REFERENCES content_lines (id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def extract_text_from_image(self, image):
        """Extract text from image using OpenAI Vision API"""
        if not OPENAI_AVAILABLE:
            return "OCR not available - OpenAI not installed"
            
        client = self.get_openai_client()
        if not client:
            return "OpenAI API not available"
            
        try:
            # Convert PIL Image to base64
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this image. Return only the text content, maintaining the original structure and formatting."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error extracting text: {str(e)}"
            
    def convert_to_short_notes(self, content_text):
        """Convert content to short notes using OpenAI"""
        if not OPENAI_AVAILABLE:
            return "AI conversion not available - OpenAI not installed"
            
        client = self.get_openai_client()
        if not client:
            return "OpenAI API not available"
            
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at converting long text into concise, memorable short notes. 
                        Create bullet points that capture the key information while being easy to memorize.
                        
                        Format your response as:
                        • Point 1
                        • Point 2
                        • Point 3
                        etc.
                        
                        Focus on:
                        - Key facts and concepts
                        - Important dates, names, numbers
                        - Cause-effect relationships
                        - Definitions and explanations
                        - Make it concise but comprehensive"""
                    },
                    {
                        "role": "user",
                        "content": f"Convert this content into short, memorable notes:\n\n{content_text}"
                    }
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error converting to short notes: {str(e)}"
            
    def generate_quiz_question(self, line_text):
        """Generate quiz question for a line"""
        if not OPENAI_AVAILABLE:
            return None
            
        client = self.get_openai_client()
        if not client:
            return None
            
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Create a quiz question based on the given text. 
                        Return a JSON object with:
                        - "question": the quiz question
                        - "correct_answer": the correct answer
                        - "options": array of 4 multiple choice options (including the correct one)
                        - "explanation": brief explanation of the answer
                        
                        Make the question challenging but fair."""
                    },
                    {
                        "role": "user",
                        "content": f"Create a quiz question for: {line_text}"
                    }
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return json.loads(response.choices[0].message.content.strip())
            
        except Exception as e:
            print(f"Error generating quiz: {e}")
            return None
            
    def generate_flashcard(self, line_text):
        """Generate flashcard for a line"""
        if not OPENAI_AVAILABLE:
            return None
            
        client = self.get_openai_client()
        if not client:
            return None
            
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Create a flashcard based on the given text.
                        Return a JSON object with:
                        - "front": the question or prompt side
                        - "back": the answer or explanation side
                        
                        Make it concise and focused on key information."""
                    },
                    {
                        "role": "user",
                        "content": f"Create a flashcard for: {line_text}"
                    }
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return json.loads(response.choices[0].message.content.strip())
            
        except Exception as e:
            print(f"Error generating flashcard: {e}")
            return None
            
    def calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        def clean_text(text):
            return ''.join(c.lower() for c in text if c.isalnum() or c.isspace()).strip()
        
        cleaned1 = clean_text(text1)
        cleaned2 = clean_text(text2)
        
        return SequenceMatcher(None, cleaned1, cleaned2).ratio()

# Initialize the app
memorization_app = MemorizationWebApp()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_content():
    """Upload text or image content"""
    if request.method == 'POST':
        content_type = request.form.get('content_type', 'text')
        
        if content_type == 'text':
            title = request.form.get('title', '').strip()
            content = request.form.get('content', '').strip()
            
            if not title or not content:
                flash('Title and content are required!', 'error')
                return redirect(url_for('upload_content'))
                
            # Save to database
            conn = memorization_app.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO content (title, content_type) VALUES (?, ?)
            ''', (title, content_type))
            
            content_id = cursor.lastrowid
            
            # Split content into lines and save
            lines = content.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    cursor.execute('''
                        INSERT INTO content_lines (content_id, line_number, line_text)
                        VALUES (?, ?, ?)
                    ''', (content_id, i + 1, line))
            
            conn.commit()
            conn.close()
            
            flash('Content uploaded successfully!', 'success')
            return redirect(url_for('view_content'))
            
        elif content_type == 'image':
            if 'image' not in request.files:
                flash('No image file provided!', 'error')
                return redirect(url_for('upload_content'))
                
            file = request.files['image']
            if file.filename == '':
                flash('No image selected!', 'error')
                return redirect(url_for('upload_content'))
                
            try:
                # Process image
                image = Image.open(file.stream)
                
                # Extract text using OCR
                extracted_text = memorization_app.extract_text_from_image(image)
                
                # Return for title input
                session['extracted_text'] = extracted_text
                return render_template('upload.html', extracted_text=extracted_text)
                
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'error')
                return redirect(url_for('upload_content'))
    
    return render_template('upload.html')

@app.route('/save_extracted', methods=['POST'])
def save_extracted_content():
    """Save extracted text content"""
    title = request.form.get('title', '').strip()
    content = request.form.get('content', '').strip()
    
    if not title or not content:
        flash('Title and content are required!', 'error')
        return redirect(url_for('upload_content'))
    
    # Save to database
    conn = memorization_app.get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO content (title, content_type) VALUES (?, ?)
    ''', (title, 'image'))
    
    content_id = cursor.lastrowid
    
    # Split content into lines and save
    lines = content.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if line:
            cursor.execute('''
                INSERT INTO content_lines (content_id, line_number, line_text)
                VALUES (?, ?, ?)
            ''', (content_id, i + 1, line))
    
    conn.commit()
    conn.close()
    
    flash('Content saved successfully!', 'success')
    return redirect(url_for('view_content'))

@app.route('/content')
def view_content():
    """View all content"""
    conn = memorization_app.get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT c.id, c.title, c.content_type, c.created_at,
               COUNT(cl.id) as line_count
        FROM content c
        LEFT JOIN content_lines cl ON c.id = cl.content_id
        GROUP BY c.id, c.title, c.content_type, c.created_at
        ORDER BY c.created_at DESC
    ''')
    
    content_list = cursor.fetchall()
    conn.close()
    
    return render_template('content.html', content_list=content_list)

@app.route('/content/<int:content_id>')
def view_content_detail(content_id):
    """View detailed content"""
    conn = memorization_app.get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM content WHERE id = ?', (content_id,))
    content = cursor.fetchone()
    
    if not content:
        flash('Content not found!', 'error')
        return redirect(url_for('view_content'))
    
    cursor.execute('''
        SELECT cl.*, lm.pronunciation_count, lm.accuracy_score, lm.is_memorized
        FROM content_lines cl
        LEFT JOIN line_memorization lm ON cl.id = lm.line_id
        WHERE cl.content_id = ?
        ORDER BY cl.line_number
    ''', (content_id,))
    
    lines = cursor.fetchall()
    conn.close()
    
    return render_template('content_detail.html', content=content, lines=lines)

@app.route('/convert/<int:content_id>')
def convert_content(content_id):
    """Convert content to short notes"""
    conn = memorization_app.get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM content WHERE id = ?', (content_id,))
    content = cursor.fetchone()
    
    if not content:
        flash('Content not found!', 'error')
        return redirect(url_for('view_content'))
    
    cursor.execute('''
        SELECT line_text FROM content_lines 
        WHERE content_id = ? 
        ORDER BY line_number
    ''', (content_id,))
    
    lines = cursor.fetchall()
    content_text = '\n'.join([line['line_text'] for line in lines])
    
    # Convert to short notes
    short_notes = memorization_app.convert_to_short_notes(content_text)
    
    conn.close()
    
    return render_template('convert_result.html', 
                         content=content, 
                         original_text=content_text,
                         short_notes=short_notes)

@app.route('/save_converted', methods=['POST'])
def save_converted_content():
    """Save converted short notes as new content"""
    title = request.form.get('title', '').strip()
    content = request.form.get('content', '').strip()
    
    if not title or not content:
        flash('Title and content are required!', 'error')
        return redirect(url_for('view_content'))
    
    # Save to database
    conn = memorization_app.get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO content (title, content_type) VALUES (?, ?)
    ''', (title, 'converted'))
    
    content_id = cursor.lastrowid
    
    # Split content into lines and save
    lines = content.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if line:
            cursor.execute('''
                INSERT INTO content_lines (content_id, line_number, line_text)
                VALUES (?, ?, ?)
            ''', (content_id, i + 1, line))
    
    conn.commit()
    conn.close()
    
    flash('Converted content saved successfully!', 'success')
    return redirect(url_for('view_content'))

@app.route('/practice/<int:content_id>')
def practice_content(content_id):
    """Practice content with speech recognition"""
    conn = memorization_app.get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM content WHERE id = ?', (content_id,))
    content = cursor.fetchone()
    
    if not content:
        flash('Content not found!', 'error')
        return redirect(url_for('view_content'))
    
    cursor.execute('''
        SELECT cl.*, lm.pronunciation_count, lm.accuracy_score
        FROM content_lines cl
        LEFT JOIN line_memorization lm ON cl.id = lm.line_id
        WHERE cl.content_id = ?
        ORDER BY cl.line_number
    ''', (content_id,))
    
    lines = cursor.fetchall()
    conn.close()
    
    return render_template('practice.html', content=content, lines=lines)

@app.route('/quiz/<int:content_id>')
def quiz_content(content_id):
    """Take quiz on content"""
    conn = memorization_app.get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM content WHERE id = ?', (content_id,))
    content = cursor.fetchone()
    
    if not content:
        flash('Content not found!', 'error')
        return redirect(url_for('view_content'))
    
    cursor.execute('''
        SELECT * FROM content_lines 
        WHERE content_id = ? 
        ORDER BY line_number
    ''', (content_id,))
    
    lines = cursor.fetchall()
    conn.close()
    
    return render_template('quiz.html', content=content, lines=lines)

@app.route('/api/generate_quiz', methods=['POST'])
def api_generate_quiz():
    """Generate quiz question for a line"""
    data = request.get_json()
    line_text = data.get('line_text', '')
    
    if not line_text:
        return jsonify({'error': 'Line text is required'}), 400
    
    quiz_data = memorization_app.generate_quiz_question(line_text)
    
    if not quiz_data:
        return jsonify({'error': 'Failed to generate quiz'}), 500
    
    return jsonify(quiz_data)

@app.route('/api/generate_flashcard', methods=['POST'])
def api_generate_flashcard():
    """Generate flashcard for a line"""
    data = request.get_json()
    line_text = data.get('line_text', '')
    
    if not line_text:
        return jsonify({'error': 'Line text is required'}), 400
    
    flashcard_data = memorization_app.generate_flashcard(line_text)
    
    if not flashcard_data:
        return jsonify({'error': 'Failed to generate flashcard'}), 500
    
    return jsonify(flashcard_data)

@app.route('/api/save_quiz_result', methods=['POST'])
def api_save_quiz_result():
    """Save quiz result"""
    data = request.get_json()
    line_id = data.get('line_id')
    question = data.get('question')
    correct_answer = data.get('correct_answer')
    user_answer = data.get('user_answer')
    is_correct = data.get('is_correct', False)
    
    conn = memorization_app.get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO quiz_results (line_id, question, correct_answer, user_answer, is_correct)
        VALUES (?, ?, ?, ?, ?)
    ''', (line_id, question, correct_answer, user_answer, is_correct))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/api/check_speech', methods=['POST'])
def api_check_speech():
    """Check speech accuracy"""
    data = request.get_json()
    original_text = data.get('original_text', '')
    spoken_text = data.get('spoken_text', '')
    
    if not original_text or not spoken_text:
        return jsonify({'error': 'Both texts are required'}), 400
    
    similarity = memorization_app.calculate_text_similarity(original_text, spoken_text)
    accuracy_score = similarity * 100
    
    return jsonify({
        'accuracy_score': accuracy_score,
        'similarity': similarity,
        'feedback': 'Excellent!' if accuracy_score > 90 else 'Good!' if accuracy_score > 70 else 'Keep practicing!'
    })

@app.route('/progress')
def view_progress():
    """View learning progress"""
    conn = memorization_app.get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT c.id, c.title, c.content_type, c.created_at,
               COUNT(cl.id) as total_lines,
               SUM(CASE WHEN lm.pronunciation_count >= 5 THEN 1 ELSE 0 END) as practiced_lines,
               SUM(CASE WHEN lm.is_memorized THEN 1 ELSE 0 END) as memorized_lines,
               AVG(lm.accuracy_score) as avg_accuracy
        FROM content c
        LEFT JOIN content_lines cl ON c.id = cl.content_id
        LEFT JOIN line_memorization lm ON cl.id = lm.line_id
        GROUP BY c.id, c.title, c.content_type, c.created_at
        ORDER BY c.created_at DESC
    ''')
    
    progress_data = cursor.fetchall()
    conn.close()
    
    return render_template('progress.html', progress_data=progress_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 
