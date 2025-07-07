from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
import requests
import os

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///summary_history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class SummaryHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_text = db.Column(db.Text, nullable=False)
    summary_text = db.Column(db.Text, nullable=False)

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HF_API_KEY = os.getenv('HF_API_KEY')
HEADERS = {"Authorization": HF_API_KEY}

@app.route('/', methods=['GET', 'POST'])
def summarize():
    summary = None
    stats = None
    error = None

    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        max_length = int(request.form.get('max_length', 130))

        if len(text) < 50:
            error = "Input must be at least 50 characters"
        else:
            try:
                response = requests.post(
                    HF_API_URL,
                    headers=HEADERS,
                    json={
                        "inputs": text,
                        "parameters": {
                            "max_length": max_length,
                            "min_length": max(int(max_length/3), 30),
                            "do_sample": False
                        }
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                summary = result[0]['summary_text']
                
                stats = {
                    'original': len(text),
                    'summary': len(summary),
                    'reduction': round((1 - len(summary)/len(text)) * 100, 2)
                }
                
                new_entry = SummaryHistory(original_text=text, summary_text=summary)
                db.session.add(new_entry)
                db.session.commit()

            except Exception as e:
                error = f"Error: {str(e)}"

    return render_template(
        'index1.html',
        summary=summary,
        stats=stats,
        error=error,
        last_input=request.form.get('text', '')
    )

@app.route('/history')
def history():
    entries = SummaryHistory.query.order_by(SummaryHistory.id.desc()).all()
    return render_template('history.html', history=entries)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(port=5000, debug=True)
