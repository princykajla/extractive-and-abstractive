from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from summarizer import Summarizer
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False)
    summary = db.Column(db.Text, nullable=False)

model = Summarizer()

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    if request.method == "POST":
        data = request.form["data"]
        maxL = int(request.form["maxL"])
        num_sentences = max(1, int(maxL // 20))  

        output = model(data, num_sentences=num_sentences)

        new_chat = ChatHistory(question=data, summary=output)
        db.session.add(new_chat)
        db.session.commit()

        return render_template("index.html", result=output)

    return render_template("index.html")

@app.route("/history")
def history():
    chats = ChatHistory.query.order_by(ChatHistory.id.desc()).all()
    return render_template("history.html", history=chats)

if __name__ == '__main__':
    app.run(debug=True)
