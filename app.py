from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Load AI Model
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("query", "")

    if not user_input:
        return jsonify({"response": "Please enter a message."})

    # Generate AI response
    ai_response = generator(user_input, max_length=50, do_sample=True)[0]["generated_text"]

    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
