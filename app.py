from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Simple bot response logic
def get_bot_response(user_input):
    responses = {
        "hello": "Hi there! 😊 How can I help you?",
        "how are you": "I'm just a bot, but I'm feeling great! 🎉",
        "symptoms": "Please describe your symptoms, and I'll try to assist. 🤔",
        "bye": "Goodbye! Take care! 👋"
    }
    return responses.get(user_input.lower(), "Sorry, I didn't understand that. 🤖")

# Route to render the chat UI
@app.route("/")
def home():
    return render_template("index.html")

# API route for chatbot responses
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    bot_reply = get_bot_response(user_message)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
