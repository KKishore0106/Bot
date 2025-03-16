from flask import Flask, request, jsonify

app = Flask(__name__)

def get_bot_response(user_input):
    responses = {
        "hello": "Hi there! How can I help you?",
        "how are you": "I'm just a bot, but I'm feeling great!",
        "symptoms": "Can you describe your symptoms?",
        "bye": "Goodbye! Take care!"
    }
    return responses.get(user_input.lower(), "Sorry, I didn't understand that.")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    bot_reply = get_bot_response(user_message)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
