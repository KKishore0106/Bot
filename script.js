function sendMessage() {
    let userInput = document.getElementById("userInput").value;
    if (!userInput.trim()) return;

    let chatBox = document.getElementById("chatBox");

    // Append user message
    let userMessage = document.createElement("div");
    userMessage.className = "chat-message user";
    userMessage.innerText = userInput;
    chatBox.appendChild(userMessage);

    // Send message to backend
    fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => response.json())
    .then(data => {
        let botMessage = document.createElement("div");
        botMessage.className = "chat-message bot";
        botMessage.innerText = data.reply;
        chatBox.appendChild(botMessage);

        // Scroll to bottom
        chatBox.scrollTop = chatBox.scrollHeight;
    });

    document.getElementById("userInput").value = "";
}
