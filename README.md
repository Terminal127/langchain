# 🤖 LangChain Agent Chat Application

A powerful conversational AI agent built with LangChain and Google Gemini 2.0 Flash, featuring tool integration, command history, and streaming responses.

## ✨ Features

### 🧠 **Intelligent Agent**
- **Google Gemini 2.0 Flash** integration for advanced AI responses
- **Tool-calling capabilities** with automatic execution
- **Multi-step task handling** with improved prompt engineering
- **Session-based chat history** with persistent storage

### 🛠️ **Built-in Tools**
- **⏰ Time & Date** - Get current timestamp
- **🧮 Calculator** - Safe mathematical expressions
- **🌤️ Weather** - Mock weather data (extensible)
- **📁 File Operations** - Read, write, and list files
- **💻 System Commands** - Execute safe shell commands
- **📜 Chat History** - Access conversation memory
- **📋 Task Planner** - Handle complex multi-step requests

### 🎨 **User Experience**
- **⌨️ Command History** - Navigate with ↑/↓ arrow keys
- **🎬 Streaming Responses** - ChatGPT-like typing effect
- **📱 Session Management** - Create, clear, and switch sessions
- **🎯 Smart Prompting** - Forces AI to actually use tools

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Setup
```bash
export GOOGLE_API_KEY="your-gemini-api-key-here"
```

### Run the Application
```bash
python new.py
# or for streaming version
python testnew.py
```

## 📖 Usage

### Basic Commands
```
🧑 You: What time is it?
🤖 AI: # The current time is 2025-07-14 12:30:45.

🧑 You: Calculate 25 * 4 + 100
🤖 AI: # The result is 200.

🧑 You: Create a story and save it to a file
🤖 AI: # I've created a bedtime story and saved it to story.txt.
```

### Application Commands
- `/help` - Show available commands
- `/history` - Display chat history
- `/clear` - Clear current session
- `/session` - Show session information
- `/new` - Start new chat session
- `/tools` - List available tools
- `/stream` - Toggle streaming (testnew.py only)
- `/quit` - Exit application

### Arrow Key Navigation
- **↑** - Previous command
- **↓** - Next command
- **Ctrl+D** - Exit gracefully

## 🏗️ Architecture

### Core Components

```
├── new.py              # Main application
├── testnew.py          # Streaming version
├── chat_history.json   # Session persistence
├── requirements.txt    # Dependencies
└── README.md          # This file
```

### Tool System
Each tool is a decorated function that the AI can call:

```python
@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

### Enhanced Prompting
The system uses advanced prompt engineering to ensure the AI:
- ✅ Actually uses tools instead of just describing them
- ✅ Follows through on complex multi-step tasks
- ✅ Provides consistent, helpful responses

## 🔧 Configuration

### Streaming Settings (testnew.py)
```python
STREAMING_ENABLED = True
STREAM_DELAY = 0.03      # Delay between characters
STREAM_CHUNK_SIZE = 1    # Characters per chunk
```

### Model Configuration
```python
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=api_key
)
```

## 📁 File Structure

```
langchain/
│
├── new.py                    # Main chat application
├── testnew.py               # Streaming version
├── chat_history.json        # Persistent chat storage
├── requirements.txt         # Python dependencies
├── README.md               # Documentation
├── .gitignore             # Git ignore rules
└── ~/.langchain_chat_history # Command history
```

## 🛡️ Security Features

### Safe Command Execution
Only whitelisted commands are allowed:
```python
safe_commands = ['ls', 'pwd', 'date', 'whoami', 'echo', 'cat', 'head', 'tail']
```

### Input Validation
- Mathematical expressions are sanitized
- File operations are contained to current directory
- Command timeouts prevent hanging

## 🎯 Advanced Usage

### Complex Task Example
```
🧑 You: Get the current time, and if it's past 10 PM, create a goodnight story for the user anubhav and save it to a file

🤖 AI: # The current time is 23:45:12. Since it's past 10 PM, I've created a personalized goodnight story for anubhav and saved it to goodnight_story_anubhav.txt.
```

### Multi-Step Workflows
The AI can handle complex requests involving multiple tools:
1. **Planning** - Uses task_planner for complex requests
2. **Execution** - Calls tools in proper sequence
3. **Verification** - Confirms task completion

## 🔄 Version Differences

### `new.py` (Standard Version)
- Immediate response display
- Standard terminal interaction
- Lightweight and fast

### `testnew.py` (Streaming Version)
- ChatGPT-like streaming effect
- Configurable typing speed
- Enhanced visual feedback
- Thinking indicators

## 🐛 Troubleshooting

### Common Issues

**API Key Not Set**
```bash
export GOOGLE_API_KEY="your-key-here"
```

**Module Not Found**
```bash
pip install -r requirements.txt
```

**Arrow Keys Not Working**
- Ensure readline is installed: `pip install readline`
- On Windows, try: `pip install pyreadline3`

**Streaming Issues**
- Use `new.py` for immediate responses
- Check terminal compatibility for streaming

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- **LangChain** - For the agent framework
- **Google Gemini** - For the AI model
- **Python readline** - For command history
- **Contributors** - For improvements and feedback

---

**Happy Chatting! 🎉**

For issues or questions, please check the troubleshooting section or create an issue in the repository.
