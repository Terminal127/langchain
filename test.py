from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (ChatPromptTemplate,SystemMessagePromptTemplate,MessagesPlaceholder)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import warnings

warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")

# Gemini Model Setup
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key="AIzaSyDsi82MHuNMwZyUoJ5q6xN8yd9Q4yBw5gM",
    convert_system_message_to_human=True
)

chatmap = {}
def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chatmap:
        chatmap[session_id] = InMemoryChatMessageHistory()
    return chatmap[session_id]

another_promt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "You are a helpful and knowledgeable assistant.\n"
         "Always reply concisely, ideally in 2–3 sentences or short numbered bullet points.\n"
         "Start your answer with '#'.\n"
         "If you don't know the answer, say 'I don't know'.\n"),
        ("human", "What is LangChain?"),
        ("ai", "# LangChain is an open-source Python framework that helps developers build applications powered by large language models (LLMs). It makes it easier to connect models to data and compose complex workflows."),
        ("human", "What is a neural network?"),
        ("ai", "# A neural network is a computing system inspired by the human brain. It learns to perform tasks by analyzing examples and adjusting internal connections between artificial neurons."),
        ("human", "What is quantum computing?"),
        ("ai", "# Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously. This enables certain computations to be done much faster than on classical computers."),
        ("human", "What is the capital of Japan?"),
        ("ai", "# The capital of Japan is Tokyo."),
        ("human", "Who won the FIFA World Cup in 2018?"),
        ("ai", "# France won the 2018 FIFA World Cup by defeating Croatia 4–2 in the final."),

        MessagesPlaceholder(variable_name="history"),
        
        ("human", "{question}")
    ]
)

pipeline = another_promt | model
session_id = "e271b573-e69c-4a24-8a92-1e81f0243029"

# Fixed: Pass the function itself, not the result of calling it
pipeline_with_history = RunnableWithMessageHistory(
    runnable=pipeline,
    get_session_history=get_chat_history,  # Remove (session_id) - pass the function
    input_messages_key="question",
    history_messages_key="history"
)

# Run example question
question = "What is the capital of France?"
response = pipeline_with_history.invoke(
    {"question": question},
    config={"configurable": {"session_id": session_id}}  # Pass session_id in config
)
print(f"Question: {question}")
print(f"Response: {response}")

# Print chat history
print("\n📝 Chat history:")
for msg in chatmap[session_id].messages:
    print(f"{msg.type}: {msg.content}")