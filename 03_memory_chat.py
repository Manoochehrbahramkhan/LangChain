"""
03 - Memory / Chat History
Run: python 03_memory_chat.py

Concepts:
- Chat history matters, LLMs are stateless
- RunnableWithMessageHistory + InMemoryChatMessageHistory = modern memory
- session_id lets you have multiple users
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

store = {}  # session_id -> history object


def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def fake_chat_model(messages) -> str:
    # messages is ChatPromptValue, look at recent history
    text = messages.to_string()
    # Very dumb "memory-aware" fake: echo last 150 chars
    return f"I remember our chat so far. Last part was: ...{text[-150:]}"


def main():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with memory."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    base_chain = prompt | RunnableLambda(fake_chat_model)

    chain_with_history = RunnableWithMessageHistory(
        base_chain,
        get_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "user-123"}}

    print("--- Turn 1 ---")
    print(chain_with_history.invoke({"question": "My name is Alex"}, config=config))
    print()
    print("--- Turn 2 (should 'remember') ---")
    print(chain_with_history.invoke({"question": "What is my name?"}, config=config))
    print()
    print("--- Raw stored history ---")
    print(get_history("user-123").messages)

    # With real OpenAI:
    # from langchain_openai import ChatOpenAI
    # base_chain = prompt | ChatOpenAI(model="gpt-4o-mini")


if __name__ == "__main__":
    main()
