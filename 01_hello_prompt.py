"""
01 - Hello Prompt: Your first LangChain building blocks
Run: python 01_hello_prompt.py

Concepts:
- ChatPromptTemplate: reusable prompt with variables
- Messages: System / Human / AI
- StrOutputParser: converts ChatMessage -> string
- RunnableLambda: a fake model so you can learn WITHOUT an API key
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


def fake_model(messages) -> str:
    # messages is a ChatPromptValue, get text for demo
    text = messages.to_string()
    return f"[Fake AI answer to]: {text[:120]}..."


def main():
    # 1. Create a prompt template with variables
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly tutor teaching LangChain to a beginner."),
        ("human", "Explain {topic} in one short sentence for a {level}."),
    ])

    # 2. See what it looks like
    messages = prompt.format_messages(topic="LCEL", level="beginner")
    print("=== Formatted Messages ===")
    for m in messages:
        print(f"{m.type}: {m.content}")
    print()

    # 3. Build a Chain with LCEL: prompt | model | parser
    # RunnableLambda acts as a stand-in for ChatOpenAI() so this runs offline
    chain = prompt | RunnableLambda(fake_model) | StrOutputParser()

    result = chain.invoke({"topic": "prompts", "level": "beginner"})
    print("=== Chain Result ===")
    print(result)
    print()

    # 4. Try your own:
    result2 = chain.invoke({"topic": "RAG", "level": "5-year-old"})
    print(result2)

    # NEXT: To use real AI, replace RunnableLambda(fake_model) with:
    # from langchain_openai import ChatOpenAI
    # model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # chain = prompt | model | StrOutputParser()


if __name__ == "__main__":
    main()
