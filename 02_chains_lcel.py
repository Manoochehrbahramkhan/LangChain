"""
02 - LCEL Chains: How modern LangChain composes things
Run: python 02_chains_lcel.py

LCEL = LangChain Expression Language
Core idea: runnable1 | runnable2 | runnable3
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def fake_translator(messages) -> str:
    # Pretend to translate: just uppercase for demo
    last = messages.to_messages()[-1].content
    return f"TRANSLATED: {last.upper()}"


def fake_summarizer(text: str) -> str:
    return f"Summary ({len(text)} chars): {text[:60]}..."


def main():
    # Chain 1: Simple sequential chain
    prompt = ChatPromptTemplate.from_template(
        "Translate this to French in one line: {sentence}"
    )
    chain1 = prompt | RunnableLambda(fake_translator) | StrOutputParser()
    print("--- Chain 1 ---")
    print(chain1.invoke({"sentence": "Hello, I want to learn LangChain"}))
    print()

    # Chain 2: Parallel + Passthrough (keep original input + add output)
    # RunnablePassthrough lets you pass input through while adding new keys
    rag_style_chain = RunnablePassthrough.assign(
        translation=chain1
    ) | RunnableLambda(
        lambda d: f"Original: {d['sentence']}\nFrench: {d['translation']}"
    )
    print("--- Chain 2 (Passthrough.assign) ---")
    print(rag_style_chain.invoke({"sentence": "LangChain is fun"}))
    print()

    # Chain 3: Python function in the middle
    full_chain = (
        ChatPromptTemplate.from_template("Write one fact about {topic}")
        | RunnableLambda(fake_translator)
        | RunnableLambda(fake_summarizer)
    )
    print("--- Chain 3 (function steps) ---")
    print(full_chain.invoke({"topic": "vectors"}))

    # TIP: .batch() runs same chain on many inputs in parallel
    print()
    print("--- Batch ---")
    print(chain1.batch([{"sentence": "hi"}, {"sentence": "good morning"}]))


if __name__ == "__main__":
    main()
