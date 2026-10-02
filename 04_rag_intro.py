"""
04 - RAG Intro (Retrieval Augmented Generation) - runs OFFLINE
Run: python 04_rag_intro.py

Real RAG = Load docs -> Split -> Embed -> Store in vector DB -> Retrieve -> Answer
This demo uses keyword search instead of embeddings so you learn the FLOW
without needing an OpenAI key. Then upgrade to real embeddings.
"""
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DOCS = [
    "LangChain is a framework for building LLM apps with chains, prompts, memory, and tools.",
    "LCEL (LangChain Expression Language) lets you compose runnables with the pipe | operator.",
    "RAG means retrieving relevant documents and giving them to the LLM as context.",
    "Vector stores like FAISS or Chroma store embeddings for fast similarity search.",
]


def fake_llm(messages) -> str:
    content = messages.to_string()
    return f"Answer based on context:\n{content[:400]}..."


def simple_retriever(question: str) -> str:
    """Naive keyword retriever: score docs by word overlap."""
    q_words = set(question.lower().split())
    scored = sorted(
        DOCS,
        key=lambda d: len(q_words & set(d.lower().split())),
        reverse=True,
    )
    return "\n".join(scored[:2])


def main():
    # 1. Splitting (important for long docs)
    splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.split_text("\n".join(DOCS))
    print(f"Split into {len(chunks)} chunks. Example:")
    print(chunks[0][:100])
    print()

    # 2. RAG chain: {context from retriever + question} -> prompt -> model
    prompt = ChatPromptTemplate.from_template(
        "Answer using ONLY this context:\n{context}\n\nQuestion: {question}"
    )

    rag_chain = (
        {
            "context": RunnableLambda(lambda x: simple_retriever(x["question"])),
            "question": RunnableLambda(lambda x: x["question"]),
        }
        | prompt
        | RunnableLambda(fake_llm)
        | StrOutputParser()
    )

    print("--- RAG Query 1 ---")
    print(rag_chain.invoke({"question": "What is LCEL?"}))
    print()
    print("--- RAG Query 2 ---")
    print(rag_chain.invoke({"question": "What is RAG?"}))

    print()
    print("TO UPGRADE TO REAL RAG:")
    print("1. from langchain_openai import OpenAIEmbeddings, ChatOpenAI")
    print("2. embeddings = OpenAIEmbeddings()")
    print("3. from langchain_community.vectorstores import FAISS")
    print("   db = FAISS.from_texts(chunks, embeddings)")
    print("   retriever = db.as_retriever()")


if __name__ == "__main__":
    main()
