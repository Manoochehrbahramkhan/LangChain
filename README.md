# LangChain Learning Lab 🦜🔗

Welcome! This folder on your Desktop (`Desktop/langchain`) is your hands-on LangChain course.
All examples run **without an API key** first, so you can learn the concepts offline.

## 1. Setup (Windows PowerShell)

```powershell
cd "$env:USERPROFILE\Desktop\langchain"
python --version  # need 3.9+
pip install -r requirements.txt
copy .env.example .env
# edit .env and add OPENAI_API_KEY when you have one
```

Run lessons in order:

```powershell
python 01_hello_prompt.py
python 02_chains_lcel.py
python 03_memory_chat.py
python 04_rag_intro.py
python 05_agent_tools.py
```

## 2. What is LangChain?

LangChain is a framework to build apps with LLMs (like GPT-4, Claude, Llama).

Core mental model:

```
Prompt (instructions + variables)
   |
   v
Model (ChatOpenAI, Ollama, etc.)
   |
   v
Parser (StrOutputParser, JSON)
   |
   v
+ Memory / Retriever / Tools
```

Modern LangChain uses **LCEL** - LangChain Expression Language:

```python
chain = prompt | model | parser
result = chain.invoke({"topic": "RAG"})
```

`|` = pipe data through. Every piece is a `Runnable`.

## 3. Lessons Map

### 01_hello_prompt.py - Prompts
- `ChatPromptTemplate.from_messages([("system",...), ("human", "{topic}")])`
- Variables like `{topic}`, `{level}`
- `chain = prompt | model | StrOutputParser()`
- **Learn:** prompts are reusable templates, not just strings.

### 02_chains_lcel.py - LCEL Composition
- `chain1 = prompt | model | parser`
- `RunnablePassthrough.assign()` = keep input + add new field (core RAG pattern)
- `RunnableLambda(func)` = put any Python function in a chain
- `.batch([...])` = parallel calls
- **Learn:** build complex flows without nested callbacks.

### 03_memory_chat.py - Memory
- LLMs are stateless. You must send history each time.
- Modern way: `RunnableWithMessageHistory + InMemoryChatMessageHistory`
- `MessagesPlaceholder("history")` in prompt
- `session_id` = different users
- **Learn:** memory = store messages + inject into prompt.

### 04_rag_intro.py - RAG
Real RAG flow:
1. `Load` docs
2. `Split` with `CharacterTextSplitter(chunk_size=500, chunk_overlap=50)`
3. `Embed` with `OpenAIEmbeddings()`
4. `Store` in `FAISS` / `Chroma`
5. `Retrieve` top-k + `Generate` answer

This demo uses keyword search so it runs offline. Swap in real embeddings when ready (see bottom of file).

### 05_agent_tools.py - Tools & Agents
- `@tool` decorator turns Python function into LLM-callable tool
- Must have docstring = tool description for LLM
- Agent loop: `Thought -> Call Tool -> Observe -> Repeat -> Final Answer`
- Real agent needs `create_tool_calling_agent + AgentExecutor` + OpenAI key (example in file).

## 4. Key Imports Cheat Sheet (2024-2026)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.tools import tool
from langchain_text_splitters import CharacterTextSplitter

# Only when you have OPENAI_API_KEY:
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
```

Old tutorials use `ConversationChain`, `LLMChain` - those are **deprecated**. Use LCEL above.

## 5. Next Steps For You

1. Run all 5 files, change variables, break them.
2. Get OpenAI key from https://platform.openai.com -> put in `.env` -> replace `RunnableLambda(fake...)` with `ChatOpenAI(model="gpt-4o-mini")`
3. Try: build a PDF Q&A, a calculator agent, a chat bot with memory.
4. Learn LangSmith tracing, LangGraph for advanced agents.

Files:
- `requirements.txt` - dependencies
- `.env.example` - copy to `.env`
- `01_hello_prompt.py` … `05_agent_tools.py` - lessons

Happy building! Ask me "explain lesson 3" or "give me a RAG project" anytime.
