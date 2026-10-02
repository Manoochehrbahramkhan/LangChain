"""
05 - Tools & Agents - runs OFFLINE with a fake LLM
Run: python 05_agent_tools.py

Concepts:
- Tool = Python function LLM can call (@tool decorator)
- Agent = LLM that decides which tool to use in a loop
This demo shows the tool part offline. Uncomment the agent part when you have a key.
"""
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together. Use for any math addition."""
    return a + b


@tool
def get_word_count(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())


def main():
    print("--- Direct tool calls (no LLM needed) ---")
    print(f"add(2, 3) = {add.invoke({'a': 2, 'b': 3})}")
    print(f"word_count = {get_word_count.invoke({'text': 'LangChain tools are powerful'})}")
    print()
    print("Available tools:", [t.name for t in [add, get_word_count]])
    print(f"Tool description example: {add.description}")
    print()

    # Simulated agent decision with a fake model
    def fake_agent_decision(user_q: str) -> str:
        if "+" in user_q or "add" in user_q.lower():
            return f"TOOL_CALL: add -> {add.invoke({'a': 10, 'b': 20})}"
        else:
            return f"TOOL_CALL: get_word_count -> {get_word_count.invoke({'text': user_q})}"

    prompt = ChatPromptTemplate.from_template("{q}")
    agent_sim = prompt | RunnableLambda(lambda v: fake_agent_decision(v.to_string()))

    print("--- Simulated agent ---")
    print(agent_sim.invoke({"q": "Can you add 10 and 20?"}))
    print(agent_sim.invoke({"q": "How many words in this sentence here?"}))
    print()
    print("WITH REAL OPENAI KEY, use this:")
    print("""
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, [add, get_word_count], prompt)
executor = AgentExecutor(agent=agent, tools=[add, get_word_count])
print(executor.invoke({"input": "What is 15 + 27?"}) 
""")


if __name__ == "__main__":
    main()
