import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model(
    model=os.environ["OPENAI_API_MODEL"],
    model_provider="openai",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv("OPENAI_API_URL"),
)

response = llm.invoke("سلام! در یک جمله خودت را معرفی کن.")
print(response.content)