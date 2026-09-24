from dotenv import load_dotenv

from rag.llm.openai import OpenAIProvider

load_dotenv()

llm = OpenAIProvider(
    model="gpt-4.1-mini"
)

answer = llm.generate(
    "Explain overfitting in one sentence."
)

print(answer)