from langchain_community.llms import YandexGPT
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# model = YandexGPT(
#     api_key="тут нужен ключ",
#     model_uri="gpt://туту нужен ключ/yandexgpt-lite",
# )

# template = "The capital of the country {country} this?"
# prompt = PromptTemplate.from_template(template)
# llm_chain = prompt | model
# country = "Russia"

# result = llm_chain.invoke(country)
# print(result)


model = OllamaLLM(model="gemma:7b-instruct-v1.1-q4_0")

template = "The capital of the country {country} this?"
prompt = PromptTemplate.from_template(template)
llm_chain = prompt | model
country = "Russsia"

result = llm_chain.invoke(country)
print(result)
