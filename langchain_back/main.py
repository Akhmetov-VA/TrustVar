import os
from typing import Optional
import openai
from fastapi import FastAPI, HTTPException, Request
from langchain.prompts import PromptTemplate
from langchain_community.llms import GigaChat, YandexGPT
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.llms import LlamaCpp
from langchain.llms import HuggingFacePipeline
from langchain.llms import OpenAI
from transformers import pipeline

app = FastAPI()

@app.post("/generate")
async def generate_locally(request: Request):
    data = await request.json()
    try:
        model_name = data["model"]
        # Модели с API
        if model_name.startswith('openai'):
            MODEL_NAME = model_name.split('/')[-1]
            OPENAI_KEY = os.getenv("OPENAI_KEY")
            model = OpenAI(
                model_name=MODEL_NAME,
                api_key=OPENAI_KEY
            )
        elif model_name.startswith('yandexgpt'):
            MODEL_NAME = model_name
            YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
            BASE_URL = os.getenv("YANDEX_BASE_URL")

            model = OpenAI(
                model_name=MODEL_NAME,
                api_key=YANDEX_API_KEY,
                base_url=BASE_URL
            )
            # YANDEX_MODEL_URI = os.getenv("YANDEX_MODEL_URI") + model_name
            # YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
            # model = YandexGPT(
            #     api_key=YANDEX_API_KEY,
            #     model_uri=YANDEX_MODEL_URI
            # )

        elif model_name.startswith('sber'):
            MODEL_NAME = model_name.split('/')[-1]
            GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")
            BASE_URL = os.getenv("GIGACHAT_BASE_URL")
            
            model = OpenAI(
                model_name=MODEL_NAME,
                api_key=GIGACHAT_API_KEY,
                base_url=BASE_URL
            )
            #model = GigaChat(model=model_name, credentials=GIGACHAT_API_TOKEN,verify_ssl_certs=False, scope="GIGACHAT_API_PERS")
        
        # Модели локальные
        elif (not model_name.startswith('ai-sage')) and model_name[0].islower(): # примитивное правило как отделить модели ollama от hf
            model = OllamaLLM(model=model_name)
        elif model_name.endswith(".gguf"):
            model = LlamaCpp(model=model_name)
        else:
            pipe = pipeline(
                "text-generation",
                model=model_name
            )
            model = HuggingFacePipeline(pipeline=pipe)
        
        prompt = ChatPromptTemplate.from_template(data["prompt"])
        output_parser = StrOutputParser()
        chain = prompt | model | output_parser
        result = chain.invoke(data["variables"])
        return result
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key: {e}")
    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=27361)
