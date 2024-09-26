import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from langchain.prompts import PromptTemplate
from langchain_community.llms import GigaChat, YandexGPT
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

app = FastAPI()


# YandexGPT Endpoint
@app.post("/generate/yandexgpt")
async def generate_yandexgpt(request: Request):
    data = await request.json()
    try:
        model = YandexGPT(
            api_key=os.getenv("YANDEX_API_KEY"),
            model_uri=os.getenv("YANDEX_MODEL_URI"),
        )
        prompt = ChatPromptTemplate.from_template(data["prompt"])
        output_parser = StrOutputParser()
        chain = prompt | model | output_parser
        result = chain.invoke(data["variables"])
        return result
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key: {e}")


# GigaChat Endpoint
@app.post("/generate/gigachat")
async def generate_gigachat(request: Request):
    data = await request.json()
    try:
        model = GigaChat(verify_ssl_certs=False, scope="GIGACHAT_API_PERS")
        prompt = PromptTemplate(
            template=data["prompt"], input_variables=list(data["variables"].keys())
        )
        chain = prompt | model
        result = chain.invoke(data["variables"])
        return result
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key: {e}")


# oLLaMa Endpoint
@app.post("/generate/ollama")
async def generate_ollama(request: Request):
    data = await request.json()
    try:
        model = OllamaLLM(model=data["model"])
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
