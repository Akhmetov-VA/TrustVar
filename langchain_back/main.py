import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from langchain.prompts import PromptTemplate
from langchain_community.llms import GigaChat, YandexGPT
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.llms import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from transformers import pipeline

app = FastAPI()


@app.post("/generate")
async def generate_locally(request: Request):
    data = await request.json()
    try:
        model_name = data["model"]
        # Models with API
        if model_name.startswith('api'):
            MODEL_NAME = model_name.split('/')[-1]
            OPENAI_KEY = os.getenv("OPENAI_KEY")
            BASE_URL = os.getenv('OPENAI_BASE_URL')
            
            model = ChatOpenAI(
                model_name=MODEL_NAME,
                api_key=OPENAI_KEY,
                base_url=BASE_URL,
                temperature=0,
                max_tokens=2048
            )
        elif model_name.startswith('yandexgpt'):
            YANDEX_MODEL_URI = os.getenv("YANDEX_MODEL_URI") + model_name
            YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
            model = YandexGPT(
                api_key=YANDEX_API_KEY,
                model_uri=YANDEX_MODEL_URI
            )
        # elif model_name[0].isupper() and not model_name.startswith('ZimaBlueAI'):
        #     pipe = pipeline(
        #         "text-generation",
        #         model=model_name
        #     )
        #     model = HuggingFacePipeline(pipeline=pipe)
        # The models are local
        else:
            model = OllamaLLM(model=model_name, base_url=os.getenv("OLLAMA_BASE_URL"))
        
        prompt = ChatPromptTemplate.from_template(data["prompt"])
        output_parser = StrOutputParser()
        chain = prompt | model | output_parser
        result = chain.invoke(data["variables"])
        
        if isinstance(result, str):
            return result
        else:
            return result.content
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key: {e}")
    except Exception as e:
        print('ERROR:', e)
    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=45321)
