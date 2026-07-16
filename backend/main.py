from model_schema import CNN,testing_trans
from fastapi import FastAPI,Request,Response,UploadFile,File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from PIL import Image
import torch
import os
from dotenv import load_dotenv
load_dotenv()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CNN().to(device=device)
model.load_state_dict(torch.load("model_weights.pth",map_location=device))

model.eval()

templates=Jinja2Templates(directory="../frontend/templates")


import json
import re

def parse_agent_response(content):
    """Extract plant_information and gardening_information as clean strings."""
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    try:
        data = json.loads(content)
        return data.get("plant_information", ""), data.get("gardening_information", "")
    except (json.JSONDecodeError, TypeError):
        pass

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return data.get("plant_information", ""), data.get("gardening_information", "")
        except json.JSONDecodeError:
            pass

    return content, ""  

"""
        ----creating langchgain tools-----
"""
@tool
def data_extraction(plant_type):
    """
    you are a great botanic scientist. i already created a cnn model for predicting the image class i used it for predingting it and i send it to you. 
    you find the details about the leaf and their impacts in medical field as well as about their fruits...
    Return plain text only, no Markdown formatting (no **, ###, or *)."""
    chat_model=init_chat_model(
        model="groq:meta-llama/llama-4-scout-17b-16e-instruct"
    )
    return chat_model.invoke(f"extract the details about {plant_type} leaves").content

@tool
def gardening_tips(plant_type):
    """
    you are a great botanic scientist. i already created a cnn model for predicting the image class i used it for predingting it and i send it to you. 
    you find the details about how to properly grow this plants including the weather,temp needed for this as well as daily rutineto grow the plant
    Return plain text only, no Markdown formatting (no **, ###, or *)."""
    chat_model=init_chat_model(
        model="groq:meta-llama/llama-4-scout-17b-16e-instruct"
    )
    return chat_model.invoke(f"extract the gardening details about {plant_type}").content

agent=create_agent(
    model="groq:meta-llama/llama-4-scout-17b-16e-instruct",
    tools=[data_extraction,gardening_tips],
    system_prompt="""
You are an expert botanist.

Whenever the user provides a plant name:

1. Call data_extraction.
2. Call gardening_tips.
3. Return the response as

{
    "plant_information": {...},
    "gardening_information": {...}
}

Always call BOTH tools.Return plain text only, no Markdown formatting (no **, ###, or *).
"""
)
classes=['Alstonia Scholaris',
 'Arjun',
 'Bael',
 'Basil',
 'Chinar',
 'Gauva',
 'Jamun',
 'Jatropha',
 'Lemon',
 'Mango',
 'Pomegranate',
 'Pongamia Pinnata']

def prediction(img):
    trans=testing_trans(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred=model(trans)
        output=torch.argmax(pred,1).item()
    return classes[output]


app=FastAPI()

@app.get("/")
def home(request:Request):
    return templates.TemplateResponse({"request":request},"index.html")



@app.post("/predict",response_class=HTMLResponse)
async def predicting(request:Request,img:UploadFile=File(...)):
    image=Image.open(img.file).convert("RGB")
    output=prediction(image)
    response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": output
            }
        ]
    }
)
    raw_content = response["messages"][-1].content
    plant_info, gardening_info = parse_agent_response(raw_content)

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "output": output,
            "plant_info": plant_info,
            "gardening_info": gardening_info,
        }
    )