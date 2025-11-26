import os, json, requests
import pickle
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_classic.agents import AgentExecutor, Tool
from langchain_classic.prompts import PromptTemplate
import azure.cognitiveservices.speech as speech
from langchain_classic.agents.mrkl.base import ZeroShotAgent

from tools import *


load_dotenv()


accounts = {
        "001": {"name": "Ini", "balance": 200000},
        "002": {"name": "Bolu", "balance": 420000},
        "003": {"name": "Ebuks", "balance": 300000},
        "004": {"name": "Daniel", "balance": 250000}
    }

def classify_intent(text: str):
    model_api_endpoint = os.getenv("model_endpoint")
    model_api_key = os.getenv("model_api_key")
    response = requests.post(url=model_api_endpoint, json={"text": text}, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {model_api_key}"
    })

    data = dict(response.json())
    print(data["prediction"])
    return data["prediction"]

def azure_llm():
    llm = AzureChatOpenAI(
        azure_endpoint = os.getenv("azure_resource_endpoint"),
        api_key = os.getenv("azure_resource_key"),
        api_version = "2024-12-01-preview",
        max_tokens=4096,
        temperature=0,
        azure_deployment="gpt-4o-mini"
    )

    return llm

def check_balance(account_id: str):
    acct = accounts.get(account_id, None)
    print(acct)
    if not acct:
        return {'error': "Account not found"}
    return {'account_id': account_id, "balance": acct['balance']}

def report_card_issue(account_id: str):
    return {"status": "blocked", "account_id": account_id, "next_step": "Collect new card in 48 hrs"}

def unsupported(text: str):
    return {"status": "Unsupported"}



if __name__=="__main__":
    llm = azure_llm()
    tools = [classify_tool, balance_tool, card_tool, unsupported_tool]

    prompt_agent = """You are a banking assistant. You MUST follow these steps:

    
    2. Then, based on the classified intent, use the appropriate tool
    3. If the user does not provide an account number, ask them to

    User query: {input}

    Let's think step by step:"""


    agent = ZeroShotAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        prefix=prompt_agent
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    result = agent_executor.invoke({"input":"How much do i have left?"})

    print(result["output"])
    