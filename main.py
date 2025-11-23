import os, json, requests
import pickle
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_classic.agents import AgentExecutor, Tool
# from langchain_classic.chains import LLMChain
from langchain_classic.prompts import PromptTemplate
import azure.cognitiveservices.speech as speech
from langchain_classic.agents.mrkl.base import ZeroShotAgent


load_dotenv()

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

def extract_account_id(text: str):
    import re
    match = re.search(r'\b(\d{3})\b', text)
    return match.group(1) if match else None


def check_balance(account_id: str):
    # llm = azure_llm()
    acct = accounts.get(extract_account_id(account_id))

    if not acct:
        return {'error': "Account not found"}
    return {'account_id': account_id, "balance": acct['balance']}

def report_card_issue(account_id: str):
    return {"status": "blocked", "account_id": account_id, "next_step": "Collect new card in 48 hrs"}


classify_tool = Tool(
    name="IntentClassifier",
    func=classify_intent,
    description="Classifies user intent in a banking conversation."
)

balance_tool = Tool(
    name="CheckBalance",
    func=check_balance,
    description="Returns account balance for a given account_id"
)

card_tool = Tool(
    name="report_card_issue",
    func=report_card_issue,
    description="Block a card and return next steps"
)

if __name__=="__main__":
    accounts = {
        "001": {"name": "Ini", "balance": 200000},
        "002": {"name": "Bolu", "balance": 420000},
        "003": {"name": "Ebuks", "balance": 300000},
        "004": {"name": "Daniel", "balance": 250000}
    }


    llm = azure_llm()
    tools = [classify_tool, balance_tool, card_tool]

    prompt_agent = """You are a banking assistant. You MUST follow these steps:

    1. First, ALWAYS use the IntentClassifier tool to understand the user's intent
    2. Then, based on the classified intent, use the appropriate tool

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

    result = agent_executor.invoke({"input":"I lost my ATM card. My account is 001. Plese block it"})

    print(result["output"])
    