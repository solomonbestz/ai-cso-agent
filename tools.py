from langchain_classic.agents import Tool
from main import *

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

unsupported_tool = Tool(
    name="unsupported",
    func=unsupported,
    description="It sends unsupported if intent returns unsupported"
)