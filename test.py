import os, json, requests
import pickle
import threading
import queue
import tempfile
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_classic.agents import AgentExecutor, Tool
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.agents.mrkl.base import ZeroShotAgent
import azure.cognitiveservices.speech as speech_sdk

load_dotenv()

accounts = {
        "001": {"name": "Ini", "balance": 200000},
        "002": {"name": "Bolu", "balance": 420000},
        "003": {"name": "Ebuks", "balance": 300000},
        "004": {"name": "Daniel", "balance": 250000}
    }


# Your existing functions remain the same
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

class VoiceBankingAssistant:
    def __init__(self):
        # Initialize Azure Speech Services
        self.speech_config = speech_sdk.SpeechConfig(
            subscription=os.getenv("AZURE_SPEECH_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION")
        )
        
        # Audio output setup
        self.audio_config = speech_sdk.audio.AudioOutputConfig(use_default_speaker=True)
        
        # Speech synthesizer for TTS
        self.speech_synthesizer = speech_sdk.SpeechSynthesizer(
            speech_config=self.speech_config, 
            audio_config=self.audio_config
        )
        
        # Speech recognizer for STT
        self.speech_recognizer = speech_sdk.SpeechRecognizer(speech_config=self.speech_config)
        
        # Initialize the banking agent
        self.agent = self._setup_banking_agent()
        
    def _setup_banking_agent(self):
        """Setup your existing banking agent"""
        
        llm = azure_llm()
        tools = [classify_tool, balance_tool, card_tool]
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        prompt_agent = """You are a friendly banking assistant. You MUST follow these steps:

        1. First, ALWAYS use the IntentClassifier tool to understand the user's intent
        2. Then, based on the classified intent, use the appropriate tool

        Previous conversation:
        {chat_history}

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
            handle_parsing_errors=True,
            memory=memory
        )
        
        return agent_executor

    def text_to_speech(self, text: str):
        """Convert text to speech using Azure TTS"""
        try:
            # Use a friendly banking voice
            ssml = f"""
            <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
                <voice name='en-US-JennyNeural'>
                    <prosody rate="medium" pitch="default">
                        {text}
                    </prosody>
                </voice>
            </speak>
            """
            
            result = self.speech_synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speech_sdk.ResultReason.SynthesizingSpeechCompleted:
                print(f"Assistant: {text}")
                print(f"TTS result reason: {result.reason}")
                return True
            else:
                print(f"TTS failed: {result.reason}")
                return False
                
        except Exception as e:
            print(f"Error in TTS: {e}")
            return False

    def speech_to_text(self) -> str:
        """Convert speech to text using Azure STT"""
        print("Listening... Speak now!")
        
        try:
            result = self.speech_recognizer.recognize_once_async().get()
            
            print(f"Debug - Recognition result: {result.reason}")

            if result.reason == speech_sdk.ResultReason.RecognizedSpeech:
                user_speech = result.text
                print(f"You said: {user_speech}")
                return user_speech
            elif result.reason == speech_sdk.ResultReason.NoMatch:
                print("No speech recognized")
                return ""
            else:
                print(f"Recognition failed: {result.reason}")
                return ""
                
        except Exception as e:
            print(f"Error in STT: {e}")
            return ""

    def process_conversation(self):
        """Process one conversation turn: Listen → Process → Speak"""
        # Step 1: Listen to user
        user_input = self.speech_to_text()
        
        if not user_input or user_input.strip() == "":
            self.text_to_speech("I didn't catch that. Could you please repeat?")
            return
        
        # Check for exit conditions
        if any(exit_word in user_input.lower() for exit_word in ['exit', 'quit', 'stop', 'goodbye']):
            self.text_to_speech("Thank you for banking with us! Have a great day!")
            return "exit"
        
        # Step 2: Process with your banking agent
        try:
            response = self.agent.invoke({"input": user_input})
            assistant_response = response["output"]
        except Exception as e:
            assistant_response = f"I encountered an error processing your request: {str(e)}"
        
        # Step 3: Speak the response
        self.text_to_speech(assistant_response)
        return "continue"

    def text_only_mode(self):
        """Text-only mode for testing"""
        print("\nBanking Assistant Started (Text Mode)")
        print("Type your messages (or 'exit' to quit)")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'stop']:
                print("Assistant: Thank you for banking with us! Have a great day!")
                break
                
            if not user_input:
                continue
                
            try:
                response = self.agent.invoke({"input": user_input})
                print(f"Assistant: {response['output']}")
            except Exception as e:
                print(f"Error: {e}")

# Simple Avatar Display
class BankingAvatar:
    def __init__(self):
        self.expressions = {
            "listening": "Avatar: Listening intently...",
            "thinking": "Avatar: Processing your request...", 
            "speaking": "Avatar: Speaking...",
            "happy": "Avatar: Happy to help!",
            "ready": "Avatar: Ready for banking assistance!",
            "error": "Avatar: Let me check that..."
        }
    
    def show_expression(self, expression: str):
        print(f"\n{self.expressions.get(expression, '🤖 Avatar: Ready!')}")

# Main execution
if __name__ == "__main__":
    # Check if Azure Speech credentials are available
    if not os.getenv("AZURE_SPEECH_KEY") or not os.getenv("AZURE_SPEECH_REGION"):
        print("Azure Speech credentials not found. Running in text-only mode.")
        print("To enable voice, add to your .env file:")
        print("AZURE_SPEECH_KEY=your_key_here")
        print("AZURE_SPEECH_REGION=your_region_here")
        
        avatar = BankingAvatar()
        assistant = VoiceBankingAssistant()
        avatar.show_expression("ready")
        assistant.text_only_mode()
    else:
        avatar = BankingAvatar()
        assistant = VoiceBankingAssistant()
        
        avatar.show_expression("ready")
        print("\nChoose interaction mode:")
        print("1.Voice Conversation")
        print("2.Text Input Only")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            print("\nVoice mode activated. Say 'exit' to end conversation.")
            avatar.show_expression("listening")
            
            while True:
                result = assistant.process_conversation()
                if result == "exit":
                    break
        else:
            avatar.show_expression("ready")
            assistant.text_only_mode()