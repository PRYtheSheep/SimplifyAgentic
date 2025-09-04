import boto3
import os
from dotenv import load_dotenv
import logging

from globals import *

class ClaudeChat:
    def __init__(self, bedrock_client, model_id="anthropic.claude-3-haiku-20240307-v1:0", system_prompt=None):
        self.client = bedrock_client
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.history = []

    def ask(self, user_input, max_tokens=200, temperature=0.5):
        # Add user input
        self.history.append({"role": "user", "content": [{"text": user_input}]})

        # Call Bedrock with system prompt + history
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": self.system_prompt}] if self.system_prompt else None,
            messages=self.history,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature}
        )

        # Extract Claude’s reply
        reply = response["output"]["message"]["content"][0]["text"]

        # Save reply to history
        self.history.append({"role": "assistant", "content": [{"text": reply}]})

        return reply

    def reset(self):
        self.history = []


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS Configuration
session = boto3.Session()
region = os.getenv("REGION", DEFAULT_REGION)
model_id = os.getenv("MODEL_ID", DEFAULT_MODEL)

print(f'Using modelId: {model_id}')
print('Using region: ', region)

bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name=region,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

system_prompt = """
You are a helpful assistant that gathers context from the user regarding a post.

You are to ask for the following context:
1. Source of the post and what is it showing.

Probe for further information if the context provided is insufficient.
DO not preamble when summarising. 

Ask the questions one at a time.
Do not ask questions that are unnecessary.
""" 

chat = ClaudeChat(
    bedrock_client,
    system_prompt=system_prompt
)

output = None
for i in range(2):
    print("---------------------")
    # Starting conversation
    if i == 0:
        entry = input("Provide context for the post\n")
        output = chat.ask(entry)
    else:
        entry = input(output+"\n")
        output = chat.ask(entry)
    
print("---------------------")
chat.ask("Summarise the context provided by the user")
print(chat.history)
