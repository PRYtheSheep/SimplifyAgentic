import json
import logging
import os
import boto3 
from botocore.exceptions import ClientError

from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import http_request

logging.getLogger("strands").setLevel(logging.INFO)
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s", 
    handlers=[logging.StreamHandler()],
    level = logging.INFO
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
import time

TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', None)
# print(TAVILY_API_KEY)
# print(f"access key: {os.getenv("AWS_ACCESS_KEY_ID", None)}")

bedrock_model = BedrockModel(
    # model_id="us.amazon.nova-lite-v1:0",
    # model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    # model_id = "anthropic.claude-3-haiku-20240307-v1:0",
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0.1,
    region_name="ap-southeast-1"
)

if TAVILY_API_KEY:
    from tavily import TavilyClient
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    # SYSTEM_PROMPT = TAVILY_SYSTEM_PROMPT
    logger.info("Tavily API found. Using the Tavily API for search queries")
else:
    # from duckduckgo_search import DDGS
    from ddgs import DDGS
    # SYSTEM_PROMPT = DUCKDUCKGO_SYSTEM_PROMPT
    logger.info("Tavily API not found. Using the Duck Duck Go API for search queries")


@tool
def web_search(query: str, max_results: int = 5):
    """
    Perform an internet search with the specified query
    
    Args:
        query (str): A question or search phrase to perform a search with
        max_results (int): Number of results to return for each web search
        
    Returns:
        summary (str): Summary of the contents of the searched URLs.
        titles (list of str): List of titles from the search results.
        urls (list of str): List of URLs from the search results.
    """
    print("*"*20)
    print(f"query: {query}")
    print(f"max_results: {max_results}")
    print("*"*20)
    domains = ["straitstimes.com/singapore","channelnewsasia.com/singapore","mothership.sg","todayonline.com"]
    # exclude_tags = ["listen","tag","topic"]
    # domains_excluded = [i+"/"+j for i in domains for j in exclude_tags]
    # print(domains_excluded)
    if TAVILY_API_KEY:
        response = tavily_client.search(
            query,
            max_results = max_results,
            time_range = "month",
            include_domains=domains,
            exclude_domains = ["youtube.com","instagram.com"],
            include_answer = "advanced",
            search_depth = "advanced"
            # include_raw_content = True
            # include_images = True
            # start_date = "2025-01-01",
            # end_date = "2025-02-01" 
        )
    else:
        response = DDGS().text(
            "python programming",
            max_results = max_results
        )
    
    summary = response["answer"]
    urls=[]
    titles = []
    contents = []
    scores = []
    for article in response["results"]:
        if article["score"]>0.6:
            urls.append(article["url"])
            titles.append(article["title"])
            contents.append(article["content"])
            scores.append(article["score"])
    # print(type(response))
    # print(len(response))
    # print(response)
    # print(json.dumps(response, indent=2, default=str))
    # print(summary, titles, urls)
    return summary, titles, urls

# class TextAgentModel:
#     def __init__(self, model_id: str, region_name: str = "ap-southeast-1"):
#         self.model_id = model_id
#         self.bedrock_client = boto3.client('bedrock-runtime', region_name=region_name)
#     def generate_search_query(self,evidence):
#         request_body = 
#         try:
#             response = self.bedrock_client.invoke_model(
#                     modelId=self.model_id,
#                     body=json.dumps(request_body),
#                     contentType='application/json'
#                 )
#         except Exception as e:
#             return f"Error: {str(e)}"
def analyse_text( prompt):
    # SYSTEM_PROMPT = """
    #     You are a text analyst agent whose goal is to receive provided details.
        
    #     1. Determine if the input is real or falsified
    #     2. If you wish to perform a web search, generate different prompts
    #     3. Use your research tools (web_search, http_request,) to find relevant information
    #     4. Include source URLs and keep findings under 500 words
    # """
    SYSTEM_PROMPT = """
    Firstly, perform an analysis of the input content.
    1. Extract all factual claims, statistics, dates, names, events, and assertions from the chat
    2. Categorize claims by type (historical facts, current events, statistics, quotes, etc.)
    3. Identify claims that are most critical to verify

    Secondly, for each unsubstantiated claim, generate tp to 3 search queries with the following guidelines:
    1. Use specific, fact-focused keywords
    2. Include dates, names, and context when relevant
    3. Vary phrasing and sentence structure for different queries

    Next, use the web-search tool to obtain evidence from reputable sources.
    Finally, for each claim, assign one of these categories with reasons for the classification:

    VERIFIED: Confirmed by reliable sources
    FALSE: Contradicted by reliable evidence
    MISLEADING: Partially true but lacking context or nuance
    UNVERIFIABLE: Insufficient evidence available
    OPINION: Subjective statement, not factual claim

    
    For each significant claim, return output in the following format:
    Claim: [Quote the exact statement]
    Status: [VERIFIED/FALSE/MISLEADING/UNVERIFIABLE/OPINION]
    Evidence: [Brief explanation with source citations]
    Source URLs: [List relevant URLs]

"""
    text_agent = Agent(
        model = bedrock_model,
        system_prompt = SYSTEM_PROMPT,
        tools = [web_search, http_request],
        callback_handler = None
    )
    response = text_agent(
        f"Analyse this evidence {prompt}"
        )
    print(str(response))
        

if __name__ == "__main__":
    # summary, titles, urls = web_search("SengKang Green Primary School bullying")
    # print(f"summary: {summary}")
    # print(f"titles: {titles}")
    # print(f"urls: {urls}")
    analyse_text(
    """A primary 6 student was bullied in Sengkang Green primary school. 
    Perform a maximum of 2 web searches""")

