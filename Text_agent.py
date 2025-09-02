import json
import logging
import os
from datetime import datetime
import re

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
    model_id = "anthropic.claude-3-haiku-20240307-v1:0",
    # model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0.1,
    region_name="ap-southeast-1",
    top_p=0.9,
    # max_tokens = 300
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
def web_search(query: str, max_results: int = 3):
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
    print(f"search start: {datetime.now()}")
    print("*"*20)
    domains = ["straitstimes.com/singapore","channelnewsasia.com/singapore","mothership.sg","todayonline.com"]
    # exclude_tags = ["listen","tag","topic"]
    # domains_excluded = [i+"/"+j for i in domains for j in exclude_tags]
    # print(domains_excluded)
    if TAVILY_API_KEY:
        try:
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
                # start_date = "2025-08-01",
                # end_date = "2025-08-30" 
            )
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
    else:
        response = DDGS().text(
            "python programming",
            max_results = max_results
        )
    
    urls=[]
    titles = []
    contents = []
    domains = []
    for article in response["results"]:
        if article["score"]>0.6:
            for domain in domains:
                pattern = re.escape(domain)
                if re.search(pattern, article["url"]):
                    domains.append(domain)
            urls.append(article["url"])
            titles.append(article["title"])
            contents.append(article["content"])
    articledict = {}
    articledict["summary"] = response["answer"]
    articledict["domains"] = domains
    articledict["urls"] = urls
    articledict["titles"] = titles
    articledict["contents"] = contents
    # print(type(response))
    # print(len(response))
    # print(response)
    print(json.dumps(response, indent=2, default=str))
    # print(summary, titles, urls)
    print(f"search done: {datetime.now()}")
    # domain title content [str] , [str] , [str]
    return articledict

def analyse_text( prompt):

    """
    Extracts factual claims from evidence.
    Generates web search queries. 
    Calls web_search function to run queries
    
    Args:
        query (str): A question or search phrase to perform a search with
        max_results (int): Number of results to return for each web search
        
    Returns:
        verified_evidence (List(dict)) : List containing a dictionary for each claim extracted
    """

    SYSTEM_PROMPT1 = """
    Extract factual claims. 
    List each in <claim1></claim1>, <claim2></claim2> format.
    Suggest possible web search queries for each claim.
    List them in <query1></query1>, <query2></query2> format. 
    
    """

    extraction_agent = Agent(
        model = bedrock_model,
        system_prompt = SYSTEM_PROMPT1,
        tools = [],
        callback_handler = None
    )
    extracted_response = extraction_agent(
        f"Analyse this {prompt}"
        )
    # print(type(extracted_response))
    print(str(extracted_response))

    pattern = re.compile(r"<claim(\d+)>(.*?)</claim\1>\s*<query\1>(.*?)</query\1>", re.DOTALL)

    claims = []
    queries = []
    for match in pattern.findall(str(extracted_response)):
        idx, claim, query = match
        claims.append(claim.strip())
        queries.append(query.strip())
    # print(claims)
    # print(queries)
    verified_evidence = []
    for i in range(len(queries)):
        web_result = web_search(queries[i])
        web_result["claim"] = claims[i]
        verified_evidence.append(web_result)
    return verified_evidence



    # response = text_agent(
    #     f"These are the claims: {str(extracted_response)}"
    #     )
    # print(str(response))
        

if __name__ == "__main__":

    start_time = datetime.now()
    print(f"Start Time: {start_time}")
    strlist = ["A primary 3 student was bullied in Sengkang Green primary school. ",
               "A chinese student was bullied in Sengkang Green primary school.",
               "Singaporeans received CDC vouchers in 2025."]
    # for i in strlist:
    #     analyse_text(
    #     i)
    # analyse_text(strlist[0])
    print(analyse_text(strlist[0] + strlist[1] + strlist[2]))
    # A chinese student was bullied in Sengkang Green primary school.
    # Singaporeans received CDC vouchers in 2025.
    # print(web_search("A secondary 2 student was bullied in Sengkang Green primary school."))
    print("Elapsed time:", (datetime.now()-start_time).total_seconds())

