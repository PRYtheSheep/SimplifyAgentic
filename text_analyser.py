import boto3
import json
import asyncio
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
import os
from dotenv import load_dotenv

from globals import *
from Text_agent import analyse_text as raw_query_web
logger = logging.getLogger(__name__)

class TextAnalyzer:
    """
    Analyzes text content for AI-generated characteristics and factual accuracy
    using Amazon Bedrock LLM with web verification capabilities.
    """
    
    def __init__(self, bedrock_client, model_id):
        self.bedrock_client = bedrock_client
        self.model_id = model_id

        self.tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "report_text_analysis",
                        "description": "Reports comprehensive text analysis results including AI-generation detection and factual accuracy assessment",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "ai_score": {
                                        "type": "integer",
                                        "description": "AI-generated score (0-100) where 0=definitely human, 100=definitely AI",
                                        "minimum": 0,
                                        "maximum": 100
                                    },
                                    "fake_score": {
                                        "type": "integer", 
                                        "description": "Fake/accuracy score (0-100) where 0=definitely factual, 100=definitely fake",
                                        "minimum": 0,
                                        "maximum": 100
                                    },
                                    "confidence": {
                                        "type": "integer",
                                        "description": "Confidence in the assessment (0-100)",
                                        "minimum": 0,
                                        "maximum": 100
                                    },
                                    "ai_evidence": {
                                        "type": "array",
                                        "description": "Specific evidence supporting AI-generation assessment",
                                        "items": {
                                            "type": "string"
                                        }
                                    },
                                    "fake_evidence": {
                                        "type": "array",
                                        "description": "Specific evidence supporting fake/accuracy assessment",
                                        "items": {
                                            "type": "string"
                                        }
                                    },
                                    "overall_assessment": {
                                        "type": "string",
                                        "description": "Summary of the overall findings and assessment"
                                    },
                                    "motive_analysis": {
                                        "type": "string",
                                        "description": "Analysis of potential writer motives and biases"
                                    },
                                    "web_verification_summary": {
                                        "type": "string",
                                        "description": "Summary of web verification findings"
                                    }
                                },
                                "required": ["ai_score", "fake_score", "confidence", "ai_evidence", "fake_evidence", "overall_assessment"]
                            }
                        }
                    }
                }
            ]
        }

        self.system_prompt = """
        You are an expert text authenticity analyst. Your role is to analyze text content to determine:
        1. Whether it appears to be AI-generated
        2. Whether it is factually accurate and plausible
        
        <analysis_framework>
        <ai_generation_indicators>
        - Check for overly formal or robotic language patterns
        - Look for repetition of certain phrases or structures
        - Analyze sentence complexity and variation
        - Look for common AI-generated patterns and templates
        </ai_generation_indicators>
        
        <factual_accuracy_indicators>
        - Verify factual claims against known information
        - Check for logical consistency and coherence
        - Look for sensational or exaggerated language
        - Analyze the motive and potential bias of the writer (if any)
        - Consider the context and timing of the information
        </factual_accuracy_indicators>
        
        <scoring_system>
        <ai_score>
        0-100 scale where:
        0 = Definitely human-written
        50 = Uncertain/balanced characteristics
        100 = Definitely AI-generated
        </ai_score>
        
        <fake_score>
        0-100 scale where:
        0 = Definitely factual and accurate
        50 = Mixed/uncertain accuracy
        100 = Definitely fake or misleading
        </fake_score>
        </scoring_system>
        </analysis_framework>
        
        <rules>
        <rule>Be objective and evidence-based in your assessment</rule>
        <rule>Consider that genuine news can sometimes sound formal</rule>
        <rule>Think step-by-step</rule>
        <rule>Do not automatically assume everything is fake or AI-generated</rule>
        <rule>Consider the potential motive and context of the content</rule>
        <rule>Provide specific evidence for your scores</rule>
        <rule>Use web verification for factual claims when available</rule>
        <rule>ALWAYS use the report_text_analysis tool to provide structured output</rule>
        </rules>
        
        <output_format>
        Return JSON with:
        - ai_score: 0-100
        - fake_score: 0-100  
        - ai_evidence: array of specific indicators found
        - fake_evidence: array of specific factual issues found
        - confidence: 0-100 confidence in assessment
        - web_verification_results: results from web fact-checking
        - overall_assessment: A summary of the findings
        </output_format>
        """
    
    async def analyze_text(self, text_content: str) -> Dict[str, Any]:
        """
        Analyze text for AI-generation and factual accuracy using Bedrock LLM
        with structured tool output to ensure consistent JSON format.
        """
        try:
            # Perform web verification for factual claims
            # Get back Travlie JSON file
            web_verification_results = await self._verify_via_web(text_content)
            
            # Prepare analysis query for Bedrock with tool enforcement
            analysis_query = f"""
            Analyze the following text content for authenticity. You MUST use the report_text_analysis tool.
            
            TEXT TO ANALYZE:
            {text_content}
            
            WEB VERIFICATION CONTEXT:
            {json.dumps(web_verification_results, indent=2)}
            
            Provide a comprehensive analysis including:
            1. AI-generated score (0-100) with specific evidence
            2. Fake/accuracy score (0-100) with specific evidence  
            3. Confidence level in your assessment (0-100)
            4. Consideration of writer's potential motive and context
            5. Integration of web verification results
            
            Remember: You MUST use the report_text_analysis tool for your response.
            """
            
            # Call Bedrock for analysis with tool enforcement
            analysis_result = await self._call_bedrock_with_tool(analysis_query)
            
            return {
                **analysis_result,
                "web_verification": web_verification_results,
                "analysis_timestamp": datetime.now().isoformat(),
                "text_length": len(text_content),
                "status": "analysis_complete"
            }
            
        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            return {
                "ai_score": 50,
                "fake_score": 50,
                "confidence": 0,
                "ai_evidence": ["Analysis failed due to error"],
                "fake_evidence": ["Analysis failed due to error"],
                "error": str(e),
                "status": "analysis_failed"
            }
    
    async def _call_bedrock_with_tool(self, query: str) -> Dict[str, Any]:
        """Call Bedrock LLM with tool enforcement for structured JSON output"""
        messages = [{"role": "user", "content": [{"text": query}]}]
        
        converse_api_params = {
            "modelId": self.model_id,
            "system": [{"text": self.system_prompt}],
            "messages": messages,
            "inferenceConfig": {
                "temperature": 0.1,
                "maxTokens": 2000
            },
            "toolConfig": self.tool_config,
        }
        
        response = self.bedrock_client.converse(**converse_api_params)
        
        # Parse the tool response for structured data
        print(f'\n\n\nResponse by LLM:\n{response}\n\n\n\n')
        return self._parse_tool_response(response)

    
    def _parse_tool_response(self, response: Dict) -> Dict[str, Any]:
        """Parse tool response to extract structured analysis results"""
        try:
            content_blocks = response["output"]["message"]["content"]
            
            for block in content_blocks:
                if "toolUse" in block:
                    tool_use = block["toolUse"]
                    if tool_use["name"] == "report_text_analysis":
                        # Return the structured input from the tool
                        return tool_use["input"]
            
            # If no tool was used (shouldn't happen with our system prompt)
            logger.warning("No tool usage found in response, using fallback")
            return self._get_structured_fallback_response("No tool usage in response")
            
        except Exception as e:
            logger.error(f"Failed to parse tool response: {e}")
            return self._get_structured_fallback_response(str(e))
    
    def _get_structured_fallback_response(self, error_msg: str = "") -> Dict[str, Any]:
        """Return a structured fallback response when analysis fails"""
        return {
            "ai_score": 50,
            "fake_score": 50,
            "confidence": 0,
            "ai_evidence": [f"Analysis system error: {error_msg}"],
            "fake_evidence": [f"Analysis system error: {error_msg}"],
            "overall_assessment": "Analysis failed - manual review recommended",
            "motive_analysis": "Unable to analyze motives due to system error",
            "web_verification_summary": "Web verification not completed due to error"
        }
    
    async def _verify_via_web(self, text_content: str) -> Dict[str, Any]:
        """
        Placeholder for web verification functionality.
        This would integrate with fact-checking APIs, search engines, or databases.
        """
        evidence = raw_query_web(text_content)
        formatted_evidence = []
        
        for result in evidence:
            claim = result["claim"]
            
            # Extract the most relevant evidence snippets
            evidence_snippets = []
            for i, (title, content, url) in enumerate(zip(result["titles"], result["contents"], result["urls"])):
                if i < 3:  # Limit to top 3 most relevant results per claim
                    evidence_snippets.append({
                        "source": url,
                        "title": title,
                        "content_snippet": content[:200] + "..." if len(content) > 200 else content,
                        "relevance_score": f"Result {i+1} of {len(result['titles'])}"
                    })
            
            formatted_evidence.append({
                "claim_to_verify": claim,
                "evidence_summary": result["summary"]
            })

        with open('web_evidence.json', 'w') as f:
            json.dump(formatted_evidence, f, indent=2)
        
        return formatted_evidence
        
    async def analyze_with_motive_assessment(self, text_content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced analysis that includes motive assessment and context awareness
        using tool functionality for structured output.
        """
        if context is None:
            context = {}
        
        motive_query = f"""
        Analyze the following text with additional context for motive assessment.
        You MUST use the report_text_analysis tool.
        
        TEXT TO ANALYZE:
        {text_content}
        
        ADDITIONAL CONTEXT:
        {json.dumps(context, indent=2)}
        
        Please provide a comprehensive analysis including motive assessment and context awareness.
        """
        
        # Perform analysis with enhanced context
        analysis_result = await self._call_bedrock_with_tool(motive_query)
        
        # Add context metadata
        analysis_result["context_aware_analysis"] = True
        analysis_result["context_provided"] = bool(context)
        
        return analysis_result

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Fake news example for testing
    fake_news_text = """
    BREAKING: Scientists at Harvard University have discovered that drinking 5 cups of coffee daily 
    can reverse aging and extend human lifespan by 25 years. The groundbreaking study, published 
    in the New England Journal of Medicine, reveals that coffee contains a previously unknown 
    compound called "caffeinol" that repairs DNA damage. President Biden has already announced 
    a national coffee initiative, and Starbucks shares have surged by 300% in pre-market trading. 
    Doctors worldwide are calling this the most significant medical breakthrough since penicillin.
    """

    async def main():
        logger.info("=== STARTING TEXT ANALYZER TEST ===")
        logger.info(f"Testing with text length: {len(fake_news_text)} characters")

        load_dotenv()

        # AWS Configuration
        session = boto3.Session()
        region = os.getenv("REGION", DEFAULT_REGION)
        model_id = os.getenv("MODEL_ID", DEFAULT_MODEL)

        print(f'Using modelId: {model_id}')
        print(f'Using region: {region}')
        
        try:
            # Initialize AWS Bedrock client
            logger.info("Initializing Bedrock client...")
            bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            
            # Create TextAnalyzer instance
            logger.info("Creating TextAnalyzer instance...")
            analyzer = TextAnalyzer(bedrock_client, model_id)
            
            # Test main analysis (only calling existing function)
            logger.info("Calling analyze_text function...")
            start_time = datetime.now()
            
            result = await analyzer.analyze_text(fake_news_text)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Analysis completed in {duration:.2f} seconds")
            
            # Log results
            logger.info("=== ANALYSIS RESULTS ===")
            logger.info(f"Status: {result.get('status', 'unknown')}")
            logger.info(f"AI Score: {result.get('ai_score', 'N/A')}/100")
            logger.info(f"Fake Score: {result.get('fake_score', 'N/A')}/100")
            logger.info(f"Confidence: {result.get('confidence', 'N/A')}/100")
            
            if 'ai_evidence' in result:
                logger.info("AI Evidence:")
                for evidence in result['ai_evidence']:
                    logger.info(f"  - {evidence}")
            
            if 'fake_evidence' in result:
                logger.info("Fake Evidence:")
                for evidence in result['fake_evidence']:
                    logger.info(f"  - {evidence}")
            
            if 'overall_assessment' in result:
                logger.info(f"Overall: {result['overall_assessment']}")
            
            # Save full results to file
            with open('analysis_results.json', 'w') as f:
                json.dump(result, f, indent=2)
            logger.info("Full results saved to analysis_results.json")
            
            logger.info("=== TEST COMPLETED ===")
            
        except Exception as e:
            logger.error(f"TEST FAILED: {e}")
            logger.error("Traceback:", exc_info=True)
    
    # Run the test
    asyncio.run(main())