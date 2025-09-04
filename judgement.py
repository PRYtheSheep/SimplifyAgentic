import boto3
import json
import asyncio
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

from globals import *
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