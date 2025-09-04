import boto3
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
from dotenv import load_dotenv

from globals import *
logger = logging.getLogger(__name__)

class JudgementBot:
    """
    Final assessment bot that analyzes all tool outputs to determine 
    if media is fake/AI-generated and provides comprehensive scoring.
    """
    
    def __init__(self, bedrock_client, model_id):
        self.bedrock_client = bedrock_client
        self.model_id = model_id
        
        self.system_prompt = """
        You are an expert media authenticity judge. Your role is to analyze comprehensive analysis results 
        from multiple specialized tools and provide a final assessment on whether the media is AI-generated or fake.

        <scoring_system>
        <ai_score>
        0-100 scale where:
        0 = Definitely human-created/organic
        25 = Mostly human with some AI assistance
        50 = Balanced/unclear
        75 = Mostly AI-generated with human editing
        100 = Definitely AI-generated
        </ai_score>

        <fake_score>
        0-100 scale where:
        0 = Definitely authentic/truthful
        25 = Mostly authentic with minor inaccuracies
        50 = Mixed authenticity/unverifiable
        75 = Mostly fake/misleading with some truth
        100 = Definitely fake/deceptive
        </fake_score>

        <confidence_score>
        0-100 scale indicating certainty in assessment
        </confidence_score>
        </scoring_system>

        <assessment_framework>
        1. Review each given context thoroughly
        2. Look for consistency across different analysis types
        3. Take the summary from the web queries as ground truth
        4. Weight more technical/objective evidence higher than subjective analysis
        5. Consider context and potential motives if available
        6. Quote specific evidence that was provided if it is relevant in the assessment
        7. Do not be overly eager to classify something as fake / AI-generated
        </assessment_framework>

        <output_requirements>
        - Give the following in a singular concise text message:
            - MUST give overall summary judgment
            - MUST include specific evidence from each analysis component
            - MUST provide both ai_score and fake_score (0-100)
            - MUST provide confidence score (0-100)
            - MUST explain reasoning for scores
        </output_requirements>

        <important_rules>
        <rule>Be objective and evidence-based - don't guess</rule>
        <rule>Consider that genuine content can sometimes have AI-like characteristics</rule>
        <rule>Weight technical analysis higher than subjective interpretation</rule>
        <rule>If evidence is conflicting, acknowledge this in confidence score</rule>
        <rule>Provide specific examples from the analysis data</rule>
        </important_rules>
        """

        self.tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "report_final_judgement",
                        "description": "Report comprehensive final assessment with AI and fake scores",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "ai_score_0_100": {
                                        "type": "integer",
                                        "description": "AI-generated likelihood score (0=human, 100=AI)",
                                        "minimum": 0,
                                        "maximum": 100
                                    },
                                    "fake_score_0_100": {
                                        "type": "integer",
                                        "description": "Fake/deceptive content score (0=authentic, 100=fake)",
                                        "minimum": 0,
                                        "maximum": 100
                                    },
                                    "confidence_0_100": {
                                        "type": "integer",
                                        "description": "Confidence in the assessment",
                                        "minimum": 0,
                                        "maximum": 100
                                    },
                                    "summary": {
                                        "type": "string",
                                        "description": "Comprehensive summary of findings and judgment"
                                    },
                                    "key_evidence": {
                                        "type": "array",
                                        "description": "Most compelling evidence supporting the assessment",
                                        "items": {"type": "string"}
                                    },
                                    "component_analysis": {
                                        "type": "object",
                                        "description": "Breakdown of findings by analysis component",
                                        "properties": {
                                            "audio": {"type": "string"},
                                            "visual": {"type": "string"},
                                            "text": {"type": "string"},
                                            "technical": {"type": "string"}
                                        }
                                    },
                                    "recommendations": {
                                        "type": "array",
                                        "description": "Recommended actions or further investigations",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["ai_score_0_100", "fake_score_0_100", "confidence_0_100", "summary", "key_evidence"]
                            }
                        }
                    }
                }
            ]
        }

    async def final_assessment(self, analysis_data: Dict[str, Any], image_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze comprehensive tool outputs and determine final authenticity assessment.
        
        Args:
            analysis_data: JSON containing all tool outputs and analysis results
            image_paths: Optional list of image paths for visual reference
            
        Returns:
            Comprehensive assessment with AI score, fake score, and detailed evidence
        """
        try:
            # Prepare the analysis query for Bedrock
            assessment_query = self._prepare_assessment_query(analysis_data, image_paths)
            
            # Call Bedrock for final judgement
            final_judgement = await self._call_bedrock_for_judgement(assessment_query)
            
            return {
                **final_judgement,
                "assessment_timestamp": datetime.now().isoformat(),
                "components_analyzed": list(analysis_data.keys()),
                "status": "assessment_complete"
            }
            
        except Exception as e:
            logger.error(f"Final assessment failed: {e}")
            return {
                "ai_score_0_100": 50,
                "fake_score_0_100": 50,
                "confidence_0_100": 0,
                "summary": f"Assessment failed due to error: {str(e)}",
                "key_evidence": ["Assessment system error occurred"],
                "error": str(e),
                "status": "assessment_failed"
            }

    def _prepare_assessment_query(self, analysis_data: Dict[str, Any], image_paths: List[str]) -> str:
        """Prepare comprehensive assessment query with all analysis data"""
        query = """
        Analyze the following comprehensive media analysis results and provide a final judgement on authenticity.
        You MUST use the report_final_judgement tool with structured output.

        COMPREHENSIVE ANALYSIS DATA:
        """
        
        # Add each component's analysis results
        for tool_name, results in analysis_data.items():
            query += f"\n\n--- {tool_name.upper()} RESULTS ---\n"
            query += json.dumps(results, indent=2, ensure_ascii=False)
        
        # Add image context if available
        if image_paths:
            query += f"\n\n--- IMAGES ANALYZED ({len(image_paths)} frames) ---\n"
            query += f"Image paths: {', '.join(image_paths)}"
            query += "\nConsider the visual analysis results in context of these images."
        
        query += """
        
        <assessment_instructions>
        1. Review ALL analysis components thoroughly
        2. Identify the strongest evidence from each component
        3. Look for consistency or contradictions across different analyses
        4. Determine overall AI-generation likelihood (ai_score)
        5. Determine overall authenticity/fakeness (fake_score)
        6. Assess confidence based on evidence strength and consistency
        7. Provide specific evidence examples in key_evidence
        8. Summarize findings comprehensively
        9. MUST use report_final_judgement tool for response
        </assessment_instructions>
        """
        
        return query

    async def _call_bedrock_for_judgement(self, query: str) -> Dict[str, Any]:
        """Call Bedrock with the assessment query and extract final judgement"""
        converse_api_params = {
            "modelId": self.model_id,
            "system": [{"text": self.system_prompt}],
            "messages": [{"role": "user", "content": [{"text": query}]}],
            "inferenceConfig": {
                "temperature": 0.1,  # Low temperature for consistent scoring
                "maxTokens": 1500,   # Enough for comprehensive assessment
                "topP": 0.9
            },
            "toolConfig": self.tool_config,
        }

        try:
            response = self.bedrock_client.converse(**converse_api_params)
            return self._extract_final_judgement(response)
            
        except Exception as e:
            logger.error(f"Bedrock judgement call failed: {e}")
            raise RuntimeError(f"Final assessment failed: {e}")

    def _extract_final_judgement(self, response: Dict) -> Dict[str, Any]:
        """Extract the final judgement from Bedrock response"""
        for content_block in response["output"]["message"]["content"]:
            if "toolUse" in content_block:
                tool_use = content_block["toolUse"]
                if tool_use["name"] == "report_final_judgement":
                    return tool_use["input"]
        
        # If no tool use found, try to parse the response content
        judgement = {
            "ai_score_0_100": 50,
            "fake_score_0_100": 50,
            "confidence_0_100": 0,
            "summary": "Could not extract structured judgement from response",
            "key_evidence": ["Assessment system parsing error"],
            "raw_response": response
        }
        
        # Try to find scores in text response as fallback
        response_text = self._extract_response_text(response)
        if response_text:
            judgement["summary"] = response_text[:1000]  # Truncate if too long
            
        return judgement

    def _extract_response_text(self, response: Dict) -> str:
        """Extract text content from response as fallback"""
        text_parts = []
        for content_block in response["output"]["message"]["content"]:
            if "text" in content_block:
                text_parts.append(content_block["text"])
        return "\n".join(text_parts)