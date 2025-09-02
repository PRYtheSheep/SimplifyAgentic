import boto3
import json
import os
import tempfile
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import ffmpeg
import cv2
import numpy as np
import asyncio
from text_analyser import TextAnalyzer
import whisper
import torch

from globals import *

# Torch and cuda avalibility
print(torch.__version__)
print(torch.version.cuda)

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

class MediaAnalysisOrchestrator:
    """
    Orchestrator bot that analyzes media files for AI-generated/fake content.
    Uses AWS Bedrock for LLM capabilities and coordinates specialized analysis.
    """
    
    def __init__(self):
        self.bedrock_client = bedrock_client
        self.model_id = model_id
        
        # Define tools for the orchestrator
        self.tools = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "extract_audio_and_frames",
                        "description": "Extract audio as MP3 and representative frames from a video file",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "video_path": {
                                        "type": "string",
                                        "description": "Path to the input video file (.mp4)"
                                    },
                                    "frames_count": {
                                        "type": "integer",
                                        "description": "Number of representative frames to extract (default: 5)",
                                        "default": 5,
                                        "minimum": 1,
                                        "maximum": 20
                                    }
                                },
                                "required": ["video_path"]
                            }
                        }
                    }
                },
                {
                    "toolSpec": {
                        "name": "analyze_audio",
                        "description": "Transcribe audio file to text",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "audio_path": {
                                        "type": "string",
                                        "description": "Path to the audio file to analyze"
                                    }
                                },
                                "required": ["audio_path"]
                            }
                        }
                    }
                },
                {
                    "toolSpec": {
                        "name": "analyze_image",
                        "description": "Analyze image file for AI-generated or fake characteristics",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "image_path": {
                                        "type": "string",
                                        "description": "Path to the image file to analyze"
                                    }
                                },
                                "required": ["image_path"]
                            }
                        }
                    }
                },
                {
                    "toolSpec": {
                        "name": "analyze_text",
                        "description": "Analyze text content for fake news or AI-generated characteristics",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "text_content": {
                                        "type": "string",
                                        "description": "Text content to analyze"
                                    }
                                },
                                "required": ["text_content"]
                            }
                        }
                    }
                },
                {
                    "toolSpec": {
                        "name": "report_final_assessment",
                        "description": "Report final assessment of media authenticity",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "fake_score_0_100": {
                                        "type": "integer",
                                        "description": "0 = definitely genuine, 100 = definitely fake",
                                        "minimum": 0,
                                        "maximum": 100
                                    },
                                    "confidence_0_100": {
                                        "type": "integer",
                                        "description": "Confidence in the assessment",
                                        "minimum": 0,
                                        "maximum": 100
                                    },
                                    "decision": {
                                        "type": "string",
                                        "enum": ["Likely genuine", "Uncertain", "Likely fake"],
                                        "description": "Overall categorical judgment"
                                    },
                                    "evidence": {
                                        "type": "array",
                                        "description": "Evidence supporting the assessment",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "component": {"type": "string", "description": "Audio/Image/Text component"},
                                                "finding": {"type": "string", "description": "What was found"},
                                                "confidence": {"type": "integer", "description": "Confidence in this finding"}
                                            }
                                        }
                                    },
                                    "notes": {
                                        "type": "string",
                                        "description": "Additional remarks about the assessment"
                                    }
                                },
                                "required": ["fake_score_0_100", "confidence_0_100", "decision", "evidence"]
                            }
                        }
                    }
                }
            ]
        }

        self.system_prompt = """
        You are a media authenticity analysis orchestrator. Your role is to analyze media files and determine if they are AI-generated or fake news.

        <workflow>
        <step1>Receive media files (videos, images, or text)</step1>

        <step2>Decompose media into appropriate components:</step2>
        <decision_tree>
        <case type="video">
            <action>Use extract_audio_and_frames tool first</action>
            <substeps>
                <substep>Extract audio component from video</substep>
                <substep>Extract representative frames from video</substep>
            </substeps>
        </case>
        <case type="image">
            <action>Proceed directly to image analysis</action>
        </case>
        <case type="text">
            <action>Proceed directly to text analysis</action>
        </case>
        </decision_tree>

        <step3>Coordinate analysis of each component using specialized tools:</step3>
        <analysis_plan>
        <if case="video">
            <component type="audio">
                <tool>analyze_audio</tool>
                <input>Extracted audio file from extract_audio_and_frames</input>
            </component>
            <component type="frames">
                <tool>analyze_image</tool>
                <input>Each extracted frame file from extract_audio_and_frames</input>
            </component>
        </if>
        <if case="image">
            <component type="image">
                <tool>analyze_image</tool>
                <input>Original image file</input>
            </component>
        </if>
        <if case="text">
            <component type="text">
                <tool>analyze_text</tool>
                <input>Text content</input>
            </component>
        </if>
        </analysis_plan>

        <step4>Synthesize results into final authenticity assessment:</step4>
        <final_step>
            <tool>report_final_assessment</tool>
            <requirements>
                <requirement>fake_score_0_100 (0=genuine, 100=fake)</requirement>
                <requirement>confidence_0_100</requirement>
                <requirement>decision (Likely genuine/Uncertain/Likely fake)</requirement>
                <requirement>evidence array with findings from all analyses</requirement>
            </requirements>
        </final_step>
        </workflow>

        <rules>
        <rule>Only use tools that are necessary for the specific media type</rule>
        <rule>For videos, you MUST call extract_audio_and_frames first before any analysis</rule>
        <rule>Always provide a comprehensive final assessment using report_final_assessment</rule>
        <rule>If any component analysis fails or returns low confidence, reflect this in the final assessment</rule>
        <rule>Consider all available evidence when making the final determination</rule>
        </rules>

        <output_format>
        <final_output>Must use report_final_assessment tool with all required parameters</final_output>
        <evidence_format>Include specific findings from each analysis component</evidence_format>
        </output_format>
        """

    async def extract_audio_and_frames(self, video_path: str, frames_count: int = 5) -> Tuple[str, List[str]]:
        """
        Extract audio as MP3 and representative frames from a video file.
        Returns tuple of (audio_path, list_of_frame_paths)
        """
        # Validate input
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if frames_count <= 0:
            raise ValueError("frames_count must be positive")
        
        # Ensure output directory exists
        output_dir = Path(OUTPUT_PATH)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Video path: {video_path}")
        # Extract audio
        audio_path = str(output_dir / "extracted_audio.mp3")
        try:
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec='libmp3lame', audio_bitrate='192k')
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error during audio extraction: {e.stderr.decode()}")
            raise RuntimeError(f"Audio extraction failed: {e.stderr.decode()}")
        
        # Extract representative frames
        logger.info(f"Extracting frames")
        frame_paths = await self._extract_representative_frames(video_path, output_dir, frames_count)
        
        return audio_path, frame_paths

    async def _extract_representative_frames(self, video_path: str, output_dir: Path, frames_count: int) -> List[str]:
        """Extract representative frames from video using smart sampling and color change detection"""
        frame_paths = []
        
        # Apply global frame limit
        frames_count = min(frames_count, MAX_FRAMES_LIMIT)
        logger.info(f"Extracting up to {frames_count} frames (global limit: {MAX_FRAMES_LIMIT})")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("Could not open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Use provided FPS or default to 60 if unavailable
            if fps <= 0:
                fps = 60
                logger.warning(f"FPS not available, assuming {fps} FPS")
            
            logger.info(f"Video properties: FPS={fps}, Total frames={total_frames}, Duration={duration:.2f}s")
            
            # Calculate sampling interval (every 0.5 seconds)
            frames_per_half_second = max(1, int(fps * 0.5))
            logger.info(f"Sampling every {frames_per_half_second} frames (~0.5 seconds)")
            
            # Extract frames with smart sampling and color change detection
            previous_frame = None
            color_change_scores = []
            sampled_frames = []
            
            frame_idx = 0
            while frame_idx < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Calculate color change score compared to previous frame
                current_score = 0
                if previous_frame is not None:
                    # Convert to HSV for better color difference detection
                    current_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    previous_hsv = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2HSV)
                    
                    # Calculate mean color difference in HSV space
                    color_diff = np.mean(np.abs(current_hsv.astype(np.float32) - previous_hsv.astype(np.float32)))
                    current_score = color_diff
                
                color_change_scores.append((frame_idx, current_score, frame))
                previous_frame = frame.copy()
                
                # Move to next sampling point
                frame_idx += frames_per_half_second
            
            cap.release()
            
            if not color_change_scores:
                raise RuntimeError("No frames could be extracted from the video")
            
            # Sort frames by color change score (most significant changes first)
            color_change_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select top frames with most significant color changes, but ensure temporal distribution
            selected_frames = self._select_diverse_frames(color_change_scores, frames_count, total_frames)
            
            # Save selected frames
            for i, (frame_idx, score, frame) in enumerate(selected_frames):
                frame_path = str(output_dir / f"frame_{i:03d}_idx{frame_idx}_score{score:.2f}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                logger.info(f"Saved frame {i}: index={frame_idx}, color_change_score={score:.2f}")
            
            return frame_paths
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            raise

    def _select_diverse_frames(self, scored_frames: List[Tuple[int, float, np.ndarray]], 
                            max_frames: int, total_frames: int) -> List[Tuple[int, float, np.ndarray]]:
        """Select frames that are both visually significant and temporally diverse"""
        selected = []
        
        # Divide video into temporal segments
        num_segments = min(max_frames, len(scored_frames))
        segment_size = total_frames / num_segments if total_frames > 0 else 1
        
        # For each segment, pick the frame with highest color change score
        for segment in range(num_segments):
            segment_start = segment * segment_size
            segment_end = (segment + 1) * segment_size
            
            # Find frames in this temporal segment
            segment_frames = [
                (idx, score, frame) for idx, score, frame in scored_frames
                if segment_start <= idx < segment_end
            ]
            
            if segment_frames:
                # Pick the frame with highest color change score in this segment
                best_frame = max(segment_frames, key=lambda x: x[1])
                selected.append(best_frame)
            else:
                # If no frames in segment, pick the highest scored frame overall
                if scored_frames and len(selected) < max_frames:
                    remaining_frames = [f for f in scored_frames if f not in selected]
                    if remaining_frames:
                        best_remaining = max(remaining_frames, key=lambda x: x[1])
                        selected.append(best_remaining)
        
        # If we still need more frames, add the next highest scored ones
        if len(selected) < max_frames:
            remaining = [f for f in scored_frames if f not in selected]
            remaining.sort(key=lambda x: x[1], reverse=True)
            selected.extend(remaining[:max_frames - len(selected)])
        
        # Sort selected frames by temporal order for better organization
        selected.sort(key=lambda x: x[0])
        
        return selected[:max_frames]
    
    async def analyze_media(self, media_path: str, media_type: str) -> Dict[str, Any]:
        """
        Main analysis function that uses Bedrock to coordinate media analysis
        """
        try:
            if not os.path.exists(media_path):
                raise FileNotFoundError(f"Media file not found: {media_path}")
            
            # Prepare query for Bedrock
            query = f"""
            Analyze this {media_type} file for authenticity: {media_path}
            
            Please:
            1. If it's a video, extract audio and frames first
            2. Analyze each component using the appropriate tools
            3. Provide a comprehensive final assessment
            
            Media type: {media_type}
            File path: {media_path}
            """

            messages = [{"role": "user", "content": [{"text": query}]}]

            converse_api_params = {
                "modelId": self.model_id,
                "system": [{"text": self.system_prompt}],
                "messages": messages,
                "inferenceConfig": {
                    "temperature": 0.1,
                    "maxTokens": 1000
                },
                "toolConfig": self.tools,
            }

            try:
                response = self.bedrock_client.converse(**converse_api_params)
                return self._parse_tool_response(response)
            except ClientError as e:
                logger.error(f"Bedrock converse call failed: {e}")
                raise RuntimeError(f"Analysis failed: {e}")

        except Exception as e:
            logger.error(f"Error analyzing media: {e}")
            raise

    def _parse_tool_response(self, response: Dict) -> Dict[str, Any]:
        """Parse tool responses from Bedrock output"""
        try:
            content_blocks = response["output"]["message"]["content"]
            tool_results = {}
            
            for block in content_blocks:
                if "toolUse" in block:
                    tool_use = block["toolUse"]
                    tool_name = tool_use.get("name")
                    tool_input = tool_use.get("input", {})
                    
                    if tool_name == "report_final_assessment":
                        tool_results["final_assessment"] = tool_input
                    else:
                        tool_results[tool_name] = tool_input
            
            return tool_results
            
        except KeyError:
            raise RuntimeError(f"Unexpected response shape: {json.dumps(response, indent=2)}")

    # Placeholder implementations for analysis tools
    async def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        logger.info(f"Audio analysis requested for: {audio_path}")
        model = whisper.load_model("medium")
        result = model.transcribe(audio_path)
        return {
            "transcript": result["text"],
            "status": "analysis_complete"
        }

    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Placeholder for image analysis"""
        logger.info(f"Image analysis requested for: {image_path}")
        return {
            "fake_score": 82,
            "confidence": 88,
            "findings": ["AI-generated artifacts detected", "Inconsistent lighting patterns"],
            "status": "analysis_complete"
        }

    async def analyze_text(self, text_content: str) -> Dict[str, Any]:
        """Analyze text content for AI-generated and fake characteristics using Bedrock"""
        logger.info(f"Text analysis requested for: {text_content[:100]}...")
        
        try:
            # Initialize text analyzer
            text_analyzer = TextAnalyzer(self.bedrock_client, self.model_id)
            
            # Perform comprehensive text analysis
            analysis_result = await text_analyzer.analyze_text(text_content)
            
            logger.info(f"Text analysis completed: AI score={analysis_result.get('ai_score', 'N/A')}, "
                    f"Fake score={analysis_result.get('fake_score', 'N/A')}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {
                "ai_score": 50,
                "fake_score": 50,
                "confidence": 0,
                "ai_evidence": [f"Analysis error: {str(e)}"],
                "fake_evidence": [f"Analysis error: {str(e)}"],
                "status": "analysis_failed",
                "error": str(e)
            }

# Initialize the orchestrator
orchestrator = MediaAnalysisOrchestrator()

# Example usage
async def example_usage():
    """Example of how to use the orchestrator"""
    try:
        # # Analyze a video file using Bedrock orchestration
        # result = await orchestrator.analyze_media(
        #     media_path=EXAMPLE_VIDEO_PATH,
        #     media_type="video"
        # )
        # print("Analysis result:", json.dumps(result, indent=2))
        
        # Extract audio and frames to preset paths
        audio_path, frame_paths = await orchestrator.extract_audio_and_frames(
            video_path=EXAMPLE_VIDEO_PATH,
            frames_count=3
        )
        print(f"Extracted audio: {audio_path}")
        print(f"Extracted frames: {frame_paths}")

        audio_analyser_output = await orchestrator.analyze_audio(audio_path=audio_path)
        print(audio_analyser_output["transcript"])
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())