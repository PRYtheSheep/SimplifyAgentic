import boto3
import json
import os
import tempfile
import subprocess
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
from image_metadata import get_exif_dict
import whisper
import torch

from globals import *
from judgement import JudgementBot

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

class ClaudeChat:
    def __init__(self):
        self.client = bedrock_client
        self.model_id = model_id
        self.history = []
        self.system_prompt = """
            You are a helpful assistant that gathers context from the user regarding a post.

            You are to ask for the following context:
            1. Source of the post and what is it showing.
            2. Why the user thinks the post might be deceiving.

            Probe for further information if the context provided is insufficient.
            DO not preamble when summarising. 

            Ask the questions one at a time.
            Do not ask questions that are unnecessary.
            """ 

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
                        "description": "Analyze image file manipulation",
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
                }
            ]
        }

        self.system_prompt = self.system_prompt = """
           ---
        """
    
    VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
    AUDIO_EXTS = {".mp3", ".wav", ".aac", ".flac"}
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

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

        def _fix_video(input_path: str) -> str:
            """Ensure the video is properly formatted for OpenCV by remuxing/re-encoding with ffmpeg"""
            fixed_path = tempfile.mktemp(suffix=".mp4")

            try:
                # Try remuxing (fast, no re-encode)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", input_path, "-c:v", "copy", "-an", fixed_path],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            except subprocess.CalledProcessError:
                # If remux fails, fallback to re-encode
                subprocess.run(
                    ["ffmpeg", "-y", "-i", input_path, "-c:v", "libx264", "-preset", "fast", "-an", fixed_path],
                    check=True
                )

            return fixed_path

        frame_paths = []
        
        # Apply global frame limit
        frames_count = min(frames_count, MAX_FRAMES_LIMIT)
        logger.info(f"Extracting up to {frames_count} frames (global limit: {MAX_FRAMES_LIMIT})")
        
        try:
            cap = cv2.VideoCapture(_fix_video(video_path))
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

    async def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        logger.info(f"Audio analysis requested for: {audio_path}")
        model = whisper.load_model("medium")
        result = model.transcribe(audio_path, task="translate")
        return {
            "transcript": result["text"],
            "status": "analysis_complete"
        }

    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        logger.info(f"Image analysis requested for: {image_path}")
        meta = get_exif_dict(image_path=image_path)
        manipulation = False
        if "photoshop" in meta.keys():
            manipulation = True
        return {
            "image_maniplulated": manipulation,
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
            
            # # Save to JSON file with proper formatting
            # with open('weblinks.json', 'w') as f:
            #     json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            
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
        
    async def analyze_media(self, media_file: str, user_context: str) -> dict:
        """
        Determines media type from file extension, then runs the appropriate workflow.
        
        Args:
            media_file (str): Path or identifier of the media file
        
        Returns:
            dict: Aggregated results from all analyses
        """
        _, ext = os.path.splitext(media_file)
        ext = ext.lower()

        results = {"user_context": user_context}

        text_from_audio, text_analysis, image_analysis = None, None, None

        if ext in self.VIDEO_EXTS:
            media_type = "video"
            # Step 1: Extract audio + frames
            audio_file_path, frame_files_path = await self.extract_audio_and_frames(media_file)
            # results["audio_extracted"] = audio_file
            # results["frames_extracted"] = frame_files

            # Step 2: Transcribe audio
            output = await self.analyze_audio(audio_file_path)
            text_from_audio = output["transcript"]
            logger.info(f"Audio analysis: {output["status"]}")

            # Step 3: Analyze text
            text_analysis = await self.analyze_text(text_from_audio)
            # results["text_analysis"] = text_analysis
            results.update({"text_analysis": text_analysis})

        elif ext in self.AUDIO_EXTS:
            media_type = "audio"
            # Step 1: Transcribe audio
            output = await self.analyze_audio(audio_file_path)
            text_from_audio = output["transcipt"]
            logger.info(f"Audio analysis: {output["status"]}")

            # Step 2: Analyze text
            text_analysis = await self.analyze_text(text_from_audio)
            # results["text_analysis"] = text_analysis
            results.update({"text_analysis": text_analysis})

        elif ext in self.IMAGE_EXTS:
            media_type = "image"
            # Step 1: Analyze image + metadata
            image_analysis = await self.analyze_image(media_file)
            # results["image_analysis"] = image_analysis
            results.update({"image_analysis": image_analysis})

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        region3 = os.getenv('REGION', DEFAULT_REGION)    
        bedrock_client2 = boto3.client(
            service_name='bedrock-runtime',
            region_name=region3,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        model_id = os.getenv("MODEL_ID2", DEFAULT_MODEL)

        judge = JudgementBot(bedrock_client2, model_id)
        # Final: Combine all outputs for judgement

        combined = {
            "Transcript": text_from_audio if text_from_audio else None,
            "Text": text_analysis if text_analysis else None,
            "ImageMetadata": image_analysis if image_analysis else None
        }
        with open('combined.json', "w") as f:
            json.dump(combined, f, indent=4)

        # results["media_type"] = media_type
        final_assessment = await judge.final_assessment(analysis_data=results)

        return final_assessment, text_analysis

    # async def _execute_tools_and_continue(self, response: Dict, media_path: str) -> Dict[str, Any]:
    #     """Execute requested tools and continue conversation until completion"""
    #     conversation_history = []
        
    #     # Add initial response to history
    #     conversation_history.append({
    #         "role": "assistant",
    #         "content": response["output"]["message"]["content"]
    #     })
        
    #     # Continue conversation until we get a final assessment
    #     while True:
    #         # Check for tool requests in the response
    #         tool_requests = self._extract_tool_requests(response)
            
    #         if not tool_requests:
    #             # No more tools to execute, return the final response
    #             return self._parse_final_response(response)
            
    #         # Execute all requested tools
    #         tool_results = []
    #         for tool_name, tool_input in tool_requests:
    #             try:
    #                 result = await self._execute_tool(tool_name, tool_input, media_path)
    #                 tool_results.append({
    #                     "toolUseId": f"tool_{len(tool_results)}",
    #                     "toolName": tool_name,
    #                     "content": [{"text": json.dumps(result)}]
    #                 })
    #             except Exception as e:
    #                 logger.error(f"Tool {tool_name} execution failed: {e}")
    #                 tool_results.append({
    #                     "toolUseId": f"tool_{len(tool_results)}",
    #                     "toolName": tool_name,
    #                     "content": [{"text": json.dumps({"error": str(e), "status": "failed"})}]
    #                 })
            
    #         # Continue conversation with tool results
    #         converse_api_params = {
    #             "modelId": self.model_id,
    #             "system": [{"text": self.system_prompt}],
    #             "messages": conversation_history,
    #             "toolResults": tool_results,
    #             "inferenceConfig": {
    #                 "temperature": 0.1,
    #                 "maxTokens": 1000
    #             },
    #             "toolConfig": self.tools,
    #         }
            
    #         response = self.bedrock_client.converse(**converse_api_params)
    #         conversation_history.append({
    #             "role": "assistant",
    #             "content": response["output"]["message"]["content"]
    #         })

    # def _extract_tool_requests(self, response: Dict) -> List[Tuple[str, Dict]]:
    #     """Extract tool requests from Bedrock response"""
    #     tool_requests = []
        
    #     for content_block in response["output"]["message"]["content"]:
    #         if "toolUse" in content_block:
    #             tool_use = content_block["toolUse"]
    #             tool_requests.append((tool_use["name"], tool_use.get("input", {})))
        
    #     return tool_requests

    # async def _execute_tool(self, tool_name: str, tool_input: Dict, media_path: str) -> Dict:
    #     """Execute the requested tool"""
    #     if tool_name == "extract_audio_and_frames":
    #         video_path = tool_input.get("video_path", media_path)
    #         frames_count = tool_input.get("frames_count", 5)
    #         return await self.extract_audio_and_frames(video_path, frames_count)
        
    #     elif tool_name == "analyze_audio":
    #         audio_path = tool_input.get("audio_path")
    #         return await self.analyze_audio(audio_path)
        
    #     elif tool_name == "analyze_image":
    #         image_path = tool_input.get("image_path")
    #         return await self.analyze_image(image_path)
        
    #     elif tool_name == "analyze_text":
    #         text_content = tool_input.get("text_content", "")
    #         return await self.analyze_text(text_content)
        
    #     elif tool_name == "report_final_assessment":
    #         return tool_input  # This is the final result
        
    #     else:
    #         raise ValueError(f"Unknown tool: {tool_name}")

    # def _parse_final_response(self, response: Dict) -> Dict:
    #     """Extract the final assessment from the response"""
    #     for content_block in response["output"]["message"]["content"]:
    #         if "toolUse" in content_block and content_block["toolUse"]["name"] == "report_final_assessment":
    #             return content_block["toolUse"]["input"]
        
    #     # If no final assessment found, return the raw response
    #     return {"raw_response": response}

# Initialize the orchestrator
orchestrator = MediaAnalysisOrchestrator()
chat = ClaudeChat()

# Example usage
async def example_usage_2(user_context,example_video_path=None):
    """Example of how to use the orchestrator"""
    try:
        # Get claude to extract context from user by repeatedly prompting user
        # user_context = input("Provide additional context about the media\n")
        if example_video_path:
            results = await orchestrator.analyze_media(example_video_path,user_context=user_context)
        else:
            results = await orchestrator.analyze_text(user_context)
        print(json.dumps(results))

        with open('final_output.json', "w") as f:
            json.dump(results, f, indent=4)

        logger.info("Analysis complete")
        return results
        
    except Exception as e:
        print(f"Error: {e}")

async def example_usage():
    """Example of how to use the orchestrator"""
    try:
        # Get claude to extract context from user by repeatedly prompting user
        user_context = input("Provide additional context about the media\n")
        results, text_analysis = await orchestrator.analyze_media(EXAMPLE_VIDEO_PATH, user_context=user_context)

        #TODO I want to extract claims only with len(evidence snippets) > 0
        # if isinstance(text_analysis, str):
        #     text_analysis = json.loads(text_analysis)

        # # now filter the claims with evidence
        # claims_with_evidence = [
        #     claim for claim in text_analysis["web_verification"]["claims"]
        #     if claim.get("evidence_snippets") and len(claim["evidence_snippets"]) > 0
        # ]

        # results.update(claims_with_evidence)
        # print(json.dumps(results))

        # # Save to JSON file with proper formatting
        # with open('weblinks.json', 'w') as f:
        #     json.dump(claims_with_evidence, f, indent=2, ensure_ascii=False)

        with open('final_output.json', "w") as f:
            json.dump(results, f, indent=4)

        logger.info("Analysis complete")
        return results
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())