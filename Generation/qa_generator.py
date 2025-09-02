#!/usr/bin/env python3
"""
Question-Answer generation using LLM with tool calling
"""
import pdb
import json
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
try:
    from .utils import log_message, log_simple, encode_image_to_base64, safe_get_response_content, safe_get_token_usage, is_verbose
    from .frame_extractor import FrameExtractor, SegmentFeatureExtractor
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from utils import log_message, log_simple, encode_image_to_base64, safe_get_response_content, safe_get_token_usage, is_verbose
    from frame_extractor import FrameExtractor, SegmentFeatureExtractor


class QAGenerator:
    """Generates question-answer pairs using LLM with tool calling"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.frame_extractor = FrameExtractor()
        self.segment_extractor = SegmentFeatureExtractor()
        self.total_tokens = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        self.cache_stats = {'cache_hits': 0, 'cached_tokens': 0, 'cache_discount': 0.0}
    
    def setup_tools_and_system_message(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Setup tools and system message for QA generation with caching support"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "REFINE_SEGMENT",
                    "description": "Samples multiple frames within a specified timestamp range and returns them in chronological order for visual analysis. \
                                    Best for understanding: movements, actions, interractions over time. \
                                    Use when you need to know what happened when",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_second": {"type": "number", "description": "Start time in seconds"},
                            "end_second": {"type": "number", "description": "End time in seconds"}
                        },
                        "required": ["start_second", "end_second"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "REFINE_FRAME",
                    "description": "Extracts a specific frame at the given timestamp and returns the actual image for visual analysis. \
                                    Best for understanding: what objects are present, their properties, spatial layout. \
                                    Use when you need to know what objects were there at that moment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timestamp_second": {"type": "number", "description": "Timestamp in seconds to extract the frame image"}
                        },
                        "required": ["timestamp_second"]
                    }
                }
            }
        ]
        
        # System message with caching support
        system_message_text = """##Overall Task
Imagine you are a user who has been wearing an AR/VR headset for an extended period, during which the device continuously recorded your surroundings. From this full recording, I will select a short clip: video V. Your job is to envision a daily-life scenario S that occurs any amount of time after the events shown in video V have ended.

Within this scenario S, think of a question Q that the user might naturally ask the AR/VR device; the answer to this question must require the device to review video V. The question Q should be rooted in everyday life, described as unambiguously as possible, and fully consistent with scenario S. Then provide the correct answer A to that question. Answer A must be absolutely accurate and unambiguous.

## Context you will receive
1. **Segment list**: an ordered set of action descriptions in video V with timestamp. 
2. **QA key frame**: a single frame image from video V with timestamp, showing visual content at that moment.
3. **Selected Object**: One object from the key frame has been specifically chosen, with its bounding box coordinates. Your eventual question or answer **MUST** be related to this selected object.
**CRITICAL** The analysis may contain errors, please cross-verify all facts using independent sources.

## Tools you can call
You **CANNOT** see raw video, but you can make openai style function calls to gather more information:
**REFINE_SEGMENT(start_second, end_second)**  
**REFINE_FRAME(timestamp_second)**  

##Complete overall task in following steps
1. Analyze full segment list and the QA key-frame image. Identify the selected object information provided in the context.
2. Brainstorm situations in which you might ask the device a daily-life question that requires information from the video V to answer.
3. Use the REFINE_SEGMENT and REFINE_FRAME tools to gather additional information needed for giving the question and answer.
 • After each function response, briefly reflect on what you learned before deciding whether another call is necessary.
 • Feel free to chain function calls: study responses, think, then request another refinement until you believe you understand enough to craft a good question-and-answer pair.
4. Use the given tools to cross-verify each facts in QA.
 • **CRITICAL** Questions and Answers must be supported by facts from at least **TWO** independent sources (frames or segment analyses)
 • **DO NOT** use duplicate timestamps for cross-verify; instead, you may use other timestamps for frame refine verification or different time intervals for segment refine verification.
5. When satisfied, produce your final output (see format below) and stop calling function.
 • Strictly follow the required format and do not generate any additional content.
 • **CRITICAL** The question or answer **MUST** be related to this selected object.

##Final output format
scenario:
<one-sentence description of the daily-life scenario when the user would ask>
Question:
<what the user says to the AR/VR assistant>
Answer:
<absolutely accurate and unambiguous answer to the Question>
Evidence:
<quote to frame or segment refine for cross-verify and QA evidence>

##Important Notes (**CRITICAL**)
1. Regarding scenario:
 • It must depict an everyday situation.
 • Scenario S should take place some time after video V ends. It does not have to be directly related to video V, but the question Q must be relevant to Scenario S.
2. Regarding QA:
 • Descriptions must be precise and answers absolutely correct. Use enough qualifiers (location, appearance) to make each object unambiguous.
 • Q or A must involve selected object.
 • No speculation: the absence of evidence in the video does not prove something never happened.
 • Q should be realistic, as if asked by an actual user.
 • If these conditions cannot be met, create a new QA pair.
3. Regarding Evidence:
 • Every fact in the QA must be backed by at least two independent information sources. If this cannot be satisfied, create a new QA pair.
4. output:
 • Never invent facts, rely only on the provided descriptions.
 • **CRITICAL** Use the standard OpenAI function-calling format for tool calls; do not invoke tools with plain text."""
        
        # Return system message as structured content with caching
        system_message = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message_text,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }
        
        return tools, system_message
    
    def create_initial_message(self, keyframe_path: str, timestamp: float, 
                             video_summary: str, selected_object: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create initial message with keyframe and context, using caching for both video_summary and keyframe"""
        # Format selected object info
        selected_object_info = ""
        if selected_object:
            # Safely prefer gemini_bbox, then normalized, then pixel bbox
            gemini_bbox = selected_object.get('gemini_bbox')
            if gemini_bbox is None:
                gemini_bbox = selected_object.get('normalized_bbox')
            if gemini_bbox is None:
                gemini_bbox = selected_object.get('bbox')

            selected_object_info = f"""
Selected Object for Question Focus:
Object Name: {selected_object.get('name', 'unknown')}
Object Bounding Box: {gemini_bbox} (format: [ymin, xmin, ymax, xmax], normalized 0-1000 if available)
"""
        
        # Split content to enable caching for both keyframe and video_summary
        keyframe_prompt = f"""Here is the QA key-frame image:
This is a key frame sampled from video at {timestamp:.1f} seconds.
{selected_object_info}"""
        
        base64_image = encode_image_to_base64(keyframe_path)
        
        # Combine all content with single cache control at the end for OpenRouter Gemini
        combined_text = f"{keyframe_prompt}\n\nHere is the Full segment list description:\n{video_summary}"
        
        return [
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image
                        }
                    },
                    {
                        "type": "text", 
                        "text": combined_text,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            }
        ]
    
    def handle_refine_segment(self, func_call: Dict[str, Any], video_path: str, 
                             temp_dir: str) -> Tuple[str, List[str]]:
        """Handle REFINE_SEGMENT function call"""
        args = json.loads(func_call['arguments'])
        start_second = float(args["start_second"])
        end_second = float(args["end_second"])
        
        log_message(f"REFINE_SEGMENT requested: {start_second}s - {end_second}s")
        
        try:
            # Get representative frames using clustering
            representative_frames = self.segment_extractor.get_representative_frames(
                video_path, start_second, end_second, n_clusters=10, temp_dir=temp_dir
            )
            
            if not representative_frames:
                return f"Cannot extract representative frames from {start_second}s - {end_second}s", []
            
            # Extract and save representative frames
            frame_paths = []
            frame_descriptions = []
            
            for frame_data in representative_frames:
                timestamp = frame_data['timestamp']
                cluster_id = frame_data['cluster_id']
                
                try:
                    frame_path = self.frame_extractor.extract_frame_at_timestamp(
                        video_path, timestamp, temp_dir
                    )
                    
                    # Rename for better identification
                    new_name = f"segment_{start_second:.1f}-{end_second:.1f}s_cluster_{cluster_id}_{timestamp:.1f}s.jpg"
                    new_path = os.path.join(temp_dir, new_name)
                    os.rename(frame_path, new_path)
                    
                    frame_paths.append(new_path)
                    frame_descriptions.append(f"{timestamp:.1f}s")
                    
                except Exception as e:
                    log_message(f"Failed to extract frame at {timestamp}s: {str(e)}")
            
            if not frame_paths:
                return f"Cannot extract valid frames from {start_second}s - {end_second}s", []
            
            # Create response text
            response_text = f"Analysis completed. Representative frames from {start_second}s - {end_second}s:"
            for i, desc in enumerate(frame_descriptions, 1):
                response_text += f"\n{i}. {desc}"
            
            return response_text, frame_paths
            
        except Exception as e:
            error_msg = f"Error analyzing segment {start_second}-{end_second}: {str(e)}"
            log_message(error_msg)
            return error_msg, []
    
    def handle_refine_frame(self, func_call: Dict[str, Any], video_path: str, 
                           temp_dir: str) -> Tuple[str, str]:
        """Handle REFINE_FRAME function call"""
        args = json.loads(func_call['arguments'])
        timestamp_second = float(args["timestamp_second"])
        
        log_message(f"REFINE_FRAME requested: {timestamp_second}s")
        
        try:
            frame_path = self.frame_extractor.extract_frame_at_timestamp(
                video_path, timestamp_second, temp_dir
            )
            
            # Rename for better identification
            new_name = f"refine_frame_{timestamp_second:.1f}s.jpg"
            new_path = os.path.join(temp_dir, new_name)
            os.rename(frame_path, new_path)
            
            response_text = f"Frame extracted at {timestamp_second}s from video. Image available for analysis."
            
            return response_text, new_path
            
        except Exception as e:
            error_msg = f"Error analyzing frame at {timestamp_second}s: {str(e)}"
            log_message(error_msg)
            return error_msg, ""
    
    def make_api_call(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Any:
        """Make API call to LLM with caching and usage tracking"""
        response = self.client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=1,
            max_tokens=65536,
            extra_body={
                "usage": {"include": True}  # Enable detailed usage tracking for cache metrics
            }
        )
        
        # Output Gemini response content to terminal only in verbose mode
        if is_verbose():
            response_content = safe_get_response_content(response)
            if response_content:
                print("\n" + "="*80)
                print("🤖 GEMINI RESPONSE:")
                print("="*80)
                print(response_content)
                print("="*80 + "\n")
            
            # Check for tool calls and display them
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                print("🔧 TOOL CALLS DETECTED:")
                for i, tool_call in enumerate(response.choices[0].message.tool_calls, 1):
                    print(f"  {i}. {tool_call.function.name}({tool_call.function.arguments})")
                print()
        
        # Track token usage including cached tokens
        token_usage = safe_get_token_usage(response)
        if token_usage:
            for key in self.total_tokens:
                if key in token_usage:
                    self.total_tokens[key] += token_usage[key]
            
            # Log cache usage details if available
            cached_tokens = token_usage.get('cached_tokens', 0)
            if cached_tokens > 0:
                cache_discount = token_usage.get('cache_discount', 0)
                self.cache_stats['cache_hits'] += 1
                self.cache_stats['cached_tokens'] += cached_tokens
                self.cache_stats['cache_discount'] += cache_discount
                log_message(f"💾 CACHE HIT: {cached_tokens} tokens saved, discount: ${cache_discount:.4f}")
            else:
                log_message(f"⚠️  No cached tokens detected in response")
        
        return response
    
    def generate_qa(self, video_path: str, keyframe_path: str, timestamp: float,
                   video_summary: str, selected_object: Dict[str, Any], 
                   temp_dir: str) -> Dict[str, Any]:
        """Generate QA pair using LLM with tool calling"""
        log_message("Starting QA generation")
        
        # Setup tools and system message
        tools, system_message = self.setup_tools_and_system_message()
        
        # Create initial messages - system_message is now already structured with cache_control
        messages = [system_message]
        messages.extend(self.create_initial_message(keyframe_path, timestamp, video_summary, selected_object))
        
        # Initial API call
        if is_verbose():
            print("\n" + "🚀 STARTING QA GENERATION" + "\n")
            print("📝 INITIAL CONTEXT:")
            print(f"   • Video: {video_path}")
            print(f"   • Timestamp: {timestamp:.1f}s")
            print(f"   • Selected Object: {selected_object.get('name', 'unknown')}")
            print()
        else:
            log_message(f"Generating QA for object '{selected_object.get('name', 'unknown')}' at {timestamp:.1f}s")
        
        log_simple("Making initial API call")
        response = self.make_api_call(messages, tools)
        
        # Process function calls iteratively
        max_iterations = 10
        iteration_count = 0
        
        while iteration_count < max_iterations:
            iteration_count += 1
            
            if is_verbose():
                print(f"\n📍 ITERATION {iteration_count} / {max_iterations}")
                print("-" * 50)
            
            # Check for tool calls
            has_tool_calls = (hasattr(response.choices[0].message, 'tool_calls') and 
                            response.choices[0].message.tool_calls)
            
            response_content = safe_get_response_content(response)
            
            # Parse text-based function calls if no tool calls
            text_function_calls = []
            if not has_tool_calls and response_content:
                text_function_calls = self._parse_text_function_calls(response_content)
            
            # If no function calls, break the loop
            if not has_tool_calls and not text_function_calls:
                if is_verbose():
                    print("✅ No more tool calls - Generation complete")
                break
            
            # Process function calls
            if has_tool_calls:
                if is_verbose():
                    print(f"🔧 Processing {len(response.choices[0].message.tool_calls)} standard tool calls:")
                else:
                    log_message(f"Processing {len(response.choices[0].message.tool_calls)} tool calls")
                messages.append(response.choices[0].message)
                
                for i, tool_call in enumerate(response.choices[0].message.tool_calls, 1):
                    if is_verbose():
                        print(f"   {i}. Executing: {tool_call.function.name}")
                    self._process_tool_call(tool_call, messages, video_path, temp_dir)
                    
            else:
                # Handle text-based function calls
                if text_function_calls:
                    if is_verbose():
                        print(f"🔧 Processing {len(text_function_calls)} text-based function calls:")
                    else:
                        log_message(f"Processing {len(text_function_calls)} text-based tool calls")
                    messages.append({"role": "assistant", "content": response_content})
                    
                    for i, func_call in enumerate(text_function_calls):
                        if is_verbose():
                            print(f"   {i+1}. Executing: {func_call['name']}")
                        mock_tool_call = type('obj', (object,), {
                            'id': f"text_call_{iteration_count}_{i}",
                            'function': type('obj', (object,), {
                                'name': func_call['name'],
                                'arguments': json.dumps(func_call['parameters'])
                            })()
                        })()
                        
                        self._process_tool_call(mock_tool_call, messages, video_path, temp_dir)
                elif not text_function_calls:
                    messages.append({"role": "assistant", "content": response_content})
            
            # Make next API call
            log_message(f"Making API call - Iteration {iteration_count}")
            response = self.make_api_call(messages, tools)
        
        # Get final result
        if is_verbose():
            print("\n🎯 EXTRACTING FINAL RESULT")
            print("=" * 50)
        else:
            log_message("Finalizing QA generation")
        
        final_result = safe_get_response_content(response)
        
        if not final_result:
            log_simple("No final result, requesting continuation")
            continue_messages = messages + [{"role": "user", "content": "Continue your analysis."}]
            response = self.make_api_call(continue_messages, tools)
            final_result = safe_get_response_content(response)
        
        if not final_result:
            final_result = "QA generation failed to complete"
        
        if final_result and is_verbose():
            print("📄 FINAL QA RESULT:")
            print("-" * 50)
            print(final_result)
            print("-" * 50)
        
        log_simple("QA generation completed")
        log_message(f"Total tokens used: {self.total_tokens['total_tokens']}")
        
        # Log cache statistics
        if self.cache_stats['cache_hits'] > 0:
            log_message(f"🎯 CACHE STATS: {self.cache_stats['cache_hits']} hits, "
                       f"{self.cache_stats['cached_tokens']} tokens cached, "
                       f"${self.cache_stats['cache_discount']:.4f} total discount")
        else:
            log_message("⚠️  NO CACHE HITS - Check if caching is working properly")
        
        # Parse components from response
        question, answer, evidence, scenario = self.extract_qa_components(final_result)
        
        # Check if parsing failed and try structured output fallback
        if question == "Could not extract question" or question == "Extraction failed":
            log_simple("QA component extraction failed, attempting structured output fallback")
            
            structured_response = self.request_structured_output(final_result)
            if structured_response:
                question, answer, evidence, scenario = self.extract_qa_from_structured_response(structured_response)
                
                if question != "Structured parsing failed":
                    log_simple("Successfully recovered QA components using structured output")
                else:
                    log_simple("Structured output fallback also failed")
            else:
                log_simple("Structured output request failed")
        else:
            log_simple("QA components extracted successfully")
        
        return {
            'raw_response': final_result,
            'question': question,
            'answer': answer, 
            'evidence': evidence,
            'scenario': scenario,
            'success': question != "Could not extract question" and question != "Extraction failed" and question != "Structured parsing failed"
        }
    
    def _parse_text_function_calls(self, response_content: str) -> List[Dict[str, Any]]:
        """Parse function calls from text response"""
        function_calls = []
        
        try:
            # Pattern for complete function call format
            full_pattern = r'\{[^{}]*"type"[^{}]*"function"[^{}]*"name"[^{}]*"parameters"[^{}]*\{[^{}]*\}[^{}]*\}'
            full_matches = re.findall(full_pattern, response_content)
            
            # Pattern for simple function call format
            simple_pattern = r'\{[^{}]*"name"[^{}]*"parameters"[^{}]*\{[^{}]*\}[^{}]*\}'
            simple_matches = re.findall(simple_pattern, response_content)
            
            all_matches = full_matches + simple_matches
            
            for match in all_matches:
                try:
                    func_json = json.loads(match)
                    if ("name" in func_json and "parameters" in func_json and
                        func_json["name"] in ["REFINE_SEGMENT", "REFINE_FRAME"]):
                        
                        if "type" not in func_json:
                            func_json["type"] = "function"
                        function_calls.append(func_json)
                except Exception:
                    continue
                    
        except Exception:
            pass
        
        return function_calls
    
    def get_structured_qa_schema(self) -> Dict[str, Any]:
        """Get JSON schema for structured QA output"""
        return {
            "name": "structured_qa_output",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "scenario": {
                        "type": "string",
                        "description": "One-sentence description of the daily-life scenario when the user would ask the question"
                    },
                    "question": {
                        "type": "string",
                        "description": "What the user says to the AR/VR assistant - the question they would naturally ask"
                    },
                    "answer": {
                        "type": "string",
                        "description": "Absolutely accurate and unambiguous answer to the question"
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Quote from frame or segment analysis used for cross-verification and QA evidence"
                    }
                },
                "required": ["scenario", "question", "answer", "evidence"],
                "additionalProperties": False
            }
        }
    
    def request_structured_output(self, original_response: str) -> Optional[str]:
        """Request structured output when parsing fails"""
        try:
            log_message("Requesting structured output due to parsing failure")
            
            structured_prompt = f"""The previous response could not be parsed properly. Please reformat your answer using the exact structured format below:

Original response:
{original_response}

Please provide a structured JSON output with the following format:
- scenario: One-sentence description of the daily-life scenario when the user would ask
- question: What the user says to the AR/VR assistant
- answer: Absolutely accurate and unambiguous answer to the question  
- evidence: Quote from frame or segment analysis for cross-verification

Ensure the JSON is valid and follows the required structure."""
            
            json_schema = self.get_structured_qa_schema()
            
            response = self.client.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=[
                    {
                        "role": "user",
                        "content": structured_prompt
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": json_schema
                },
                temperature=0.1,
                max_tokens=65536
            )
            
            structured_response = safe_get_response_content(response)
            if structured_response:
                log_message("✅ Successfully obtained structured QA output")
                return structured_response
            else:
                log_message("❌ Failed to get structured output")
                return None
                
        except Exception as e:
            log_message(f"Structured output request failed: {str(e)}")
            return None
    
    def extract_qa_components(self, qa_response: str) -> Tuple[str, str, str, str]:
        """Extract question, answer, evidence, and scenario from QA response"""
        try:
            # Extract scenario
            scenario_pattern = r"[Ss]cenario:\s*\n?(.*?)(?=\n[Qq]uestion:|$)"
            scenario_match = re.search(scenario_pattern, qa_response, re.DOTALL)
            
            if scenario_match:
                scenario = scenario_match.group(1).strip()
            else:
                # Alternative pattern
                lines = qa_response.split('\n')
                scenario = "No scenario found"
                for i, line in enumerate(lines):
                    if line.strip().lower().startswith('scenario:'):
                        scenario = line.split(':', 1)[1].strip()
                        if i + 1 < len(lines) and not lines[i + 1].strip().lower().startswith(('question:', 'answer:', 'evidence:')):
                            scenario += " " + lines[i + 1].strip()
                        break
            
            # Extract question and answer
            question_pattern = r"[Qq]uestion:\s*\n?(.*?)(?=\n[Aa]nswer:|$)"
            answer_pattern = r"[Aa]nswer:\s*\n?(.*?)(?=\n[Ee]vidence:|$)"
            evidence_pattern = r"[Ee]vidence:\s*\n?(.*?)(?=\n|$)"
            
            question_match = re.search(question_pattern, qa_response, re.DOTALL)
            answer_match = re.search(answer_pattern, qa_response, re.DOTALL)
            evidence_match = re.search(evidence_pattern, qa_response, re.DOTALL)
            
            if question_match and answer_match:
                question = question_match.group(1).strip()
                answer = answer_match.group(1).strip()
                evidence = evidence_match.group(1).strip() if evidence_match else "No evidence found"
                
                log_message("✅ Successfully extracted QA components from response")
                return question, answer, evidence, scenario
            
            # Try alternative patterns
            lines = qa_response.split('\n')
            question = None
            answer = None
            evidence = None
            
            for i, line in enumerate(lines):
                if line.strip().lower().startswith('question:'):
                    question = line.split(':', 1)[1].strip()
                    if i + 1 < len(lines):
                        question += " " + lines[i + 1].strip()
                elif line.strip().lower().startswith('answer:'):
                    answer = line.split(':', 1)[1].strip()
                    if i + 1 < len(lines):
                        answer += " " + lines[i + 1].strip()
                elif line.strip().lower().startswith('evidence:'):
                    evidence = line.split(':', 1)[1].strip()
                    if i + 1 < len(lines):
                        evidence += " " + lines[i + 1].strip()
            
            if question and answer:
                log_message("✅ Extracted QA components using alternative method")
                return question, answer, evidence or "No evidence found", scenario
            
            log_message("❌ Failed to extract QA components")
            return "Could not extract question", "Could not extract answer", "No evidence found", scenario
            
        except Exception as e:
            log_message(f"Error extracting QA components: {str(e)}")
            return "Extraction failed", "Extraction failed", "Extraction failed", "Extraction failed"
    
    def extract_qa_from_structured_response(self, structured_response: str) -> Tuple[str, str, str, str]:
        """Extract question, answer, evidence, scenario from structured JSON response"""
        try:
            log_message("Parsing structured JSON QA response")
            
            # Clean response text and extract JSON
            cleaned_response = structured_response.strip()
            
            # Remove code block markers if present
            if cleaned_response.startswith('```'):
                json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned_response, re.DOTALL)
                if json_match:
                    cleaned_response = json_match.group(1).strip()
            
            # Extract JSON object
            brace_start = cleaned_response.find('{')
            if brace_start != -1:
                brace_count = 0
                brace_end = brace_start
                for i, char in enumerate(cleaned_response[brace_start:], brace_start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            brace_end = i
                            break
                cleaned_response = cleaned_response[brace_start:brace_end + 1]
            
            # Parse JSON
            parsed_data = json.loads(cleaned_response)
            
            # Extract fields
            question = parsed_data.get('question', 'Failed to parse question').strip()
            answer = parsed_data.get('answer', 'Failed to parse answer').strip()
            evidence = parsed_data.get('evidence', 'Failed to parse evidence').strip()
            scenario = parsed_data.get('scenario', 'Failed to parse scenario').strip()
            
            log_message("✅ Successfully extracted components from structured response")
            return question, answer, evidence, scenario
            
        except Exception as e:
            log_message(f"Structured parsing failed: {str(e)}")
            return "Structured parsing failed", "Structured parsing failed", "Structured parsing failed", "Structured parsing failed"
    
    def _process_tool_call(self, tool_call: Any, messages: List[Dict[str, Any]], 
                          video_path: str, temp_dir: str) -> None:
        """Process individual tool call"""
        if tool_call.function.name == "REFINE_SEGMENT":
            if is_verbose():
                print(f"     🎞️  REFINE_SEGMENT: {tool_call.function.arguments}")
            else:
                # Parse arguments for simple logging
                try:
                    args = json.loads(tool_call.function.arguments)
                    start_s = args.get("start_second", "?")
                    end_s = args.get("end_second", "?")
                    log_message(f"Refining video segment {start_s}s-{end_s}s")
                except:
                    log_message("Refining video segment")
            
            response_text, frame_paths = self.handle_refine_segment(
                {'arguments': tool_call.function.arguments}, video_path, temp_dir
            )
            
            if is_verbose():
                print(f"     ✅ Extracted {len(frame_paths)} representative frames")
            else:
                log_message(f"Extracted {len(frame_paths)} representative frames")
            
            # Add tool response
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": response_text
            })
            
            # Add frames if available
            if frame_paths:
                content = [{"type": "text", "text": "Representative frames from the segment:"}]
                for i, frame_path in enumerate(frame_paths):
                    content.append({"type": "text", "text": f"Frame {i+1}:"})
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": encode_image_to_base64(frame_path)}
                    })
                messages.append({"role": "user", "content": content})
                
        elif tool_call.function.name == "REFINE_FRAME":
            if is_verbose():
                print(f"     🖼️  REFINE_FRAME: {tool_call.function.arguments}")
            else:
                # Parse arguments for simple logging
                try:
                    args = json.loads(tool_call.function.arguments)
                    timestamp = args.get("timestamp_second", "?")
                    log_message(f"Extracting frame at {timestamp}s")
                except:
                    log_message("Extracting frame")
            
            response_text, frame_path = self.handle_refine_frame(
                {'arguments': tool_call.function.arguments}, video_path, temp_dir
            )
            
            if is_verbose():
                if frame_path and os.path.exists(frame_path):
                    print(f"     ✅ Extracted frame: {os.path.basename(frame_path)}")
                else:
                    print(f"     ❌ Failed to extract frame")
            else:
                if frame_path and os.path.exists(frame_path):
                    log_message(f"Frame extracted successfully")
                else:
                    log_message(f"Frame extraction failed")
            
            # Add tool response
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": response_text
            })
            
            # Add frame if available
            if frame_path and os.path.exists(frame_path):
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is the extracted frame:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": encode_image_to_base64(frame_path)}
                        }
                    ]
                })


def main():
    """Test QA generation functionality"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test QA generator')
    parser.add_argument('--api-key', required=True, help='OpenRouter API key')
    parser.add_argument('--video-path', required=True, help='Video file path')
    parser.add_argument('--keyframe-path', required=True, help='Key frame image path')
    parser.add_argument('--timestamp', type=float, required=True, help='Key frame timestamp')
    parser.add_argument('--temp-dir', default='tmp', help='Temporary directory')
    
    args = parser.parse_args()
    
    # Create temp directory
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Mock data for testing
    mock_summary = "0.0s-30.0s: Person working at desk\n30.0s-60.0s: Person reading book"
    mock_selected_object = {
        'name': 'laptop',
        'bbox': [100, 100, 300, 200],
        'normalized_bbox': [100, 100, 300, 200],
        'gemini_bbox': [100, 100, 300, 200]
    }
    
    # Test QA generation
    generator = QAGenerator(args.api_key)
    
    try:
        result = generator.generate_qa(
            args.video_path, args.keyframe_path, args.timestamp,
            mock_summary, mock_selected_object, args.temp_dir
        )
        
        log_message("QA Generation Result:")
        print(f"Success: {result['success']}")
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Evidence: {result['evidence']}")
        print(f"Scenario: {result['scenario']}")
        print(f"Raw Response: {result['raw_response']}")
        
    except Exception as e:
        log_message(f"QA generation test failed: {str(e)}")


if __name__ == "__main__":
    main() 
