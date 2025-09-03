#!/usr/bin/env python3
"""
QA Review and MCQ Refinement system using LLM with tool calling
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


class QAReviewerRefiner:
    """Reviews and refines question-answer pairs with MCQ conversion"""
    
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
        """Setup tools and system message for QA review and refinement"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "VERIFY_SEGMENT",
                    "description": "Samples frames within a timestamp range for visual verification",
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
                    "name": "VERIFY_FRAME",
                    "description": "Extracts a specific frame at given timestamp for detailed verification",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timestamp_second": {"type": "number", "description": "Timestamp in seconds"}
                        },
                        "required": ["timestamp_second"]
                    }
                }
            }
        ]
        
        # System message with review checklist and refinement instructions
        system_message_text = """## Task Overview
You are a VQA Quality Expert who reviews, refines, and converts QA pairs into multiple choice questions.

## Context You Will Receive
1. **Question**: The generated question
2. **Answer**: The generated answer
3. **Evidence Timestamps**: List of timestamps where evidence was found (frames and segments)
4. **Video Summary**: Action descriptions with timestamps
5. **Video Access**: Through VERIFY_FRAME and VERIFY_SEGMENT tools

## Review Checklist

### 1. Fact Verification
- Verify the consistency between evidence and the generated answer
- Ensure all facts in the answer are visually verifiable

### 2. Ambiguity Review
- **Spatial descriptions**: Ensure location descriptions are viewpoint-independent (avoid "left/right")
- **Object uniqueness**: Verify that object descriptions are unambiguous - check the video to make sure no other objects match the same description
- Check for any potential confusion in object identification

### 3. Logic Chain Review
- Determine if the answer is the ONLY logical conclusion from the evidence
- Consider alternative conclusions that could also be supported by the same evidence
- Imagine a contradictory conclusion that still fits the evidence - if possible, the logic is weak

### 4. Wording Review
- **No external time references**: Remove any references to time outside the video
- **No timestamps**: Replace video timestamps with action sequences or behavioral markers
- **Natural phrasing**: Ensure questions sound like natural everyday queries about personal activities
- **Detail-focused**: Questions should ask about specific details, not general information
- **No video awareness**: Remove any words like "in the video", "footage", "recording" - questions should be as if asking about real memory

## Refinement and MCQ Conversion Process

### Step 1: Refine the QA
Based on your review, refine the question and answer to fix all issues identified in the checklist

### Step 2: Generate MCQ Options
Create 4 incorrect options that are:
- May share some elements with the correct answer
- Have visually clear distinguishing details that make them wrong
- Fit the scenario and action type

## Output Format
After completing your review and refinement, provide output in this EXACT format:

REVIEW_CHECKLIST:
1. Fact Verification: [PASS/FAIL with specific issues]
2. Ambiguity Review: [PASS/FAIL with specific issues]
3. Logic Chain: [PASS/FAIL with specific issues]  
4. Wording: [PASS/FAIL with specific issues]

REFINEMENT_RATIONALE:
[Explain what changes you made and why]

REFINED_QUESTION:
[The improved question]

REFINED_OPTIONS:
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
E. [Option E]

CORRECT_ANSWER: [A/B/C/D/E]

DISTINCTION_NOTES:
[Brief explanation of what makes wrong answers incorrect]

## Critical Requirements
- ALWAYS use tools to verify evidence timestamps
- Be specific about visual details observed
- Ensure MCQ options are challenging but fair
- Focus on creating natural, memory-like questions"""
        
        # Return system message with caching
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
    
    def create_review_message(self, question: str, answer: str, 
                             evidence_timestamps: Dict[str, Any],
                             video_summary: str) -> List[Dict[str, Any]]:
        """Create message for review and refinement"""
        
        # Format timestamps for clarity
        timestamp_text = "Evidence Timestamps:\n"
        if evidence_timestamps.get("frames"):
            timestamp_text += "Single frames: " + ", ".join(
                [f"{f}s" for f in evidence_timestamps["frames"]]
            ) + "\n"
        if evidence_timestamps.get("segments"):
            timestamp_text += "Segments: " + ", ".join(
                [f"{s['start']}s-{s['end']}s" for s in evidence_timestamps["segments"]]
            )
        
        combined_text = f"""Please review and refine this QA pair:

**QUESTION**: {question}

**ANSWER**: {answer}

**EVIDENCE TIMESTAMPS**:
{timestamp_text}

**VIDEO SUMMARY**:
{video_summary}

Instructions:
1. Use the tools to verify the evidence at the provided timestamps
2. Complete the review checklist 
3. Refine the QA based on issues found
4. Convert to MCQ with 5 challenging options"""
        
        return [
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text", 
                        "text": combined_text,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            }
        ]
    
    def handle_verify_segment(self, func_call: Dict[str, Any], video_path: str, 
                             temp_dir: str) -> Tuple[str, List[str]]:
        """Handle VERIFY_SEGMENT function call"""
        args = json.loads(func_call['arguments'])
        start_second = float(args["start_second"])
        end_second = float(args["end_second"])
        
        log_message(f"REVIEWER VERIFY_SEGMENT: {start_second}s - {end_second}s")
        
        try:
            # Get representative frames
            representative_frames = self.segment_extractor.get_representative_frames(
                video_path, start_second, end_second, n_clusters=8, temp_dir=temp_dir
            )
            
            if not representative_frames:
                return f"Cannot extract frames from {start_second}s - {end_second}s", []
            
            # Extract frames
            frame_paths = []
            frame_descriptions = []
            
            for frame_data in representative_frames:
                timestamp = frame_data['timestamp']
                cluster_id = frame_data['cluster_id']
                
                try:
                    frame_path = self.frame_extractor.extract_frame_at_timestamp(
                        video_path, timestamp, temp_dir
                    )
                    
                    # Rename for identification
                    new_name = f"review_segment_{start_second:.1f}-{end_second:.1f}s_cluster_{cluster_id}_{timestamp:.1f}s.jpg"
                    new_path = os.path.join(temp_dir, new_name)
                    os.rename(frame_path, new_path)
                    
                    frame_paths.append(new_path)
                    frame_descriptions.append(f"{timestamp:.1f}s")
                    
                except Exception as e:
                    log_message(f"Failed to extract frame at {timestamp}s: {str(e)}")
            
            if not frame_paths:
                return f"Cannot extract valid frames from {start_second}s - {end_second}s", []
            
            response_text = f"Verification frames from {start_second}s - {end_second}s:"
            for i, desc in enumerate(frame_descriptions, 1):
                response_text += f"\n{i}. {desc}"
            
            return response_text, frame_paths
            
        except Exception as e:
            error_msg = f"Error verifying segment {start_second}-{end_second}: {str(e)}"
            log_message(error_msg)
            return error_msg, []
    
    def handle_verify_frame(self, func_call: Dict[str, Any], video_path: str, 
                           temp_dir: str) -> Tuple[str, str]:
        """Handle VERIFY_FRAME function call"""
        args = json.loads(func_call['arguments'])
        timestamp_second = float(args["timestamp_second"])
        
        log_message(f"REVIEWER VERIFY_FRAME: {timestamp_second}s")
        
        try:
            frame_path = self.frame_extractor.extract_frame_at_timestamp(
                video_path, timestamp_second, temp_dir
            )
            
            # Rename for identification
            new_name = f"review_frame_{timestamp_second:.1f}s.jpg"
            new_path = os.path.join(temp_dir, new_name)
            os.rename(frame_path, new_path)
            
            response_text = f"Frame extracted at {timestamp_second}s for verification."
            
            return response_text, new_path
            
        except Exception as e:
            error_msg = f"Error verifying frame at {timestamp_second}s: {str(e)}"
            log_message(error_msg)
            return error_msg, ""
    
    def make_api_call(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Any:
        """Make API call to LLM"""
        response = self.client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=65536,
            extra_body={
                "usage": {"include": True}
            }
        )
        
        # Output response in verbose mode
        if is_verbose():
            response_content = safe_get_response_content(response)
            if response_content:
                print("\n" + "="*80)
                print("🔍 REVIEWER RESPONSE:")
                print("="*80)
                print(response_content)
                print("="*80 + "\n")
            
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                print("🔧 REVIEWER TOOL CALLS:")
                for i, tool_call in enumerate(response.choices[0].message.tool_calls, 1):
                    print(f"  {i}. {tool_call.function.name}({tool_call.function.arguments})")
                print()
        
        # Track token usage
        token_usage = safe_get_token_usage(response)
        if token_usage:
            for key in self.total_tokens:
                if key in token_usage:
                    self.total_tokens[key] += token_usage[key]
            
            cached_tokens = token_usage.get('cached_tokens', 0)
            if cached_tokens > 0:
                cache_discount = token_usage.get('cache_discount', 0)
                self.cache_stats['cache_hits'] += 1
                self.cache_stats['cached_tokens'] += cached_tokens
                self.cache_stats['cache_discount'] += cache_discount
                log_message(f"💾 REVIEWER CACHE HIT: {cached_tokens} tokens saved, discount: ${cache_discount:.4f}")
        
        return response
    
    def review_and_refine(self, question: str, answer: str, 
                         evidence_timestamps: Dict[str, Any],
                         video_path: str, video_summary: str, 
                         temp_dir: str) -> Dict[str, Any]:
        """Review QA and refine to MCQ"""
        log_message("Starting QA review and MCQ refinement")
        
        # Setup tools and system message
        tools, system_message = self.setup_tools_and_system_message()
        
        # Create messages
        messages = [system_message]
        messages.extend(self.create_review_message(
            question, answer, evidence_timestamps, video_summary
        ))
        
        if is_verbose():
            print("\n" + "🔍 STARTING QA REVIEW & REFINEMENT" + "\n")
            print("📝 CONTEXT:")
            print(f"   • Question: {question}")
            print(f"   • Answer: {answer}")
            print(f"   • Evidence timestamps: {evidence_timestamps}")
            print()
        else:
            log_message(f"Reviewing QA: '{question}' -> '{answer}'")
        
        log_simple("Making initial review API call")
        response = self.make_api_call(messages, tools)
        
        # Process function calls
        max_iterations = 12
        iteration_count = 0
        
        while iteration_count < max_iterations:
            iteration_count += 1
            
            if is_verbose():
                print(f"\n📍 REVIEW ITERATION {iteration_count} / {max_iterations}")
                print("-" * 50)
            
            # Check for tool calls
            has_tool_calls = (hasattr(response.choices[0].message, 'tool_calls') and 
                            response.choices[0].message.tool_calls)
            
            response_content = safe_get_response_content(response)
            
            # If no function calls, break
            if not has_tool_calls:
                if is_verbose():
                    print("✅ No more tool calls - Review complete")
                break
            
            # Process tool calls
            if is_verbose():
                print(f"🔧 Processing {len(response.choices[0].message.tool_calls)} tool calls:")
            else:
                log_message(f"Reviewer processing {len(response.choices[0].message.tool_calls)} tool calls")
            
            messages.append(response.choices[0].message)
            
            for i, tool_call in enumerate(response.choices[0].message.tool_calls, 1):
                if is_verbose():
                    print(f"   {i}. Executing: {tool_call.function.name}")
                self._process_tool_call(tool_call, messages, video_path, temp_dir)
            
            # Make next API call
            log_message(f"Making reviewer API call - Iteration {iteration_count}")
            response = self.make_api_call(messages, tools)
        
        # Get final result
        if is_verbose():
            print("\n🎯 EXTRACTING REVIEW & REFINEMENT RESULT")
            print("=" * 50)
        else:
            log_message("Finalizing review and refinement")
        
        final_result = safe_get_response_content(response)
        
        if not final_result:
            log_simple("No final result, requesting continuation")
            continue_messages = messages + [{
                "role": "user", 
                "content": "Please provide your final review and refined MCQ."
            }]
            response = self.make_api_call(continue_messages, tools)
            final_result = safe_get_response_content(response)
        
        if not final_result:
            final_result = "Review and refinement failed to complete"
        
        if final_result and is_verbose():
            print("📄 FINAL RESULT:")
            print("-" * 50)
            print(final_result)
            print("-" * 50)
        
        log_simple("QA review and refinement completed")
        log_message(f"Reviewer total tokens used: {self.total_tokens['total_tokens']}")
        
        # Log cache statistics
        if self.cache_stats['cache_hits'] > 0:
            log_message(f"🎯 REVIEWER CACHE STATS: {self.cache_stats['cache_hits']} hits, "
                       f"{self.cache_stats['cached_tokens']} tokens cached, "
                       f"${self.cache_stats['cache_discount']:.4f} total discount")
        
        # Parse the review and refinement
        parsed_result = self.parse_review_response(final_result)
        
        return {
            'raw_response': final_result,
            'parsed_result': parsed_result,
            'success': bool(parsed_result.get('refined_question'))
        }
    
    def parse_review_response(self, response: str) -> Dict[str, Any]:
        """Parse the structured review and refinement response"""
        try:
            log_message("Parsing review and refinement response")
            
            parsed = {
                'review_checklist': {},
                'refinement_rationale': '',
                'refined_question': '',
                'refined_options': [],
                'correct_answer': '',
                'distinction_notes': ''
            }
            
            # Parse review checklist items
            checklist_pattern = r'REVIEW_CHECKLIST:(.*?)(?=REFINEMENT_RATIONALE:|$)'
            checklist_match = re.search(checklist_pattern, response, re.DOTALL | re.IGNORECASE)
            if checklist_match:
                checklist_text = checklist_match.group(1)
                # Parse individual checklist items
                fact_match = re.search(r'1\.\s*Fact Verification:\s*([^\n]+)', checklist_text)
                ambiguity_match = re.search(r'2\.\s*Ambiguity Review:\s*([^\n]+)', checklist_text)
                logic_match = re.search(r'3\.\s*Logic Chain:\s*([^\n]+)', checklist_text)
                wording_match = re.search(r'4\.\s*Wording:\s*([^\n]+)', checklist_text)
                
                if fact_match:
                    parsed['review_checklist']['fact_verification'] = fact_match.group(1).strip()
                if ambiguity_match:
                    parsed['review_checklist']['ambiguity_review'] = ambiguity_match.group(1).strip()
                if logic_match:
                    parsed['review_checklist']['logic_chain'] = logic_match.group(1).strip()
                if wording_match:
                    parsed['review_checklist']['wording'] = wording_match.group(1).strip()
            
            # Parse refinement rationale
            rationale_match = re.search(
                r'REFINEMENT_RATIONALE:\s*\n?(.*?)(?=\n\s*REFINED_QUESTION:|$)',
                response, re.DOTALL | re.IGNORECASE
            )
            if rationale_match:
                parsed['refinement_rationale'] = rationale_match.group(1).strip()
            
            # Parse refined question
            question_match = re.search(
                r'REFINED_QUESTION:\s*\n?(.*?)(?=\n\s*REFINED_OPTIONS:|$)',
                response, re.DOTALL | re.IGNORECASE
            )
            if question_match:
                parsed['refined_question'] = question_match.group(1).strip()
            
            # Parse refined options
            options_match = re.search(
                r'REFINED_OPTIONS:(.*?)(?=CORRECT_ANSWER:|$)',
                response, re.DOTALL | re.IGNORECASE
            )
            if options_match:
                options_text = options_match.group(1)
                option_pattern = r'[A-E]\.\s*(.+?)(?=[A-E]\.|$)'
                option_matches = re.findall(option_pattern, options_text, re.DOTALL)
                parsed['refined_options'] = [opt.strip() for opt in option_matches]
            
            # Parse correct answer
            answer_match = re.search(
                r'CORRECT_ANSWER:\s*([A-E])',
                response, re.IGNORECASE
            )
            if answer_match:
                parsed['correct_answer'] = answer_match.group(1).upper()
            
            # Parse distinction notes
            notes_match = re.search(
                r'DISTINCTION_NOTES:\s*\n?(.*?)(?=\n|$)',
                response, re.DOTALL | re.IGNORECASE
            )
            if notes_match:
                parsed['distinction_notes'] = notes_match.group(1).strip()
            
            log_message(f"✅ Parsing successful: found {len(parsed['refined_options'])} options")
            return parsed
            
        except Exception as e:
            log_message(f"Review parsing failed: {str(e)}")
            return {
                'review_checklist': {},
                'refinement_rationale': 'Parsing failed',
                'refined_question': '',
                'refined_options': [],
                'correct_answer': '',
                'distinction_notes': 'Parsing failed'
            }
    
    def _process_tool_call(self, tool_call: Any, messages: List[Dict[str, Any]], 
                          video_path: str, temp_dir: str) -> None:
        """Process individual tool call"""
        if tool_call.function.name == "VERIFY_SEGMENT":
            if is_verbose():
                print(f"     🎞️  VERIFY_SEGMENT: {tool_call.function.arguments}")
            else:
                try:
                    args = json.loads(tool_call.function.arguments)
                    start_s = args.get("start_second", "?")
                    end_s = args.get("end_second", "?")
                    log_message(f"Verifying segment {start_s}s-{end_s}s")
                except:
                    log_message("Verifying video segment")
            
            response_text, frame_paths = self.handle_verify_segment(
                {'arguments': tool_call.function.arguments}, video_path, temp_dir
            )
            
            if is_verbose():
                print(f"     ✅ Extracted {len(frame_paths)} frames")
            else:
                log_message(f"Extracted {len(frame_paths)} verification frames")
            
            # Add tool response
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": response_text
            })
            
            # Add frames if available
            if frame_paths:
                content = [{
                    "type": "text", 
                    "text": "Verification frames from the segment:"
                }]
                for i, frame_path in enumerate(frame_paths):
                    content.append({"type": "text", "text": f"Frame {i+1}:"})
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": encode_image_to_base64(frame_path)}
                    })
                messages.append({"role": "user", "content": content})
                
        elif tool_call.function.name == "VERIFY_FRAME":
            if is_verbose():
                print(f"     🖼️  VERIFY_FRAME: {tool_call.function.arguments}")
            else:
                try:
                    args = json.loads(tool_call.function.arguments)
                    timestamp = args.get("timestamp_second", "?")
                    log_message(f"Verifying frame at {timestamp}s")
                except:
                    log_message("Verifying frame")
            
            response_text, frame_path = self.handle_verify_frame(
                {'arguments': tool_call.function.arguments}, video_path, temp_dir
            )
            
            if is_verbose():
                if frame_path and os.path.exists(frame_path):
                    print(f"     ✅ Frame extracted: {os.path.basename(frame_path)}")
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
                        {"type": "text", "text": "Verification frame:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": encode_image_to_base64(frame_path)}
                        }
                    ]
                })


def main():
    """Test QA reviewer and refiner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test QA reviewer and refiner')
    parser.add_argument('--api-key', required=True, help='OpenRouter API key')
    parser.add_argument('--video-path', required=True, help='Video file path')
    parser.add_argument('--question', required=True, help='Test question')
    parser.add_argument('--answer', required=True, help='Test answer')
    parser.add_argument('--temp-dir', default='tmp', help='Temporary directory')
    
    args = parser.parse_args()
    
    # Create temp directory
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Mock data for testing
    mock_timestamps = {
        "frames": [{"timestamp": 10.5}, {"timestamp": 25.3}],
        "segments": [{"start": 30.0, "end": 45.0}]
    }
    mock_summary = "0.0s-30.0s: Person working at desk\n30.0s-60.0s: Person reading book"
    
    # Test review and refinement
    reviewer = QAReviewerRefiner(args.api_key)
    
    try:
        result = reviewer.review_and_refine(
            args.question, args.answer, mock_timestamps,
            args.video_path, mock_summary, args.temp_dir
        )
        
        log_message("Review & Refinement Result:")
        print(f"Success: {result['success']}")
        print(f"Review Checklist: {result['parsed_result']['review_checklist']}")
        print(f"Refined Question: {result['parsed_result']['refined_question']}")
        print(f"Refined Options: {result['parsed_result']['refined_options']}")
        print(f"Correct Answer: {result['parsed_result']['correct_answer']}")
        print(f"Distinction Notes: {result['parsed_result']['distinction_notes']}")
        
    except Exception as e:
        log_message(f"Review test failed: {str(e)}")


if __name__ == "__main__":
    main()
