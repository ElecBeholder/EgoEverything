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
        # Store approved MCQ data when review passes
        self.approved_mcq = None
        self.review_attempt_count = 0  # Track number of review attempts

        # Reviewer session state management
        self.reviewer_messages = None  # Reviewer LLM conversation messages
        self.last_provide_feedback_call_id = None  # ID of last PROVIDE_FEEDBACK tool call
        self.reviewer = None  # QAReviewerRefiner instance
    
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
            },
            {
                "type": "function",
                "function": {
                    "name": "REQUEST_REVIEW",
                    "description": "Submit the generated question and answer options for review by the QA review agent. \
                                    Use this ONLY AFTER you have fully completed generating the MCQ. \
                                    The reviewer will check fact accuracy, ambiguity, logic and wording according to strict criteria.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "The generated question"},
                            "option_a": {"type": "string", "description": "First answer option (A)"},
                            "option_b": {"type": "string", "description": "Second answer option (B)"},
                            "option_c": {"type": "string", "description": "Third answer option (C)"},
                            "option_d": {"type": "string", "description": "Fourth answer option (D)"},
                            "option_e": {"type": "string", "description": "Fifth answer option (E)"},
                            "correct_answer": {"type": "string", "description": "The correct answer letter (A, B, C, D, or E)"},
                            "evidence_timestamps": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "List of timestamp(s) in seconds that provide evidence for the answer"
                            }
                        },
                        "required": ["question", "option_a", "option_b", "option_c", "option_d", "option_e", "correct_answer", "evidence_timestamps"]
                    }
                }
            }
        ]
        
        # System message with caching support
        system_message_text = """##Overall Task
Imagine you are a user who has been wearing an AR/VR headset for an extended period, during which the device continuously recorded your surroundings. From this full recording, I will select a short clip: video V. Your job is to envision a daily-life scenario S that occurs any amount of time after the events shown in video V have ended.

Within this scenario S, think of a question Q that the user might naturally ask the AR/VR device; the answer to this question must require the device to review video V. The question Q should be rooted in everyday life, described as unambiguously as possible, and fully consistent with scenario S. Then create a multiple choice question with 5 options where one is the correct answer.

## Context you will receive
1. **Segment list**: an ordered set of action descriptions in video V with timestamp.
2. **QA key frame**: a single frame image from video V with timestamp, showing visual content at that moment.
3. **Selected Object**: One object from the key frame has been specifically chosen, with its bounding box coordinates. Your eventual question or answer **MUST** be related to this selected object.
**CRITICAL** The analysis may contain errors, please cross-verify all facts using independent sources.

## Tools you can call
You **CANNOT** see raw video, but you can use tools to gather more information:
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
5. Generate 5 multiple choice options (A, B, C, D, E) where one is correct and four are challenging but incorrect distractors.
 • The distractors should be plausible and close in meaning to confuse someone who hasn't paid close attention
 • All options should be stylistically consistent in length, structure, and detail level
6. When satisfied with your MCQ, call REQUEST_REVIEW to submit for review.
 • Include the question, all 5 options, correct answer letter, and evidence timestamps
 • The reviewer will provide feedback or approve with "pass"
 • If feedback is provided, refine your MCQ based on the feedback and submit for review again
 • **CRITICAL** The question or answer **MUST** be related to this selected object.

##Final output format
Use REQUEST_REVIEW tool with the following parameters:
- question: The question you generated
- option_a through option_e: The five answer options
- correct_answer: The letter of the correct answer (A/B/C/D/E)
- evidence_timestamps: Array of timestamp numbers that support your answer

DO NOT provide any text output format - use ONLY the REQUEST_REVIEW tool call.

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
3. Regarding MCQ Options:
 • Create 4 challenging distractors that are plausible but incorrect
 • All options must be stylistically consistent and similar in length/detail
 • Distractors should be close enough in meaning to be confusing
4. Regarding Evidence:
 • Every fact in the QA must be backed by at least two independent information sources. If this cannot be satisfied, create a new QA pair.
5. output:
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

    def handle_request_review(self, func_call: Dict[str, Any], video_path: str,
                             video_summary: str, temp_dir: str) -> str:
        """Handle REQUEST_REVIEW function call with bidirectional communication"""
        args = json.loads(func_call['arguments'])

        # Extract MCQ data
        question = args.get("question", "")
        option_a = args.get("option_a", "")
        option_b = args.get("option_b", "")
        option_c = args.get("option_c", "")
        option_d = args.get("option_d", "")
        option_e = args.get("option_e", "")
        correct_answer = args.get("correct_answer", "A")
        evidence_timestamps = args.get("evidence_timestamps", [])

        options_list = [option_a, option_b, option_c, option_d, option_e]

        # Increment review attempt counter
        self.review_attempt_count += 1
        max_attempts = 3

        log_message(f"REQUEST_REVIEW called (attempt {self.review_attempt_count}/{max_attempts})")

        if is_verbose():
            print("\n" + "="*80)
            print(f"🔄 GENERATOR → REVIEWER (Attempt {self.review_attempt_count}/{max_attempts})")
            print("="*80)

        try:
            # Import reviewer here to avoid circular imports
            try:
                from .qa_reviewer import QAReviewerRefiner
            except ImportError:
                from qa_reviewer import QAReviewerRefiner

            # Check if this is the first review request
            if self.reviewer_messages is None:
                # First time: Create new reviewer session
                if is_verbose():
                    print("📤 FIRST REVIEW - Creating new reviewer session")
                    print(f"   • Question: {question}")
                    print(f"   • Options: {options_list}")
                    print(f"   • Correct Answer: {correct_answer}")
                    print(f"   • Evidence: {len(evidence_timestamps)} timestamps")

                # Initialize reviewer
                self.reviewer = QAReviewerRefiner(self.client.api_key)

                # Extract evidence images
                evidence_images = []
                for timestamp in evidence_timestamps:
                    try:
                        frame_path = self.frame_extractor.extract_frame_at_timestamp(
                            video_path, float(timestamp), temp_dir
                        )
                        review_name = f"review_evidence_{timestamp:.1f}s.jpg"
                        review_path = os.path.join(temp_dir, review_name)
                        os.rename(frame_path, review_path)
                        evidence_images.append({
                            'type': 'frame',
                            'timestamp': timestamp,
                            'path': review_path
                        })
                    except Exception as e:
                        log_message(f"Failed to extract evidence frame at {timestamp}s: {str(e)}")

                # Create initial reviewer message
                tools, system_message = self.reviewer.setup_tools_and_system_message()
                self.reviewer_messages = [system_message]

                # Get correct answer text
                correct_index = ord(correct_answer.upper()) - ord('A') if correct_answer.upper() in 'ABCDE' else 0
                answer = options_list[correct_index] if 0 <= correct_index < len(options_list) else option_a

                # Add initial review request message
                self.reviewer_messages.extend(
                    self.reviewer.create_review_message(
                        question, answer, evidence_images, video_summary, options_list, correct_answer
                    )
                )

                # Call reviewer LLM
                response = self.reviewer.make_api_call(self.reviewer_messages, tools)

                # Process reviewer tool calls
                return self._process_reviewer_response(response, video_path, video_summary, temp_dir,
                                                      question, options_list, correct_answer, evidence_timestamps, max_attempts)

            else:
                # Subsequent review: Continue existing reviewer session
                if is_verbose():
                    print("📤 SUBSEQUENT REVIEW - Continuing reviewer session")
                    print(f"   • Modified Question: {question}")
                    print(f"   • Providing as tool response to previous PROVIDE_FEEDBACK")

                # Add modified QA as tool response to previous PROVIDE_FEEDBACK call
                self.reviewer_messages.append({
                    "role": "tool",
                    "name": "PROVIDE_FEEDBACK",
                    "tool_call_id": self.last_provide_feedback_call_id,
                    "content": f"""Modified MCQ after addressing feedback:

Question: {question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
E. {option_e}

Correct Answer: {correct_answer}
Evidence Timestamps: {evidence_timestamps}"""
                })

                # Continue reviewer LLM conversation
                tools, _ = self.reviewer.setup_tools_and_system_message()
                response = self.reviewer.make_api_call(self.reviewer_messages, tools)

                # Process reviewer tool calls
                return self._process_reviewer_response(response, video_path, video_summary, temp_dir,
                                                      question, options_list, correct_answer, evidence_timestamps, max_attempts)

        except Exception as e:
            error_msg = f"Review process failed: {str(e)}"
            log_message(error_msg)
            return error_msg

    def _process_reviewer_response(self, response: Any, video_path: str, video_summary: str,
                                   temp_dir: str, question: str, options_list: List[str],
                                   correct_answer: str, evidence_timestamps: List[float],
                                   max_attempts: int) -> str:
        """Process reviewer LLM response and handle tool calls"""
        tools, _ = self.reviewer.setup_tools_and_system_message()

        # Process tool calls in a loop
        max_iterations = 15
        iteration_count = 0

        while iteration_count < max_iterations:
            iteration_count += 1

            if is_verbose():
                print(f"\n📍 REVIEWER ITERATION {iteration_count}/{max_iterations}")

            # Check for tool calls
            has_tool_calls = (hasattr(response.choices[0].message, 'tool_calls') and
                            response.choices[0].message.tool_calls)

            response_content = safe_get_response_content(response)

            # Parse text-based function calls if no standard tool calls
            text_function_calls = []
            if not has_tool_calls and response_content:
                text_function_calls = self._parse_reviewer_text_function_calls(response_content)

            if not has_tool_calls and not text_function_calls:
                if is_verbose():
                    print("⚠️  No tool calls from reviewer - reminding to use tools")
                else:
                    log_message("Reviewer didn't use tools - sending reminder")

                # Add assistant message if there's content
                if response_content:
                    self.reviewer_messages.append({
                        "role": "assistant",
                        "content": response_content
                    })

                # Remind reviewer to use tools
                self.reviewer_messages.append({
                    "role": "user",
                    "content": "You must call a tool (VERIFY_SEGMENT, VERIFY_FRAME, or PROVIDE_FEEDBACK). Do not provide text output."
                })

                # Continue with another API call
                response = self.reviewer.make_api_call(self.reviewer_messages, tools)
                continue

            # Process standard tool calls
            if has_tool_calls:
                # Add assistant message with tool calls to reviewer messages
                self.reviewer_messages.append(response.choices[0].message)

                # Process each tool call
                for tool_call in response.choices[0].message.tool_calls:
                    func_name = tool_call.function.name

                    if func_name == "VERIFY_SEGMENT":
                        # Handle VERIFY_SEGMENT
                        if is_verbose():
                            print(f"   🎞️  Reviewer calling VERIFY_SEGMENT")

                        response_text, frame_paths = self.reviewer.handle_verify_segment(
                            {'arguments': tool_call.function.arguments}, video_path, temp_dir
                        )

                        # Add tool response with name field for Gemini
                        self.reviewer_messages.append({
                            "role": "tool",
                            "name": "VERIFY_SEGMENT",
                            "tool_call_id": tool_call.id,
                            "content": response_text
                        })

                        # Add frames if available
                        if frame_paths:
                            content = [{"type": "text", "text": "Verification frames:"}]
                            for frame_path in frame_paths:
                                content.append({
                                    "type": "image_url",
                                    "image_url": {"url": encode_image_to_base64(frame_path)}
                                })
                            self.reviewer_messages.append({"role": "user", "content": content})

                    elif func_name == "VERIFY_FRAME":
                        # Handle VERIFY_FRAME
                        if is_verbose():
                            print(f"   🖼️  Reviewer calling VERIFY_FRAME")

                        response_text, frame_path = self.reviewer.handle_verify_frame(
                            {'arguments': tool_call.function.arguments}, video_path, temp_dir
                        )

                        # Add tool response with name field for Gemini
                        self.reviewer_messages.append({
                            "role": "tool",
                            "name": "VERIFY_FRAME",
                            "tool_call_id": tool_call.id,
                            "content": response_text
                        })

                        # Add frame if available
                        if frame_path and os.path.exists(frame_path):
                            self.reviewer_messages.append({
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Verification frame:"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": encode_image_to_base64(frame_path)}
                                    }
                                ]
                            })

                    elif func_name == "PROVIDE_FEEDBACK":
                        # Handle PROVIDE_FEEDBACK - this is the key decision point
                        if is_verbose():
                            print(f"   📝 Reviewer calling PROVIDE_FEEDBACK")

                        args = json.loads(tool_call.function.arguments)
                        all_pass = args.get("all_pass", False)

                        # Save tool_call_id for potential subsequent reviews
                        self.last_provide_feedback_call_id = tool_call.id

                        if all_pass:
                            # All checklist items passed - save and approve
                            self.approved_mcq = {
                                'question': question,
                                'options': options_list,
                                'correct_answer': correct_answer,
                                'evidence_timestamps': evidence_timestamps,
                                'review_passed': True,
                                'review_attempts': self.review_attempt_count
                            }

                            if is_verbose():
                                print("\n📥 REVIEWER → GENERATOR:")
                                print("     ✅ All checklist items PASSED")
                                print("     🎉 MCQ approved!")
                                print("="*80 + "\n")
                            else:
                                log_message("Review passed - all checklist items satisfied")

                            return "APPROVED"

                        else:
                            # Checklist items failed - check if max attempts reached
                            if self.review_attempt_count >= max_attempts:
                                # Max attempts reached - save with flag
                                self.approved_mcq = {
                                    'question': question,
                                    'options': options_list,
                                    'correct_answer': correct_answer,
                                    'evidence_timestamps': evidence_timestamps,
                                    'review_passed': False,
                                    'review_attempts': self.review_attempt_count,
                                    'final_feedback': json.dumps(args, indent=2)
                                }

                                if is_verbose():
                                    print("\n📥 REVIEWER → GENERATOR:")
                                    print(f"     ⚠️  Max attempts ({max_attempts}) reached")
                                    print("     💾 Saving MCQ with review_not_passed flag")
                                    print("="*80 + "\n")
                                else:
                                    log_message(f"Max review attempts reached")

                                return "MAX_ATTEMPTS_REACHED"

                            else:
                                # Return feedback for generator to refine
                                feedback_text = self._format_feedback(args)

                                if is_verbose():
                                    print("\n📥 REVIEWER → GENERATOR:")
                                    print(f"     📋 Issues found (attempt {self.review_attempt_count}/{max_attempts})")
                                    print(f"     🔄 Requesting refinement")
                                    print("="*80 + "\n")
                                else:
                                    log_message(f"Review feedback provided for refinement")

                                return feedback_text

            else:
                # Process text-based function calls
                if is_verbose():
                    print(f"🔧 Processing {len(text_function_calls)} text-based reviewer tool calls")
                else:
                    log_message(f"Processing {len(text_function_calls)} text-based reviewer tool calls")

                # Add assistant message
                self.reviewer_messages.append({"role": "assistant", "content": response_content})

                # Process each text function call
                for i, func_call in enumerate(text_function_calls):
                    func_name = func_call['name']

                    # Create mock tool_call object
                    mock_tool_call = type('obj', (object,), {
                        'id': f"text_call_{iteration_count}_{i}",
                        'function': type('obj', (object,), {
                            'name': func_name,
                            'arguments': json.dumps(func_call['parameters'])
                        })()
                    })()

                    if func_name == "VERIFY_SEGMENT":
                        if is_verbose():
                            print(f"   🎞️  Reviewer calling VERIFY_SEGMENT (text format)")

                        response_text, frame_paths = self.reviewer.handle_verify_segment(
                            {'arguments': mock_tool_call.function.arguments}, video_path, temp_dir
                        )

                        self.reviewer_messages.append({
                            "role": "tool",
                            "name": "VERIFY_SEGMENT",
                            "tool_call_id": mock_tool_call.id,
                            "content": response_text
                        })

                        if frame_paths:
                            content = [{"type": "text", "text": "Verification frames:"}]
                            for frame_path in frame_paths:
                                content.append({
                                    "type": "image_url",
                                    "image_url": {"url": encode_image_to_base64(frame_path)}
                                })
                            self.reviewer_messages.append({"role": "user", "content": content})

                    elif func_name == "VERIFY_FRAME":
                        if is_verbose():
                            print(f"   🖼️  Reviewer calling VERIFY_FRAME (text format)")

                        response_text, frame_path = self.reviewer.handle_verify_frame(
                            {'arguments': mock_tool_call.function.arguments}, video_path, temp_dir
                        )

                        self.reviewer_messages.append({
                            "role": "tool",
                            "name": "VERIFY_FRAME",
                            "tool_call_id": mock_tool_call.id,
                            "content": response_text
                        })

                        if frame_path and os.path.exists(frame_path):
                            self.reviewer_messages.append({
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Verification frame:"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": encode_image_to_base64(frame_path)}
                                    }
                                ]
                            })

                    elif func_name == "PROVIDE_FEEDBACK":
                        if is_verbose():
                            print(f"   📝 Reviewer calling PROVIDE_FEEDBACK (text format)")

                        args = func_call['parameters']
                        all_pass = args.get("all_pass", False)

                        self.last_provide_feedback_call_id = mock_tool_call.id

                        if all_pass:
                            self.approved_mcq = {
                                'question': question,
                                'options': options_list,
                                'correct_answer': correct_answer,
                                'evidence_timestamps': evidence_timestamps,
                                'review_passed': True,
                                'review_attempts': self.review_attempt_count
                            }

                            if is_verbose():
                                print("\n📥 REVIEWER → GENERATOR:")
                                print("     ✅ All checklist items PASSED")
                                print("     🎉 MCQ approved!")
                                print("="*80 + "\n")
                            else:
                                log_message("Review passed")

                            return "APPROVED"

                        else:
                            if self.review_attempt_count >= max_attempts:
                                self.approved_mcq = {
                                    'question': question,
                                    'options': options_list,
                                    'correct_answer': correct_answer,
                                    'evidence_timestamps': evidence_timestamps,
                                    'review_passed': False,
                                    'review_attempts': self.review_attempt_count,
                                    'final_feedback': json.dumps(args, indent=2)
                                }

                                if is_verbose():
                                    print("\n📥 REVIEWER → GENERATOR:")
                                    print(f"     ⚠️  Max attempts ({max_attempts}) reached")
                                    print("="*80 + "\n")
                                else:
                                    log_message(f"Max review attempts reached")

                                return "MAX_ATTEMPTS_REACHED"

                            else:
                                feedback_text = self._format_feedback(args)

                                if is_verbose():
                                    print("\n📥 REVIEWER → GENERATOR:")
                                    print(f"     📋 Issues found (attempt {self.review_attempt_count}/{max_attempts})")
                                    print("="*80 + "\n")
                                else:
                                    log_message(f"Review feedback provided")

                                return feedback_text

            # Continue reviewer conversation
            response = self.reviewer.make_api_call(self.reviewer_messages, tools)

        # If we exhaust iterations without PROVIDE_FEEDBACK
        log_message("⚠️  Reviewer exhausted iterations without providing feedback")
        return "Review failed - no feedback provided"

    def _format_feedback(self, feedback_args: Dict[str, Any]) -> str:
        """Format feedback from PROVIDE_FEEDBACK tool call"""
        feedback_text = "Review Feedback:\n\n"
        feedback_text += f"1. Fact Verification: {feedback_args.get('fact_verification', 'N/A')}\n"
        feedback_text += f"2. Ambiguity Review: {feedback_args.get('ambiguity_review', 'N/A')}\n"
        feedback_text += f"3. Logic Chain: {feedback_args.get('logic_chain', 'N/A')}\n"
        feedback_text += f"4. Wording: {feedback_args.get('wording', 'N/A')}\n\n"

        suggestions = feedback_args.get('modification_suggestions', '')
        if suggestions:
            feedback_text += f"Modification Suggestions:\n{suggestions}\n"

        return feedback_text

    def _parse_reviewer_text_function_calls(self, response_content: str) -> List[Dict[str, Any]]:
        """Parse reviewer text-based function calls from response"""
        function_calls = []

        try:
            # Pattern for print(default_api.FUNCTION_NAME(...)) format
            print_api_pattern = r'print\(default_api\.(VERIFY_SEGMENT|VERIFY_FRAME|PROVIDE_FEEDBACK)\((.*?)\)\)'
            print_api_matches = re.finditer(print_api_pattern, response_content, re.DOTALL)

            for match in print_api_matches:
                function_name = match.group(1)
                parameters_str = match.group(2).strip()

                try:
                    parameters = {}

                    if function_name == "VERIFY_SEGMENT":
                        # Extract start_second and end_second
                        start_match = re.search(r'start_second\s*=\s*([\d.]+)', parameters_str)
                        end_match = re.search(r'end_second\s*=\s*([\d.]+)', parameters_str)

                        if start_match and end_match:
                            parameters["start_second"] = float(start_match.group(1))
                            parameters["end_second"] = float(end_match.group(1))

                    elif function_name == "VERIFY_FRAME":
                        # Extract timestamp_second
                        timestamp_match = re.search(r'timestamp_second\s*=\s*([\d.]+)', parameters_str)

                        if timestamp_match:
                            parameters["timestamp_second"] = float(timestamp_match.group(1))

                    elif function_name == "PROVIDE_FEEDBACK":
                        # Extract PROVIDE_FEEDBACK parameters
                        all_pass_match = re.search(r'all_pass\s*=\s*(True|False)', parameters_str, re.IGNORECASE)
                        if all_pass_match:
                            parameters["all_pass"] = all_pass_match.group(1).lower() == 'true'

                        # Extract text fields with quotes
                        fact_match = re.search(r'fact_verification\s*=\s*["\']([^"\']*)["\']', parameters_str)
                        if fact_match:
                            parameters["fact_verification"] = fact_match.group(1)

                        ambiguity_match = re.search(r'ambiguity_review\s*=\s*["\']([^"\']*)["\']', parameters_str)
                        if ambiguity_match:
                            parameters["ambiguity_review"] = ambiguity_match.group(1)

                        logic_match = re.search(r'logic_chain\s*=\s*["\']([^"\']*)["\']', parameters_str)
                        if logic_match:
                            parameters["logic_chain"] = logic_match.group(1)

                        wording_match = re.search(r'wording\s*=\s*["\']([^"\']*)["\']', parameters_str)
                        if wording_match:
                            parameters["wording"] = wording_match.group(1)

                        suggestions_match = re.search(r'modification_suggestions\s*=\s*["\']([^"\']*)["\']', parameters_str)
                        if suggestions_match:
                            parameters["modification_suggestions"] = suggestions_match.group(1)

                    # Create function call object if parameters were parsed successfully
                    if parameters:
                        function_calls.append({
                            "type": "function",
                            "name": function_name,
                            "parameters": parameters
                        })

                        if is_verbose():
                            log_message(f"📝 Parsed print(default_api.{function_name}) with params: {parameters}")

                except Exception as e:
                    if is_verbose():
                        log_message(f"Failed to parse print(default_api.{function_name}): {str(e)}")
                    continue

        except Exception:
            pass

        return function_calls

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

        # Reset state for new QA generation
        self.approved_mcq = None
        self.review_attempt_count = 0
        self.reviewer_messages = None
        self.last_provide_feedback_call_id = None
        self.reviewer = None

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
            should_stop = False
            if has_tool_calls:
                if is_verbose():
                    print(f"🔧 Processing {len(response.choices[0].message.tool_calls)} standard tool calls:")
                else:
                    log_message(f"Processing {len(response.choices[0].message.tool_calls)} tool calls")
                messages.append(response.choices[0].message)

                for i, tool_call in enumerate(response.choices[0].message.tool_calls, 1):
                    if is_verbose():
                        print(f"   {i}. Executing: {tool_call.function.name}")
                    stop_flag = self._process_tool_call(tool_call, messages, video_path, temp_dir, video_summary)
                    if stop_flag:
                        should_stop = True
                        break

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

                        stop_flag = self._process_tool_call(mock_tool_call, messages, video_path, temp_dir, video_summary)
                        if stop_flag:
                            should_stop = True
                            break
                elif not text_function_calls:
                    messages.append({"role": "assistant", "content": response_content})

            # Check if we should stop after processing tool calls
            if should_stop:
                if is_verbose():
                    print("\n🛑 Stopping generation - Review approved or max attempts reached")
                break
            
            # Make next API call
            log_message(f"Making API call - Iteration {iteration_count}")
            response = self.make_api_call(messages, tools)

        # If we have approved MCQ, return it directly without any further processing
        if self.approved_mcq:
            log_simple("QA generation completed")
            log_message(f"Total tokens used: {self.total_tokens['total_tokens']}")

            # Log cache statistics
            if self.cache_stats['cache_hits'] > 0:
                log_message(f"🎯 CACHE STATS: {self.cache_stats['cache_hits']} hits, "
                           f"{self.cache_stats['cached_tokens']} tokens cached, "
                           f"${self.cache_stats['cache_discount']:.4f} total discount")

            question = self.approved_mcq['question']
            options = self.approved_mcq['options']
            correct_answer = self.approved_mcq['correct_answer']
            evidence_timestamps = self.approved_mcq['evidence_timestamps']
            review_passed = self.approved_mcq.get('review_passed', True)
            review_attempts = self.approved_mcq.get('review_attempts', 0)

            # Get the correct answer text from options
            correct_index = ord(correct_answer.upper()) - ord('A') if correct_answer.upper() in 'ABCDE' else 0
            answer = options[correct_index] if 0 <= correct_index < len(options) else options[0]

            if review_passed:
                log_simple(f"✅ MCQ approved by reviewer on attempt {review_attempts}")
            else:
                log_simple(f"⚠️  MCQ did not pass review after {review_attempts} attempts")

            return {
                'raw_response': f"Approved by reviewer (attempt {review_attempts})",
                'question': question,
                'answer': answer,
                'options': options,
                'correct_answer': correct_answer,
                'evidence': f"Evidence timestamps: {evidence_timestamps}",
                'scenario': "Generated through review process",
                'success': review_passed,
                'review_passed': review_passed,
                'review_attempts': review_attempts
            }

        # Fallback: No approved MCQ - extract from final result
        log_simple("MCQ generation did not complete - no approved MCQ found")

        final_result = safe_get_response_content(response)

        if not final_result:
            log_simple("No final result, requesting continuation")
            continue_messages = messages + [{"role": "user", "content": "Continue your analysis."}]
            response = self.make_api_call(continue_messages, tools)
            final_result = safe_get_response_content(response)

        if not final_result:
            final_result = "QA generation failed to complete"

        question = "Generation incomplete"
        options = ["Option A", "Option B", "Option C", "Option D", "Option E"]
        correct_answer = "A"
        evidence = "No evidence found"
        scenario = "No scenario found"
        success = False

        try:
            question, correct_answer, options, evidence, scenario = self.extract_mcq_components(final_result)
            success = question != "Could not extract question" and question != "Extraction failed"
        except:
            pass

        return {
            'raw_response': final_result,
            'question': question,
            'options': options,
            'correct_answer': correct_answer,
            'evidence': evidence,
            'scenario': scenario,
            'success': success,
            'review_passed': False,
            'review_attempts': self.review_attempt_count
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
                        func_json["name"] in ["REFINE_SEGMENT", "REFINE_FRAME", "REQUEST_REVIEW"]):
                        
                        if "type" not in func_json:
                            func_json["type"] = "function"
                        function_calls.append(func_json)
                except Exception:
                    continue
            
            # Pattern for print(default_api.FUNCTION_NAME(...)) format
            print_api_pattern = r'print\(default_api\.(REFINE_SEGMENT|REFINE_FRAME|REQUEST_REVIEW)\((.*?)\)\)'
            print_api_matches = re.finditer(print_api_pattern, response_content, re.DOTALL)
            
            for match in print_api_matches:
                function_name = match.group(1)
                parameters_str = match.group(2).strip()
                
                try:
                    # Parse parameters from string like "start_second = 10, end_second = 20"
                    parameters = {}
                    
                    if function_name == "REFINE_SEGMENT":
                        # Extract start_second and end_second
                        start_match = re.search(r'start_second\s*=\s*([\d.]+)', parameters_str)
                        end_match = re.search(r'end_second\s*=\s*([\d.]+)', parameters_str)
                        
                        if start_match and end_match:
                            parameters["start_second"] = float(start_match.group(1))
                            parameters["end_second"] = float(end_match.group(1))
                    
                    elif function_name == "REFINE_FRAME":
                        # Extract timestamp_second
                        timestamp_match = re.search(r'timestamp_second\s*=\s*([\d.]+)', parameters_str)

                        if timestamp_match:
                            parameters["timestamp_second"] = float(timestamp_match.group(1))

                    elif function_name == "REQUEST_REVIEW":
                        # Extract REQUEST_REVIEW parameters
                        question_match = re.search(r'question\s*=\s*["\']([^"\']*)["\']', parameters_str)
                        if question_match:
                            parameters["question"] = question_match.group(1)

                        # Extract options
                        for opt_letter in ['a', 'b', 'c', 'd', 'e']:
                            opt_match = re.search(rf'option_{opt_letter}\s*=\s*["\']([^"\']*)["\']', parameters_str)
                            if opt_match:
                                parameters[f"option_{opt_letter}"] = opt_match.group(1)

                        # Extract correct answer
                        correct_match = re.search(r'correct_answer\s*=\s*["\']([A-E])["\']', parameters_str)
                        if correct_match:
                            parameters["correct_answer"] = correct_match.group(1)

                        # Extract evidence timestamps (array)
                        timestamps_match = re.search(r'evidence_timestamps\s*=\s*\[([\d.,\s]+)\]', parameters_str)
                        if timestamps_match:
                            timestamps_str = timestamps_match.group(1)
                            parameters["evidence_timestamps"] = [float(t.strip()) for t in timestamps_str.split(',') if t.strip()]
                    
                    # Create function call object if parameters were parsed successfully
                    if parameters:
                        function_calls.append({
                            "type": "function",
                            "name": function_name,
                            "parameters": parameters
                        })
                        
                        if is_verbose():
                            log_message(f"📝 Detected print(default_api.{function_name}) pattern with params: {parameters}")
                        
                except Exception as e:
                    if is_verbose():
                        log_message(f"Failed to parse print(default_api.{function_name}) parameters: {str(e)}")
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
    
    def extract_mcq_components(self, qa_response: str) -> Tuple[str, str, List[str], str, str]:
        """Extract question, options, correct answer, evidence, and scenario from MCQ response"""
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
                        if i + 1 < len(lines) and not lines[i + 1].strip().lower().startswith(('question:', 'options:', 'evidence:')):
                            scenario += " " + lines[i + 1].strip()
                        break

            # Extract question
            question_pattern = r"[Qq]uestion:\s*\n?(.*?)(?=\n[Oo]ptions:|$)"
            question_match = re.search(question_pattern, qa_response, re.DOTALL)

            if question_match:
                question = question_match.group(1).strip()
            else:
                question = "Could not extract question"

            # Extract options
            options_pattern = r"[Oo]ptions:\s*\n?(.*?)(?=\n[Cc]orrect [Aa]nswer:|$)"
            options_match = re.search(options_pattern, qa_response, re.DOTALL)

            options = []
            if options_match:
                options_text = options_match.group(1).strip()
                # Parse individual options A., B., C., D., E.
                option_pattern = r"[A-E]\.\s*(.*?)(?=[A-E]\.|$)"
                option_matches = re.findall(option_pattern, options_text, re.DOTALL)
                options = [opt.strip() for opt in option_matches if opt.strip()]

            # Extract correct answer
            correct_answer_pattern = r"[Cc]orrect [Aa]nswer:\s*([A-E])"
            correct_answer_match = re.search(correct_answer_pattern, qa_response)

            if correct_answer_match:
                correct_answer = correct_answer_match.group(1).upper()
            else:
                correct_answer = "A"  # Default fallback

            # Extract evidence
            evidence_pattern = r"[Ee]vidence:\s*\n?(.*?)(?=\n[A-Z][a-z]*:|$)"
            evidence_match = re.search(evidence_pattern, qa_response, re.DOTALL)

            if evidence_match:
                evidence = evidence_match.group(1).strip()
            else:
                evidence = "No evidence found"

            # Validate we have 5 options
            if len(options) != 5:
                log_message(f"❌ Expected 5 options, got {len(options)}. Padding with defaults.")
                while len(options) < 5:
                    options.append(f"Option {chr(65 + len(options))}")
                options = options[:5]  # Truncate if too many

            if question != "Could not extract question" and len(options) == 5:
                log_message("✅ Successfully extracted MCQ components from response")
                return question, correct_answer, options, evidence, scenario

            log_message("❌ Failed to extract MCQ components")
            return "Could not extract question", "A", ["Option A", "Option B", "Option C", "Option D", "Option E"], "No evidence found", scenario

        except Exception as e:
            log_message(f"Error extracting MCQ components: {str(e)}")
            return "Extraction failed", "A", ["Option A", "Option B", "Option C", "Option D", "Option E"], "Extraction failed", "Extraction failed"
    
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
                          video_path: str, temp_dir: str, video_summary: str = "") -> bool:
        """Process individual tool call and return True if should stop iteration"""
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

            return False  # Continue iteration

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

            return False  # Continue iteration

        elif tool_call.function.name == "REQUEST_REVIEW":
            if is_verbose():
                print(f"     📝 REQUEST_REVIEW: Submitting MCQ for review")
            else:
                log_message("Submitting MCQ for review")

            review_feedback = self.handle_request_review(
                {'arguments': tool_call.function.arguments}, video_path, video_summary, temp_dir
            )

            # Check if review approved or max attempts reached
            if review_feedback == "APPROVED":
                if is_verbose():
                    print(f"     ✅ Review APPROVED! Stopping generation.")
                else:
                    log_message("Review APPROVED - stopping generation")

                # Add final tool response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "Your MCQ has been approved. Generation complete."
                })
                return True  # Signal to stop iteration

            elif review_feedback == "MAX_ATTEMPTS_REACHED":
                if is_verbose():
                    print(f"     ⚠️  Max attempts reached! Saving current MCQ and stopping.")
                else:
                    log_message("Max review attempts reached - stopping generation")

                # Add final tool response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "Maximum review attempts reached. Current MCQ has been saved."
                })
                return True  # Signal to stop iteration

            else:
                # Continue with feedback for refinement
                if is_verbose():
                    print(f"     📋 Review feedback received - refinement needed")
                else:
                    log_message("Review feedback received")

                # Add tool response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": review_feedback
                })
                return False  # Continue iteration


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
        print(f"Options: {result['options']}")
        print(f"Correct Answer: {result['correct_answer']}")
        print(f"Evidence: {result['evidence']}")
        print(f"Scenario: {result['scenario']}")
        print(f"Raw Response: {result['raw_response']}")
        
    except Exception as e:
        log_message(f"QA generation test failed: {str(e)}")


if __name__ == "__main__":
    main() 
