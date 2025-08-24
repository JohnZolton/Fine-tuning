import json
import random
import time
from typing import List, Dict, Any, Literal
from google import genai
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
NUM_SAMPLES = 5000
# Construct paths relative to the script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "sft_data.jsonl")
TOOLS_FILE = os.path.join(SCRIPT_DIR, "tools.json")
MODEL_NAME = "gemini-2.5-flash-lite"

# --- Google GenAI Client Setup ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in a .env file in the dataset directory.")
client = genai.Client(api_key=api_key)

# --- Load Tools ---
# The .env file should also be in the same directory as the script for load_dotenv() to find it easily.
# If .env is in the parent, adjust load_dotenv path, e.g., load_dotenv(os.path.join(SCRIPT_DIR, '..', '.env'))
# For now, assuming .env is in the same directory as the script.
with open(TOOLS_FILE, "r") as f:
    tools_data = json.load(f)
    tools_definitions = tools_data["tools"]

# --- Pydantic Models for Structured Generation ---

class ToolCall(BaseModel):
    name: str
    # Parameters can be None if the tool call doesn't require any
    parameters: Dict[str, Any] | None = None

class QuestionToolPair(BaseModel):
    question: str = Field(description="A user request that can be fulfilled by one of the available tools.")
    tool_call: ToolCall

# --- Helper Functions ---

def get_tool_by_name(name: str) -> Dict[str, Any]:
    """Finds a tool definition by its name."""
    for tool in tools_definitions:
        if tool["name"] == name:
            return tool
    raise ValueError(f"Tool with name '{name}' not found.")

def generate_prompt_for_tool(tool: Dict[str, Any]) -> str:
    """Generates a diverse prompt for a given tool to solicit a question."""
    tool_name = tool["name"]
    tool_desc = tool["description"]
    params_info = []
    for param_name, param_details in tool.get("parameters", {}).get("properties", {}).items():
        param_type = param_details.get("type", "string")
        param_desc = param_details.get("description", "")
        params_info.append(f"- '{param_name}' ({param_type}): {param_desc}")
    
    params_str = "\n".join(params_info) if params_info else "No parameters."

    prompt_variations = [
        f"Generate a user question asking to '{tool_desc}'. The question should naturally imply the need for the following parameters:\n{params_str}\n\nFormulate the question as a user would ask an assistant. Do not include parameter names directly in the question.",
        f"Create a realistic user request for a personal assistant that would trigger the '{tool_name}' tool. The tool's purpose is: '{tool_desc}'. It requires these parameters: {params_str}. The question should be conversational and contain the necessary information for the tool call.",
        f"Imagine a user wants to '{tool_desc}'. What would they ask their assistant? The request must provide all the necessary information for the tool to be called, which includes: {params_str}. Make the question sound natural.",
        f"Formulate a user query that would be answered by using the '{tool_name}' tool. The tool description is '{tool_desc}'. The parameters needed are: {params_str}. The question should be a single sentence.",
        f"Come up with a user question for an AI assistant that necessitates the use of the '{tool_name}' tool. The tool does this: '{tool_desc}'. It needs these details to work: {params_str}. The question should be phrased naturally."
    ]
    return random.choice(prompt_variations)

# --- Data Generation Logic ---

def generate_sft_data():
    """Generates SFT data and saves it to a JSONL file."""
    sft_data = []
    print(f"Starting SFT data generation for {NUM_SAMPLES} samples using {MODEL_NAME}...")

    for i in range(NUM_SAMPLES):
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{NUM_SAMPLES} samples...")

        # 1. Select a random tool
        selected_tool_def = random.choice(tools_definitions)
        tool_name = selected_tool_def["name"]

        # 2. Generate a prompt for the LLM to create a user question
        prompt_for_question = generate_prompt_for_tool(selected_tool_def)

        try:
            # 3. Call Gemini to generate a question
            # We expect a single string output for the question
            question_response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt_for_question,
                config={
                    "response_mime_type": "text/plain", # Expect plain text for the question
                },
            )
            generated_question = question_response.text.strip().strip('"')

            # 4. Generate a prompt for the LLM to create the tool call
            # This prompt will ask the model to structure the output based on the tool's schema
            prompt_for_tool_call = (
                f"Given the following user question: '{generated_question}'\n\n"
                f"And the following tool definition:\n"
                f"Tool Name: {selected_tool_def['name']}\n"
                f"Tool Description: {selected_tool_def['description']}\n"
                f"Tool Parameters (JSON Schema):\n{json.dumps(selected_tool_def.get('parameters', {}), indent=2)}\n\n"
                f"Generate the appropriate tool call. Extract the necessary parameters directly from the user question. "
                f"If a required parameter cannot be inferred, use a reasonable placeholder appropriate for the parameter type and description. "
                f"The output must strictly adhere to the following Pydantic model schema for a ToolCall:\n"
                f"class ToolCall(BaseModel):\n    name: str\n    parameters: Dict[str, Any]\n"
            )

            # 5. Define the schema for the tool call response
            tool_params_schema = selected_tool_def.get("parameters", {}).get("properties", {})
            
            if tool_params_schema:
                # Schema for tools that have parameters
                class DynamicToolCallWithParams(BaseModel):
                    name: Literal[tool_name] = tool_name
                    parameters: Dict[str, Any]
                response_schema_model = DynamicToolCallWithParams
            else:
                # Schema for tools that do NOT have parameters
                class DynamicToolCallWithoutParams(BaseModel):
                    name: Literal[tool_name] = tool_name
                response_schema_model = DynamicToolCallWithoutParams

            # 6. Call Gemini to generate the structured tool call
            tool_call_response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt_for_tool_call,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_schema_model,
                },
            )
            
            # Use .parsed to get the Pydantic object directly
            # The parsed object will be either DynamicToolCallWithParams or DynamicToolCallWithoutParams
            # We need to convert it to our generic ToolCall model
            parsed_specific_tool_call = tool_call_response.parsed
            
            if tool_params_schema:
                # It's a DynamicToolCallWithParams
                parsed_tool_call = ToolCall(name=parsed_specific_tool_call.name, parameters=parsed_specific_tool_call.parameters)
            else:
                # It's a DynamicToolCallWithoutParams, parameters should be None
                parsed_tool_call = ToolCall(name=parsed_specific_tool_call.name, parameters=None)

            # 7. Construct the final pair
            pair = QuestionToolPair(
                question=generated_question,
                tool_call=parsed_tool_call
            )
            sft_data.append(pair.model_dump_json())

        except Exception as e:
            print(f"Error generating sample {i+1} for tool '{tool_name}': {e}")
            # Optionally, add a placeholder or skip this sample
            # For now, we'll just skip and try the next one.
            time.sleep(0.1) # Short delay before retrying
            continue
        
        # Add a small delay to avoid hitting rate limits, adjusted for paid tier
        # 4000 RPM means ~0.015s/request. With 2 requests/sample, ~0.03s/sample.
        # Using 0.05s for a safety margin.
        time.sleep(0.05)

    # 8. Save to JSONL file
    # Ensure the output directory exists if it's different from script dir (not needed here)
    with open(OUTPUT_FILE, "w") as f:
        for item in sft_data:
            f.write(item + "\n")
    
    print(f"\nSuccessfully generated {len(sft_data)} samples and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_sft_data()
