from mcp.server.fastmcp import FastMCP
from typing import Any
import httpx
from google import genai
from google.genai.types import Content, Part
import os,json,sys

mcp = FastMCP("Demo")

@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

@mcp.tool()
def subtract(a:int, b:int) -> int:
    return a - b

@mcp.tool()
def multiply(a:int, b:int) -> int:
    return a * b

@mcp.tool()
def divide(a:int, b:int) -> float:
    return a/b

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    return "Hello, {}".format(name)


@mcp.tool(
    # name="text_math_parser",
    # description="Extracts arithmetic operation and arguments from a paragraph",
    # parameters={
    #     "text": {
    #         "type": "string",
    #         "description": "Paragraph describing a math problem"
    #     }
    # }
)
async def parse_math_from_text(text: str) -> str:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model_version = "gemini-2.0-flash"
    prompt = f"""
You are a math extraction assistant.

Extract the operation and numbers from the following problem.

Return JSON like:
{{"operation": "add"|"subtract"|"multiply"|"divide", "a": <num>, "b": <num>}}

Input: "{text}"
Output:
    """

    response = client.models.generate_content(
        model=model_version,
        contents=[Content(role="user", parts=[Part.from_text(text=prompt)])]
    )
    print(response,file=sys.stderr)
    print("==================",file=sys.stderr)
    try:
        raw = response.candidates[0].content.parts[0].text
        print(raw[7:-4], file=sys.stderr)
        parsed = json.loads(raw[7:-4])
        return json.dumps(parsed)
    except Exception as e:
        return json.dumps({"error": "Failed to parse math", "details": str(e)})


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')