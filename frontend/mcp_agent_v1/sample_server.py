
from mcp.server.fastmcp import FastMCP
import base64
from typing import Any
import httpx

mcp = FastMCP("Demo")

@mcp.tool(description="Add two ints together")
def add(a: int, b: int) -> int:
    
    return a + b

@mcp.tool(description="Subtrat two ints")
def subtract(a:int, b:int) -> int:
    return a - b

@mcp.tool(description="Multiply two ints")
def multiply(a:int, b:int) -> int:
    return a * b

@mcp.tool(description = "Divid two ints")
def divide(a:int, b:int) -> float:
    return a/b

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    return "Hello, {}".format(name)


@mcp.tool(description="generate blue square")
def generate_red_square() -> str:

    from PIL import Image
    import io

    # Create 100x100 red image
    img = Image.new("RGB", (100, 100), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    # Encode to base64 string for transport
    return base64.b64encode(img_bytes).decode("utf-8")


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
