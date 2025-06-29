import sys
sys.path.append("/ocean/projects/cis240137p/dgarg2/github/EnzyGen/")

from mcp.server.fastmcp import FastMCP
from typing import Annotated
from pydantic import Field

mcp = FastMCP("enzygen")

@mcp.tool()
def run_enzygen(input_json: Annotated[str, Field(description="Location of script directory")]) -> str:
    # TODO
    
    return "Sucessfully completed step."

if __name__ == "__main__":
    mcp.run(transport='stdio')
