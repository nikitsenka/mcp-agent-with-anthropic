import arxiv
import json
import os
from typing import List
from mcp.server.fastmcp import FastMCP
from search_papers import search_papers as original_search_papers
from extract_info import extract_info as original_extract_info

mcp = FastMCP("research")

@mcp.tool()
def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.

    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)

    Returns:
        List of paper IDs found in the search
    """
    return original_search_papers(topic, max_results)

@mcp.tool()
def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.

    Args:
        paper_id: The ID of the paper to look for

    Returns:
        JSON string with paper information if found, error message if not found
    """
    return original_extract_info(paper_id)

if __name__ == "__main__":
    mcp.run(transport='stdio')
