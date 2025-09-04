"""
Search and Information Retrieval Tools

Provides search capabilities for ReAct agents. These tools demonstrate
async execution, external API integration, and result processing.
"""

import asyncio
import logging

from ..decorators import tool
from ..models import ToolCategory

logger = logging.getLogger("aetherflow.agents.tools.search")


@tool(
    name="mock_search",
    description="Simulate web search with mock results for testing",
    category=ToolCategory.SEARCH,
    timeout=10.0,
)
async def mock_search(
    query: str,
    max_results: int = 5,
    result_type: str = "web",
    language: str = "en",
) -> dict[str, str | list[dict[str, str]]]:
    """
    Mock search tool that simulates web search functionality.

    This tool is useful for testing and development when external
    search APIs are not available or desired.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (1-20)
        result_type: Type of search (web, images, news, academic)
        language: Language code for results (en, zh, es, etc.)

    Returns:
        Dictionary with query metadata and list of mock results
    """
    # Simulate network delay
    await asyncio.sleep(0.1)

    if not query.strip():
        raise ValueError("Search query cannot be empty")

    if max_results < 1 or max_results > 20:
        raise ValueError("max_results must be between 1 and 20")

    # Generate mock results based on query
    results = []
    base_domains = ["example.com", "demo.org", "sample.net", "test.edu", "mock.gov"]

    for i in range(min(max_results, 10)):  # Limit to reasonable number
        result = {
            "title": f"Result {i + 1}: {query} - Comprehensive Guide",
            "url": f"https://{base_domains[i % len(base_domains)]}/search-result-{i + 1}",
            "snippet": f"This is a mock search result for '{query}'. "
            f"It contains relevant information about {query.lower()} "
            f"and related topics. Result number {i + 1} of {max_results}.",
            "relevance_score": max(0.95 - i * 0.1, 0.1),
            "source": base_domains[i % len(base_domains)],
        }

        # Customize based on result type
        if result_type == "news":
            result["publish_date"] = f"2024-0{(i % 9) + 1}-15"
            result["author"] = f"Reporter {i + 1}"
        elif result_type == "academic":
            result["authors"] = [f"Dr. Smith {i}", f"Prof. Johnson {i}"]
            result["journal"] = f"Journal of {query.title()} Studies"
            result["year"] = 2024 - (i % 5)
        elif result_type == "images":
            result["image_url"] = (
                f"https://{base_domains[i % len(base_domains)]}/image-{i + 1}.jpg"
            )
            result["dimensions"] = f"{800 + i * 100}x{600 + i * 75}"

        results.append(result)

    # Prepare response
    response = {
        "query": query,
        "result_type": result_type,
        "language": language,
        "total_results_found": max_results * 10,  # Simulate large result set
        "results_returned": len(results),
        "search_time_ms": 150 + (len(query) * 2),  # Simulate search time
        "results": results,
        "suggestions": [
            f"{query} tutorial",
            f"{query} examples",
            f"best {query}",
            f"{query} guide",
        ][:3],  # Limit suggestions
    }

    logger.info(f"Mock search completed for '{query}': {len(results)} results")
    return response


@tool(
    name="text_search",
    description="Search within provided text content using simple keyword matching",
    category=ToolCategory.SEARCH,
)
def text_search(
    content: str,
    query: str,
    case_sensitive: bool = False,
    max_matches: int = 10,
    context_chars: int = 100,
) -> dict[str, str | int | list[dict[str, str | int]]]:
    """
    Search for text patterns within provided content.

    Args:
        content: Text content to search within
        query: Search query or pattern
        case_sensitive: Whether search should be case sensitive
        max_matches: Maximum number of matches to return
        context_chars: Number of characters of context around each match

    Returns:
        Dictionary with search results and metadata
    """
    if not content.strip():
        raise ValueError("Content cannot be empty")

    if not query.strip():
        raise ValueError("Search query cannot be empty")

    # Prepare search
    search_content = content if case_sensitive else content.lower()
    search_query = query if case_sensitive else query.lower()

    matches = []
    start_pos = 0
    match_count = 0

    while match_count < max_matches:
        # Find next occurrence
        pos = search_content.find(search_query, start_pos)
        if pos == -1:
            break

        # Extract context around the match
        context_start = max(0, pos - context_chars)
        context_end = min(len(content), pos + len(query) + context_chars)
        context = content[context_start:context_end]

        # Determine line number (approximate)
        line_number = content[:pos].count("\n") + 1

        match_info = {
            "position": pos,
            "line_number": line_number,
            "context": context,
            "match_text": content[pos : pos + len(query)],
            "before_context": content[context_start:pos],
            "after_context": content[pos + len(query) : context_end],
        }

        matches.append(match_info)
        match_count += 1
        start_pos = pos + 1  # Move past this match

    # Calculate statistics
    total_occurrences = search_content.count(search_query)

    result = {
        "query": query,
        "content_length": len(content),
        "case_sensitive": case_sensitive,
        "total_occurrences": total_occurrences,
        "matches_returned": len(matches),
        "matches": matches,
    }

    logger.debug(f"Text search for '{query}' found {total_occurrences} occurrences")
    return result


@tool(
    name="keyword_extractor",
    description="Extract keywords and key phrases from text content",
    category=ToolCategory.SEARCH,
)
def keyword_extractor(
    text: str,
    max_keywords: int = 10,
    min_length: int = 3,
    exclude_common: bool = True,
) -> dict[str, list[str] | int]:
    """
    Extract important keywords from text content.

    This is a simplified keyword extraction tool that uses
    frequency analysis and basic filtering.

    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords to return
        min_length: Minimum length of keywords
        exclude_common: Whether to exclude common stop words

    Returns:
        Dictionary with extracted keywords and statistics
    """
    import re
    from collections import Counter

    if not text.strip():
        raise ValueError("Text content cannot be empty")

    # Common stop words to exclude (simplified list)
    stop_words = (
        {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "can",
            "may",
            "might",
            "must",
            "shall",
            "not",
            "no",
            "nor",
            "so",
            "such",
            "only",
        }
        if exclude_common
        else set()
    )

    # Extract words using regex
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    # Filter words
    filtered_words = [
        word
        for word in words
        if len(word) >= min_length and (not exclude_common or word not in stop_words)
    ]

    # Count word frequencies
    word_counts = Counter(filtered_words)

    # Get top keywords
    top_keywords = word_counts.most_common(max_keywords)
    keywords = [word for word, count in top_keywords]
    keyword_frequencies = dict(top_keywords)

    # Simple phrase extraction (2-word combinations)
    phrases = []
    for i in range(len(filtered_words) - 1):
        phrase = f"{filtered_words[i]} {filtered_words[i + 1]}"
        phrases.append(phrase)

    phrase_counts = Counter(phrases)
    top_phrases = [
        phrase for phrase, count in phrase_counts.most_common(5) if count > 1
    ]

    result = {
        "keywords": keywords,
        "keyword_frequencies": keyword_frequencies,
        "phrases": top_phrases,
        "total_words": len(words),
        "unique_words": len(set(words)),
        "filtered_words": len(filtered_words),
        "text_length": len(text),
    }

    logger.debug(f"Extracted {len(keywords)} keywords from {len(text)} character text")
    return result
