# Integration script for Perplexity MCP and Odoo MCP
#
# This script queries both MCP servers with a business question, fetches and correlates data, and outputs a markdown report.

import requests
import json
from typing import Dict, Any

PERPLEXITY_MCP_URL = "http://localhost:8000/mcp"  # Adjust if needed
ODOO_MCP_URL = "http://localhost:8069/mcp"        # Adjust if needed


def query_perplexity(question: str) -> Dict[str, Any]:
    """Query Perplexity MCP server with a business question."""
    response = requests.post(
        f"{PERPLEXITY_MCP_URL}/ask",
        json={"question": question},
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def query_odoo(question: str) -> Dict[str, Any]:
    """Query Odoo MCP server for ERP data relevant to the question."""
    response = requests.post(
        f"{ODOO_MCP_URL}/ask",
        json={"question": question},
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def correlate_and_summarize(perplexity_data: Dict[str, Any], odoo_data: Dict[str, Any], question: str) -> str:
    """Correlate and summarize data from both sources into a markdown report."""
    report = f"# Business Question\n\n{question}\n\n"
    report += "## Perplexity MCP Response\n\n"
    report += json.dumps(perplexity_data, indent=2)
    report += "\n\n## Odoo MCP Response\n\n"
    report += json.dumps(odoo_data, indent=2)
    report += "\n\n## Correlation & Summary\n\n"
    report += "- [ ] TODO: Implement correlation and summary logic.\n"
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Integrate Perplexity MCP and Odoo MCP for business questions.")
    parser.add_argument("question", type=str, help="Business question to ask both MCP servers.")
    parser.add_argument("--output", type=str, default="report.md", help="Output markdown file.")
    args = parser.parse_args()

    print(f"Querying Perplexity MCP with: {args.question}")
    perplexity_data = query_perplexity(args.question)
    print("Querying Odoo MCP...")
    odoo_data = query_odoo(args.question)
    print("Correlating and summarizing...")
    report = correlate_and_summarize(perplexity_data, odoo_data, args.question)
    with open(args.output, "w") as f:
        f.write(report)
    print(f"Report written to {args.output}")

if __name__ == "__main__":
    main()
