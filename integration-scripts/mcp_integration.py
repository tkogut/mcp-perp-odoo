import requests
import markdown2

def query_perplexity_mcp(question, endpoint="http://localhost:8000/mcp"):
    """Query the Perplexity MCP server with a business question."""
    payload = {
        "question": question
    }
    response = requests.post(f"{endpoint}/ask", json=payload)
    response.raise_for_status()
    return response.json()

def query_odoo_mcp(question, endpoint="http://localhost:8069/mcp"):
    """Query the Odoo MCP server for ERP data based on the question."""
    payload = {
        "question": question
    }
    response = requests.post(f"{endpoint}/ask", json=payload)
    response.raise_for_status()
    return response.json()

def correlate_and_summarize(perplexity_data, odoo_data, question):
    """Correlate and summarize information from both sources into a markdown report."""
    report = f"""
# Business Question
{question}

# Perplexity MCP Response
{perplexity_data.get('answer', 'No answer')}

# Odoo ERP Data
{odoo_data.get('data', 'No data')}

# Correlation & Insights
- (Add logic to correlate and summarize both sources)
"""
    return report

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Integrate Perplexity MCP and Odoo MCP for business questions.")
    parser.add_argument("question", type=str, help="Business question to ask.")
    parser.add_argument("--perplexity-endpoint", type=str, default="http://localhost:8000/mcp", help="Perplexity MCP endpoint.")
    parser.add_argument("--odoo-endpoint", type=str, default="http://localhost:8069/mcp", help="Odoo MCP endpoint.")
    parser.add_argument("--output", type=str, default="report.md", help="Output markdown file.")
    args = parser.parse_args()

    print("Querying Perplexity MCP...")
    perplexity_data = query_perplexity_mcp(args.question, args.perplexity_endpoint)
    print("Querying Odoo MCP...")
    odoo_data = query_odoo_mcp(args.question, args.odoo_endpoint)
    print("Correlating and summarizing...")
    report_md = correlate_and_summarize(perplexity_data, odoo_data, args.question)
    with open(args.output, "w") as f:
        f.write(report_md)
    print(f"Report written to {args.output}")

if __name__ == "__main__":
    main()
