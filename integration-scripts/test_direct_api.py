import asyncio
import aiohttp
import os
import json

async def test_call_perplexity_direct(query: str, recency: str) -> str:
    print("=== DEBUGGING PERPLEXITY API CALL ===")
    
    # Check environment variable first
    api_key = os.getenv('PERPLEXITY_API_KEY')
    print(f"API Key found: {api_key is not None}")
    print(f"API Key length: {len(api_key) if api_key else 0}")
    print(f"API Key starts with: {api_key[:10] if api_key else 'None'}...")
    
    if not api_key:
        return "ERROR: PERPLEXITY_API_KEY not found in environment!"

    url = "https://api.perplexity.ai/chat/completions"
    model = os.getenv("PERPLEXITY_MODEL", "sonar")
    print(f"Using model: {model}")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": query},
        ],
        "max_tokens": "512",
        "temperature": 0.2,
        "top_p": 0.9,
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": recency,
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1,
        "return_citations": True,
        "search_context_size": "low",
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    print("Making request to:", url)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                print(f"Response status: {response.status}")
                
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error response body: {error_text}")
                    response.raise_for_status()
                
                data = await response.json()
                content = data["choices"][0]["message"]["content"]
                return content
                
    except aiohttp.ClientResponseError as e:
        print(f"ClientResponseError: {e.status} - {e.message}")
        raise
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        raise

async def main():
    result = await test_call_perplexity_direct("What is the capital of France?", "week")
    print("\n=== RESULT ===")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
