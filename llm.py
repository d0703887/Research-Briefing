import requests

def query_ollama(
        user_prompt: str,
        system_prompt: str,
        model: str,
        temperature: float=0.7,
        base_url="http://localhost:11434"
) -> str:
    endpoint = f"{base_url.rstrip('/')}/api/chat"

    # Payload structure
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "options": {
            "temperature": temperature
        },
        "stream": False  # Return the whole response at once
    }

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()  # Raise error for 4xx or 5xx status codes

        # Parse response
        result = response.json()

        # Extract the actual content
        return result.get("message", {}).get("content", "")

    except requests.exceptions.RequestException as e:
        return f"Error communicating with Ollama: {e}"



