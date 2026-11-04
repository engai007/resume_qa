import json
import re


def extract_json(text: str) -> dict:
    """Pull the JSON object out of a model reply (drops <think> blocks and ``` fences)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M)
    return json.loads(text[text.find("{"): text.rfind("}") + 1])
