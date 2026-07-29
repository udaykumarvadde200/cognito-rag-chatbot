from typing import Any, Dict
import requests
import os
import re
from dotenv import load_dotenv
from graph.state import GraphState

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def extract_city(question: str) -> str:
    """Extract city name from question using simple pattern matching."""
    patterns = [
        r"weather in ([a-zA-Z\s]+)",
        r"temperature in ([a-zA-Z\s]+)",
        r"forecast for ([a-zA-Z\s]+)",
        r"climate in ([a-zA-Z\s]+)",
        r"([a-zA-Z\s]+) weather",
        r"([a-zA-Z\s]+) temperature",
    ]
    for pattern in patterns:
        match = re.search(pattern, question.lower())
        if match:
            return match.group(1).strip().title()
    return "Hyderabad"  # default city


def weather_node(state: GraphState) -> Dict[str, Any]:
    print("---weather tool---")
    question = state["question"]
    city = extract_city(question)
    print(f"---fetching weather for: {city}---")

    try:
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }
        response = requests.get(url, params=params)
        data = response.json()

        if response.status_code == 200:
            temp        = data["main"]["temp"]
            feels_like  = data["main"]["feels_like"]
            humidity    = data["main"]["humidity"]
            description = data["weather"][0]["description"].capitalize()
            wind_speed  = data["wind"]["speed"]

            tool_output = (
                f"Weather in {city}:\n"
                f"🌡️  Temperature : {temp}°C (Feels like {feels_like}°C)\n"
                f"🌤️  Condition   : {description}\n"
                f"💧 Humidity    : {humidity}%\n"
                f"💨 Wind Speed  : {wind_speed} m/s"
            )
        else:
            tool_output = f"Could not fetch weather for '{city}'. Error: {data.get('message', 'Unknown error')}"

    except Exception as e:
        tool_output = f"Weather fetch failed: {str(e)}"

    return {
        "question":    question,
        "tool_type":   "weather",
        "tool_output": tool_output,
        "city":        city,
        "documents":   [],
    }