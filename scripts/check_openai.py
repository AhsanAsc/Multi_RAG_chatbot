# type: ignore
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def check_openai_api_key():
    """Check if OpenAI API key is valid by making a test request."""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    try:
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        if response.status_code == 200:
            return True, "✅ OpenAI API key is valid."
        elif response.status_code == 401:
            return False, "❌ Invalid or expired OpenAI API key."
        else:
            return False, f"⚠️ Unexpected response: {response.text}"
    except Exception as e:
        return False, f"⚠️ Error connecting to OpenAI: {str(e)}"


def check_openai_usage():
    """Check OpenAI usage details for the current month."""
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        today = datetime.now().date()
        first_day = today.replace(day=1)
        url = f"https://api.openai.com/v1/usage?start_date={first_day}&end_date={today}"

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            usage_data = response.json()
            total_usage = (
                usage_data.get("total_usage", 0) / 100
            )  # Convert from cents to USD
            return True, f"📊 Total usage this month: ${total_usage:.2f} USD"
        else:
            return False, f"⚠️ Could not retrieve usage: {response.text}"
    except Exception as e:
        return False, f"⚠️ Error getting usage: {str(e)}"


if __name__ == "__main__":
    print("🔐 Checking OpenAI API Key...")
    key_status, key_message = check_openai_api_key()
    print(key_message)

    if key_status:
        print("🔄 Checking API usage...")
        usage_status, usage_message = check_openai_usage()
        print(usage_message)
