import json
import os
import logging
from datetime import datetime
from telegram import Bot
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Telegram bot configuration
TOKEN = os.getenv('8016197633:AAHumqNqdV-7BRkuR8SNF5r6P2sF2QvuDls')
CHAT_ID = os.getenv('-1002655081001')

# File paths
RESULTS_FILE = "data/results.json"
BANNED_USERS_FILE = "banned_users.json"

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def load_banned_users():
    if os.path.exists(BANNED_USERS_FILE):
        with open(BANNED_USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle both list of strings and list of dictionaries
            if data and isinstance(data[0], str):
                # Convert list of strings to list of dictionaries
                return [{"username": username, "reason": "Posted drug-related content", "timestamp": "Unknown"} for username in data]
            return data
    return []

def save_banned_users(banned_users):
    with open(BANNED_USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(banned_users, f, indent=4, ensure_ascii=False)
    logger.info("Saved banned_users.json successfully")

async def send_telegram_alert(bot, alert_text):
    if not TOKEN or not CHAT_ID:
        logger.error("Telegram bot token or chat ID not configured in .env")
        return False

    try:
        await bot.send_message(chat_id=CHAT_ID, text=alert_text)
        logger.info(f"Telegram alert sent to chat {CHAT_ID}")
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")
        return False

async def send_alerts():
    results = load_results()
    banned_users = load_banned_users()
    banned_usernames = {user["username"] for user in banned_users if isinstance(user, dict) and "username" in user}

    bot = Bot(token=TOKEN)
    new_alerts_count = 0

    for entry in results:
        # Skip if already alerted
        if entry.get("alerted"):
            continue

        # Check if the entry is drug-related
        is_drug_related = (
            entry.get("text_label") == "drug-related" or
            entry.get("image_label") == "drug-related"
        )

        if is_drug_related:
            user = entry.get("user", "Unknown")
            timestamp = entry.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            # Prepare alert text
            alert_text = f"Drug-related content detected!\nUser: {user}\nTimestamp: {timestamp}\n"
            if entry.get("message"):
                alert_text += f"Message: {entry['message']}\n"
            if entry.get("image_path"):
                alert_text += f"Image: {entry['image_path']}\n"

            # Send alert
            success = await send_telegram_alert(bot, alert_text)
            if success:
                entry["alerted"] = True
                new_alerts_count += 1

            # Update banned users
            if user != "Unknown" and user not in banned_usernames:
                banned_users.append({
                    "username": user,
                    "reason": "Posted drug-related content",
                    "timestamp": timestamp
                })
                banned_usernames.add(user)

    # Save updated results and banned users
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    save_banned_users(banned_users)
    logger.info(f"Sent {new_alerts_count} new alerts")