import os
import json
import logging
from datetime import datetime
from telethon.sync import TelegramClient
from telethon.tl.types import InputMessagesFilterPhotos
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Telegram API credentials
api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")
phone = os.getenv("TELEGRAM_PHONE")

# Initialize Telegram client
client = TelegramClient('session_name', api_id, api_hash)

async def scrape_data():
    try:
        await client.start(phone=phone)
        logger.info("Telegram client started successfully")

        # Specify the Telegram channel
        channel_username = '@DrugMonitoringChannel'  # Replace with your channel username
        channel = await client.get_entity(channel_username)

        # Create directory for images
        os.makedirs("data/telegram_images", exist_ok=True)

        messages_data = []
        # Scrape the last 100 messages
        async for message in client.iter_messages(channel, limit=100):
            timestamp = message.date.strftime("%Y-%m-%d %H:%M:%S")
            user = message.sender.username if message.sender and message.sender.username else "Anonymous"
            
            # Handle text messages
            if message.text and not message.media:
                messages_data.append({
                    "timestamp": timestamp,
                    "user": user,
                    "type": "text",
                    "message": message.text,
                    "file_id": None,
                    "media_group_id": message.grouped_id
                })
            
            # Handle media (photos)
            if message.photo:
                file_id = str(message.photo.id)
                filename = f"{user}_{timestamp.replace(':', '')}_fileid{message.id}.jpg"
                image_path = os.path.join("data/telegram_images", filename)
                
                # Download the photo
                await client.download_media(message, image_path)
                logger.info(f"Downloaded image to {image_path}")

                messages_data.append({
                    "timestamp": timestamp,
                    "user": user,
                    "type": "photo",
                    "message": message.message or "",
                    "file_id": file_id,
                    "media_group_id": message.grouped_id,
                    "image_path": f"telegram_images/{filename}"
                })

        # Save scraped data to JSON
        os.makedirs("data", exist_ok=True)
        with open("data/data_log.json", "w", encoding="utf-8") as f:
            json.dump(messages_data, f, indent=4, ensure_ascii=False)
            logger.info("Saved scraped data to data/data_log.json")

    except Exception as e:
        logger.error(f"Error during scraping: {e}")
    finally:
        await client.disconnect()
        logger.info("Telegram client disconnected")

if __name__ == "__main__":
    import asyncio
    asyncio.run(scrape_data())