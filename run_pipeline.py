import telegram_scraper
import preprocess_data
import prepare_dataset
import train_text_model
import train_image_model
import classify_data
import store_data
import dashboard
import logging
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting pipeline...")
    logger.info("Step 1: Scraping data")
    asyncio.run(telegram_scraper.scrape_data())  # Properly await the async function
    logger.info("Step 2: Preprocessing data")
    preprocess_data.preprocess_data()
    logger.info("Step 3: Preparing dataset")
    prepare_dataset.prepare_dataset()
    logger.info("Step 4: Training text model")
    train_text_model.train_text_model()
    logger.info("Step 5: Training image model")
    train_image_model.train_image_model()
    logger.info("Step 6: Classifying data")
    classify_data.classify_data()
    logger.info("Step 7: Storing data")
    store_data.store_data()
    logger.info("Step 8: Starting dashboard")
    dashboard.app.run(host="0.0.0.0", port=5000, debug=True)