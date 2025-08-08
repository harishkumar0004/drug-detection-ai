import json
import os
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from pathlib import Path
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def load_results():
    """Load and validate results from JSON file"""
    results_file = Path("data/results.json")
    
    if not results_file.exists():
        logger.warning("Results file not found!")
        return []
    
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
            
            # Validate each entry has required fields
            valid_results = []
            for item in results:
                if all(k in item for k in ["timestamp", "user", "type"]):
                    valid_results.append(item)
                else:
                    logger.warning(f"Skipping invalid entry: {item}")
            
            return valid_results
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Error loading results: {e}")
        return []

def save_results(results):
    """Save results to JSON file"""
    results_file = Path("data/results.json")
    try:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info("Successfully saved results to results.json")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def analyze_data(results):
    """Process results data for dashboard visualization"""
    analysis = {
        "drug_count": 0,
        "safe_count": 0,
        "time_series": defaultdict(int),
        "notifications": []
    }
    
    now = datetime.now()
    last_24h = now - timedelta(hours=24)
    
    for result in results:
        text_label = result.get("text_label", "N/A")
        image_label = result.get("image_label", "N/A")
        
        # Determine if drug-related
        is_drug_related = text_label == "drug-related" or image_label == "drug-related"
        
        # Update counters
        if is_drug_related:
            analysis["drug_count"] += 1
        elif text_label == "safe" or image_label == "safe":
            analysis["safe_count"] += 1
        else:
            logger.warning(f"Unclassified entry: {result}")
        
        # Time series data (daily drug-related posts)
        try:
            timestamp = datetime.strptime(result["timestamp"], "%Y-%m-%d %H:%M:%S")
            if is_drug_related:
                date_key = timestamp.strftime("%Y-%m-%d")
                analysis["time_series"][date_key] += 1
        except ValueError:
            logger.warning(f"Invalid timestamp format: {result['timestamp']}")
        
        # Notifications (last 24 hours)
        try:
            timestamp = datetime.strptime(result["timestamp"], "%Y-%m-%d %H:%M:%S")
            if is_drug_related and timestamp >= last_24h:
                analysis["notifications"].append(result)
        except ValueError:
            logger.warning(f"Invalid timestamp for notification: {result['timestamp']}")

    logger.info(f"Analysis - Drug count: {analysis['drug_count']}, Safe count: {analysis['safe_count']}")
    return analysis

@app.route('/')
def dashboard():
    results = load_results()
    
    # Sort by timestamp (newest first)
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Process data for visualization
    analysis = analyze_data(results)
    
    # Prepare time series data
    dates = sorted(analysis["time_series"].keys())
    counts = [analysis["time_series"][date] for date in dates]
    
    # Get current time in IST
    ist = pytz.timezone("Asia/Kolkata")
    current_time = datetime.now(ist)
    formatted_time = current_time.strftime("%I:%M %p IST on %A, %B %d, %Y")
    
    return render_template(
        "dashboard.html",
        results=results[:200],  # Limit to 200 most recent
        drug_count=analysis["drug_count"],
        safe_count=analysis["safe_count"],
        dates=dates,
        counts=counts,
        notifications=analysis["notifications"][:10],  # Last 10 alerts
        current_time=formatted_time
    )

@app.route('/update_action', methods=['POST'])
def update_action():
    """Handle action updates (ban or ignore) for a post"""
    data = request.get_json()
    timestamp = data.get('timestamp')
    user = data.get('user')
    action = data.get('action')

    if action not in ['banned', 'ignored']:
        logger.error(f"Invalid action: {action}")
        return jsonify({"status": "error", "message": "Invalid action"}), 400

    # Load results
    results = load_results()

    # Find the entry and update the action
    updated = False
    for entry in results:
        if entry["timestamp"] == timestamp and entry["user"] == user:
            entry["action"] = action
            updated = True
            logger.info(f"Updated action for user {user} at {timestamp} to {action}")
            break

    if not updated:
        logger.warning(f"Entry not found for timestamp {timestamp} and user {user}")
        return jsonify({"status": "error", "message": "Entry not found"}), 404

    # Save updated results
    save_results(results)
    return jsonify({"status": "success", "action": action})

@app.route('/api/data')
def api_data():
    """Endpoint for AJAX data requests"""
    results = load_results()
    return jsonify({
        "total": len(results),
        "drug_related": sum(1 for r in results if r.get("text_label") == "drug-related" or r.get("image_label") == "drug-related")
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)