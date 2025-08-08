from flask import Flask, request, jsonify, Response
import logging
import sys
import os
import json
import queue
import threading
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import from sibling directories
sys.path.append(str(Path(__file__).parent.parent))

from models.nlp_model import NLPModel
from models.cv_model import CVModel
from analysis.text_analysis import analyze_text
from analysis.image_analysis import analyze_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize models
nlp_model = NLPModel()
cv_model = CVModel()

# Message queue for SSE
message_queue = queue.Queue()

def format_sse(data: dict, event=None) -> str:
    """Format data for SSE"""
    msg = f'data: {json.dumps(data)}\n\n'
    if event is not None:
        msg = f'event: {event}\n{msg}'
    return msg

def broadcast_event(data: dict, event=None):
    """Add message to queue for all listeners"""
    message_queue.put((data, event))

@app.route('/api/events')
def events():
    """SSE endpoint for real-time updates"""
    def event_stream():
        while True:
            try:
                data, event = message_queue.get(timeout=20)  # 20 second timeout
                yield format_sse(data, event)
            except queue.Empty:
                yield format_sse({"type": "ping"}, "ping")  # Keep connection alive
    
    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/predict/text', methods=['POST'])
def predict_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text']
        result = nlp_model.predict(text)
        
        # Additional analysis
        analysis = analyze_text(text)
        
        # Prepare response data
        response_data = {
            "prediction": result,
            "analysis": analysis,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "type": "text",
            "content": text[:100] + "..." if len(text) > 100 else text  # Truncate long text
        }
        
        # Broadcast event for real-time updates
        broadcast_event(
            {"detection": response_data},
            "new_detection"
        )
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in text prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/image', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save temporarily
        temp_path = os.path.join("temp", image_file.filename)
        os.makedirs("temp", exist_ok=True)
        image_file.save(temp_path)

        # Make prediction
        result = cv_model.predict(temp_path)
        
        # Additional analysis
        analysis = analyze_image(temp_path)
        
        # Prepare response data
        response_data = {
            "prediction": result,
            "analysis": analysis,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "type": "image",
            "filename": image_file.filename
        }
        
        # Broadcast event for real-time updates
        broadcast_event(
            {"detection": response_data},
            "new_detection"
        )
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in image prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze/bulk', methods=['POST'])
def analyze_bulk():
    try:
        data = request.get_json()
        if not data or 'items' not in data:
            return jsonify({"error": "No items provided"}), 400

        results = []
        for item in data['items']:
            if 'type' not in item or 'content' not in item:
                continue
                
            if item['type'] == 'text':
                result = nlp_model.predict(item['content'])
                analysis = analyze_text(item['content'])
            elif item['type'] == 'image':
                # Assuming content is a base64 encoded image or image URL
                result = "Not implemented"  # Implement based on your needs
                analysis = {}
            else:
                continue
                
            results.append({
                "id": item.get('id'),
                "type": item['type'],
                "prediction": result,
                "analysis": analysis
            })

        return jsonify({
            "results": results,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error in bulk analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/timeline')
def get_timeline_data():
    try:
        # Load results from the data file
        results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'results.json')
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Group detections by hour
        timeline = {}
        for result in results:
            timestamp = datetime.fromisoformat(result.get('timestamp', ''))
            hour = timestamp.replace(minute=0, second=0, microsecond=0).isoformat()
            
            if hour not in timeline:
                timeline[hour] = {
                    'total': 0,
                    'drug_related': 0
                }
            
            timeline[hour]['total'] += 1
            if result.get('prediction') == 'drug-related':
                timeline[hour]['drug_related'] += 1
        
        # Convert to sorted list format
        sorted_data = []
        for hour in sorted(timeline.keys()):
            sorted_data.append({
                'hour': hour,
                'total': timeline[hour]['total'],
                'drug_related': timeline[hour]['drug_related']
            })
        
        return jsonify({
            'status': 'success',
            'data': sorted_data
        })
    except Exception as e:
        logger.error(f"Error getting timeline data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats')
def get_stats():
    try:
        # Load results from the data file
        results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'results.json')
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Calculate statistics
        total_posts = len(results)
        drug_related = sum(1 for r in results if r.get('prediction') == 'drug-related')
        percentage = (drug_related / total_posts * 100) if total_posts > 0 else 0
        
        # Get unique users with drug-related content
        flagged_users = set()
        for result in results:
            if result.get('prediction') == 'drug-related':
                flagged_users.add(result.get('user', 'unknown'))
        
        # Get recent activity (last 24 hours)
        current_time = datetime.now()
        recent_activity = []
        for result in sorted(results, key=lambda x: x.get('timestamp', ''), reverse=True):
            timestamp = datetime.fromisoformat(result.get('timestamp', ''))
            if (current_time - timestamp).total_seconds() <= 86400:  # 24 hours
                recent_activity.append(result)
            if len(recent_activity) >= 10:  # Limit to 10 most recent
                break
        
        return jsonify({
            'status': 'success',
            'stats': {
                'total_posts': total_posts,
                'drug_related': drug_related,
                'percentage': round(percentage, 1),
                'flagged_users': len(flagged_users),
                'recent_activity': recent_activity
            }
        })
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)