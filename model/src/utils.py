import json
import os
from datetime import datetime

def log_conversation(user_text, ai_text, intent):
    log_dir = "data/conversation_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    conversation_data = {
        "user": user_text,
        "ai": ai_text,
        "intuition": {
            "intent": intent,
            "response": ai_text
        },
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(log_file, 'w') as f:
        json.dump(conversation_data, f, indent=4)
