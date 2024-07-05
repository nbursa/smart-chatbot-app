import os
import json
from datetime import datetime

def log_conversation(user_text, ai_text, intuition_data):
    log_dir = 'data/conversation_logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_file = os.path.join(log_dir, f'log_{timestamp}.json')

    log_data = {
        'user': user_text,
        'ai': ai_text,
        'intuition': intuition_data,
        'timestamp': timestamp
    }

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)
