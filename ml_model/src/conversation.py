class Conversation:
    def __init__(self):
        self.state = {}

    def update_state(self, user_id, key, value):
        if user_id not in self.state:
            self.state[user_id] = []
        self.state[user_id].append({key: value})

    def get_context(self, user_id):
        if user_id in self.state:
            context = ""
            for item in self.state[user_id][-5:]:  # Limit context to last 5 exchanges
                for k, v in item.items():
                    context += f"{k}: {v}\n"
            return context.strip()
        return ""
