import os
import json
import matplotlib.pyplot as plt

def plot_loss(log_dir='data/conversation_logs'):
    losses = []
    for log_file in os.listdir(log_dir):
        with open(os.path.join(log_dir, log_file), 'r') as f:
            log_data = json.load(f)
            if 'loss' in log_data:
                losses.append(log_data['loss'])

    plt.plot(losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.show()

if __name__ == "__main__":
    plot_loss()
