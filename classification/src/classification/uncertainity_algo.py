import numpy as np


import numpy as np

def entropy(probabilities):
    """Compute the entropy of a probability distribution."""
    probabilities = np.array(probabilities)
    probabilities = np.clip(probabilities, 1e-10, 1)  # Avoid log(0)
    return -np.sum(probabilities * np.log2(probabilities))

# Example: 4-class probability distributions
probs1 = [0.25, 0.25, 0.25, 0.25]  # Maximum uncertainty
probs2 = [0.99, 0.0, 0.0, 0.01]   # Low uncertainty

print("Entropy 1:", entropy(probs1))  # High entropy
print("Entropy 2:", entropy(probs2))  # Low entropy


def generate_probabilities(num_samples=10, num_classes=4):
    """Generate random probability distributions that sum to 1."""
    samples = np.random.dirichlet(np.ones(num_classes), size=num_samples)
    return samples


class UncertaintyTracker:
    def __init__(self, window_size=5, alpha=0.7):
        self.window_size = window_size
        self.alpha = alpha  # For exponential moving average
        self.prob_history = []  # Stores past probability distributions
        self.entropy_history = []  # Stores past entropy values
    
    def entropy(self, probabilities):
        """Compute entropy of a probability distribution."""
        probabilities = np.clip(probabilities, 1e-10, 1)  # Avoid log(0)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def update(self, probs):
        """Update the history with new probabilities and compute uncertainty."""
        ent = self.entropy(probs)

        # Store probabilities and entropy
        self.prob_history.append(probs)
        self.entropy_history.append(ent)

        # Keep only the last `window_size` elements
        if len(self.prob_history) > self.window_size:
            self.prob_history.pop(0)
            self.entropy_history.pop(0)
        
        # Compute moving average of entropy
        moving_avg_entropy = np.mean(self.entropy_history)

        # Compute exponential moving average for probabilities
        smoothed_probs = np.average(self.prob_history, axis=0, weights=np.linspace(1, self.alpha, len(self.prob_history)))

        return smoothed_probs, moving_avg_entropy









"""
# Generate 20 test cases
test_cases = generate_probabilities(num_samples=20)

# Print test cases
for i, probs in enumerate(test_cases):
    print(f"Case {i+1}: {probs}")
    print("Entropy:", entropy(probs))
    print()
"""





"""
# Example usage
tracker = UncertaintyTracker(window_size=3)

test_probs = [
    [0.9, 0.05, 0.03, 0.02],  # High confidence
    [0.85, 0.08, 0.05, 0.02],  # Still confident
    [0.7, 0.2, 0.08, 0.02],  # Slightly more uncertain
    [0.6, 0.25, 0.1, 0.05],  # More uncertain
    [0.5, 0.3, 0.15, 0.05],  # High uncertainty
    [0.3, 0.3, 0.2, 0.2],  # Very uncertain
    [0.4, 0.4, 0.1, 0.1],  # Uncertainty persists
    [0.6, 0.3, 0.08, 0.02],  # Slight recovery
    [0.75, 0.15, 0.07, 0.03],  # Confidence increasing
    [0.85, 0.1, 0.03, 0.02],  # Almost confident again
    [0.9, 0.05, 0.03, 0.02],  # Fully confident again
    [0.8, 0.1, 0.05, 0.05],  # Slight dip but still confident
    [0.6, 0.25, 0.1, 0.05],  # Sudden uncertainty
    [0.5, 0.3, 0.15, 0.05],  # Uncertainty remains
    [0.4, 0.3, 0.2, 0.1],  # Even more uncertain
    [0.3, 0.3, 0.2, 0.2],  # Highly uncertain
    [0.5, 0.3, 0.15, 0.05],  # Starting to recover
    [0.6, 0.25, 0.1, 0.05],  # Getting better
    [0.75, 0.15, 0.07, 0.03],  # Near confident again
    [0.9, 0.05, 0.03, 0.02],  # Back to full confidence
]


for i, probs in enumerate(test_probs):
    smoothed_probs, avg_entropy = tracker.update(probs)
    print(f"Step {i+1}: Smoothed Probs: {smoothed_probs}, Avg Entropy: {avg_entropy}")
"""