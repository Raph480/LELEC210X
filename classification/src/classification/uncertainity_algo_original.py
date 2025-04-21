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

import time

class UncertaintyTracker:
    def __init__(self, window_size=5, alpha=0.7, reset_interval=3):
        self.window_size = window_size
        self.alpha = alpha  # For exponential moving average
        self.reset_interval = reset_interval  # Time in seconds to reset memory
        self.last_reset_time = time.time()
        self.reset_memory()
    
    def reset_memory(self):
        """Resets the stored probability and entropy history."""
        print("MEMORY RESET")
        self.prob_history = []  # Stores past probability distributions
        self.entropy_history = []  # Stores past entropy values
    
    def entropy(self, probabilities):
        """Compute entropy of a probability distribution."""
        probabilities = np.clip(probabilities, 1e-10, 1)  # Avoid log(0)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def update(self, probs):
        """Update the history with new probabilities and compute uncertainty."""
        current_time = time.time()
        
        # Reset memory if the interval has elapsed
        if current_time - self.last_reset_time >= self.reset_interval:
            self.reset_memory()
            self.last_reset_time = current_time
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
        smoothed_probs = np.average(
            self.prob_history, axis=0, weights=np.linspace(1, self.alpha, len(self.prob_history))
        )

        return smoothed_probs, moving_avg_entropy



class DoubleRunningSumDetector:
    def __init__(self, N_STS=50, N_LTS=5000, K=1, BASELINE=2048):
        """Initialize the double running sum algorithm."""
        self.N_STS = N_STS  # Short-term sum window size
        self.N_LTS = N_LTS  # Long-term sum window size
        self.K = K  # Scaling factor (not used explicitly, can be added later)
        self.BASELINE = BASELINE  # Baseline level for signal shift

        # Buffers for short-term and long-term sums
        self.sts_buffer = np.zeros(self.N_STS)
        self.lts_buffer = np.zeros(self.N_LTS)

        # Running sums
        self.sts_sum = 0.0
        self.lts_sum = 0.0

        # Indexes for circular buffer updates
        self.sts_idx = 0
        self.lts_idx = 0

    def update(self, sample, threshold_factor=1.0):
        """
        Process one sample and update the running sums.

        Args:
            sample (float or int): Single ADC sample
            threshold_factor (float): Scaling factor for detection sensitivity

        Returns:
            bool: True if a signal packet is detected, False otherwise
        """
        sample_shifted = float(sample) - self.BASELINE
        sample_mag = abs(sample_shifted)

        # Update short-term sum
        self.sts_sum -= self.sts_buffer[self.sts_idx]
        self.sts_buffer[self.sts_idx] = sample_mag
        self.sts_sum += sample_mag
        self.sts_idx = (self.sts_idx + 1) % self.N_STS

        # Update long-term sum
        self.lts_sum -= self.lts_buffer[self.lts_idx]
        self.lts_buffer[self.lts_idx] = self.sts_buffer[(self.sts_idx + self.N_STS - 1) % self.N_STS]
        self.lts_sum += self.lts_buffer[self.lts_idx]
        self.lts_idx = (self.lts_idx + 1) % self.N_LTS

        # Normalize running sums
        norm_sts = self.sts_sum / self.N_STS
        norm_lts = self.lts_sum / self.N_LTS

        print(f"STS: {norm_sts}, LTS: {threshold_factor * norm_lts}")

        # Detection condition
        return norm_sts > threshold_factor * norm_lts




def decision(payload, probas,i, DTYPE, predicted_class, detector, tracker, a_sum, b_entropy, c_memory, save_payloads=None):
    """Make a decision based on the probabilities and uncertainty."""

    send = True
    
    class_names = ["chainsaw", "fire", "fireworks", "gun"]
    predicted_class_name = class_names[predicted_class]

    print(f"Predicted probabilities: {probas}")
    print("Theoretical class: ", predicted_class_name)

    uncertainity = 0

    #1. Double running sum: TODO: TEST
    adc_samples = np.array(
        [int(payload[i : i + 4], 16) for i in range(0, len(payload), 4)],
        dtype=DTYPE
    ) 

    if not detector.update(adc_samples[i], threshold_factor=2.5):
        uncertainity += a_sum
        print("DOUBLE SUM UNCERTAIN, uncertainity += ", uncertainity)

    #2. Entropy: measure of uncertainty % probability distribution
    en = entropy(probas)
    
    entropy_normalized = en  * 1/0.7 #If entropy is 0.7, uncertainity is 1 -TODO: TEST
    print("Entropy: ", en)
    print("Entropy normalized: ", entropy_normalized)

    uncertainity += b_entropy * en * entropy_normalized 
    print("Entropy uncertainity += ", b_entropy * en * entropy_normalized )

    #3. Memory effect:
    #TODO: SEARCH IF GUN CLASS BEHAVIOUR INTERESTING

    smoothed_probs, avg_entropy = tracker.update(probas)
    #print(f"Step {i+1}: Smoothed Probs: {smoothed_probs}, Avg Entropy: {avg_entropy}")
    
    print("Proba predicted: ", probas[0][predicted_class])
    print("Proba smoothed: ", smoothed_probs[0][predicted_class])

    if probas[0][predicted_class] > smoothed_probs[0][predicted_class]:
        uncertainity += c_memory * (probas[0][predicted_class] - smoothed_probs[0][predicted_class])
        print("MEMORY UNCERTAIN, uncertainity += ", c_memory * (probas[0][predicted_class] - smoothed_probs[0][predicted_class]))
    
    print("Final Uncertainity: ", uncertainity)

    if uncertainity > 1:
        send = False
        print("SEND FALSE")
    else:
        send = True
    
    # Save the entropy and smoothed probs/entr
    if save_payloads:
        with open(save_payloads, "a") as save_file:  # mode 'a' to append without overwriting
            #save the theoretical class, the entropy, the normalized entropy, the smoothed probabilities, the average entropy and the uncertainity
            save_file.write("\n" +predicted_class_name)
            save_file.write(f"{i}\t{probas[0]}\t{en}\t{entropy_normalized}\t{smoothed_probs[0]}\t{avg_entropy}\t{uncertainity}\n")

    return send