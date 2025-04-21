import numpy as np
import requests 
DTYPE = np.dtype(np.uint16).newbyteorder("<")
np.set_printoptions(precision=2, suppress=True)


def entropy(probabilities):
    """Compute the entropy of a probability distribution."""
    probabilities = np.array(probabilities)
    probabilities = np.clip(probabilities, 1e-10, 1)  # Avoid log(0)
    return -np.sum(probabilities * np.log2(probabilities))


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

    def reset(self):
        """Reset the detector state."""
        self.sts_buffer.fill(0)
        self.lts_buffer.fill(0)
        self.sts_sum = 0.0
        self.lts_sum = 0.0
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

        #TODO: Adapt to return a value between 0 and 1 instead of True/False
        #If norm_sts > thresh * norm_lts, then value returned is 1
        #else, the returned value is close to one if the difference is small, and close to 0 if the difference is high

        # Compute confidence score between 0 and 1
        if norm_sts > threshold_factor * norm_lts:
            return 1.0
        else:
            # Confidence decreases as STS diverges from threshold
            ratio = norm_sts / (threshold_factor * norm_lts + 1e-6)  # prevent div by 0
            return max(0.0, min(ratio, 1.0))  # clamp to [0, 1]




def compute_uncertainity(payloads, probas, idx, DTYPE, detector, a_sum, b_entropy, save_payloads=None):
    """Returns the uncertainity of the packet given the double sum and model predictions."""    

    predicted_class = np.argmax(probas)
    class_names = ["chainsaw", "fire", "fireworks", "gun"]
    predicted_class_name = class_names[predicted_class]

    print(f"Predicted probabilities: {probas*100}")
    print("Theoretical class: ", predicted_class_name)

    uncertainty = 0

    #WARNING: was originally payload but changed to fv #TO NOT NORMALIZE BEFORE
    #1. Double running sum
    if detector: # Detector set to none for multiple packets
        adc_samples = np.array(
            [int(payloads[i : i + 4], 16) for i in range(0, len(payloads), 4)],
            dtype=DTYPE
        )

        if idx >= len(adc_samples):
            print("Index out of range for adc_samples - init of detector")
            detector.reset()

        if not detector.update(adc_samples[idx], threshold_factor=1.0):
            uncertainty += a_sum
            print("Double sum uncertainity: ", uncertainty)

    #2. Entropy
    en = entropy(probas)
    
    entropy_normalized = en * 1 / 0.7  # Normalize entropy
    #print("Entropy: ", en)
    #print("Normalized entropy: ", entropy_normalized)

    uncertainty += b_entropy * entropy_normalized 
    #print("Entropy uncertainty: ", b_entropy * entropy_normalized)

    #print("Final Uncertainty: ", uncertainty)

    return uncertainty 

