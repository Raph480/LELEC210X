"""
Test file, provided to easily check your implementations.
"""

import numpy as np
import pytest
from chain import BasicChain
from sim import add_cfo, add_delay


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng()


class TestBasicChain:
    chain = BasicChain()

    @pytest.mark.parametrize("size", (1, 10, 100))
    def test_demodulate(self, rng: np.random.Generator, size: int):
        bits = rng.integers(2, size=size)  # choice of bits to send
        x_pay = self.chain.modulate(bits)  # modulated signal with payload
        x = x_pay

        y, delay = add_delay(
            self.chain, x, 0
        )  # application of ideal channel (if TX and RX oversampling factors are different)

        bits_hat = self.chain.demodulate(y)  # call to demodulation function
        
        # Print the input and output bits
        print(f"Input bits: {bits}")
        print(f"Output (demodulated) bits: {bits_hat}")

        np.testing.assert_equal(bits_hat, bits)

    @pytest.mark.parametrize("size", (1, 10, 100))
    @pytest.mark.parametrize("cfo_val", (0, 100, 1000))
    def test_cfo_estimation(self, rng: np.random.Generator, size: int, cfo_val: float):
        bits = rng.integers(2, size=size)  # choice of bits to send
        x_pay = self.chain.modulate(bits)  # modulated signal with payload

        x_pr = self.chain.modulate(
            self.chain.preamble
        )  # modulated signal containing preamble
        x_sync = self.chain.modulate(
            self.chain.sync_word
        )  # modulated signal containing sync_word
        x = np.concatenate((x_pr, x_sync, x_pay))

        y, delay = add_delay(self.chain, x, 0)  # application of ideal channel
        y_cfo = add_cfo(self.chain, y, cfo_val)  # adding CFO
        cfo_hat = self.chain.cfo_estimation(y_cfo)

        np.testing.assert_allclose(cfo_hat, cfo_val)
