"""Test data smoothing functions"""

# %%

import pywt

from src import dwt
from src import retrieve_data

from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

# * Constants
MOTHER = pywt.Wavelet("db4")

print("Testing trim_signal catches odd-numbered signals")
test_signal = list(range(1000))
print(f"Signal length: {len(test_signal)}")
trim = dwt.trim_signal(test_signal, test_signal)
assert len(trim) == len(test_signal)
print(f"Signal length: {len(test_signal)}")
test_signal = list(range(1001))
print(f"Signal length: {len(test_signal)}")
trim = dwt.trim_signal(test_signal, test_signal)
assert len(trim) != len(test_signal)

raw_data = retrieve_data.get_insee_data("000857179")
_, t, y = retrieve_data.clean_insee_data(raw_data)


logger.info("Test creation of DataForDWT class instance")
data_for_dwt = dwt.DataForDWT(y, MOTHER)
assert len(data_for_dwt.y_values) == len(y)
assert data_for_dwt.levels is None
logger.info("Test passed")

logger.info("Testing DWT with INSEE data")
# * Apply DWT and smooth signal
results_from_dwt = dwt.run_dwt(data_for_dwt)
assert data_for_dwt.levels is None
logger.info("Test passed")

logger.info("Testing smooth signal with INSEE data")
# * Apply DWT and smooth signal
results_from_dwt = dwt.smooth_signal(data_for_dwt)
assert results_from_dwt.levels is not None
assert len(
    results_from_dwt.smoothed_signal_dict[results_from_dwt.levels]["signal"]
) == len(y)
logger.info("Test passed")

print("DWT testing complete.")
