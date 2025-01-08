# """Correction to PyCWT library Morlet class"""

# # TODO create patches for `cwt` and `wct` functions in wavelet.py

# import functools
# import types

# import matplotlib.pyplot as plt
# import numpy as np
# import numpy.typing as npt

# from pycwt import helpers
# from pycwt.helpers import rect, fft, fft_kwargs
# from pycwt.mothers import Morlet
# from scipy.signal import convolve2d


# from utils.logging_config import get_logger

# # * Logging settings
# logger = get_logger(__name__)

# # * Constants for patches
# DELTAJ0 = 0.60
# # Can be turned off, e.g. for MKL optimizations
# _FFT_NEXT_POW2 = True

# logger.info("Initializing patches")


# # * Wrapper to correct rect function from PyCWT
# def rect_wrapper(func):
#     """Wrapper to correct UnboundLocalError."""

#     @functools.wraps(func)
#     def wrapper(x, normalize=False):
#         try:
#             if isinstance(x, (int, float, np.integer)):
#                 shape = [np.int64(x)]
#             elif isinstance(x, (list, dict)):
#                 shape = x
#             elif isinstance(x, (np.ndarray, np.ma.core.MaskedArray)):
#                 shape = x.shape
#             else:
#                 raise TypeError(f"Unsupported input type: {type(x)}")

#             # Call the original function with the processed input

#             logger.debug(x)
#             logger.debug("shape %s", shape)

#             X = np.zeros(shape, dtype=np.float64)
#             X[0] = X[-1] = 0.5
#             X[1:-1] = 1

#             if normalize:
#                 X /= X.sum()

#             return X

#         except Exception as e:
#             print(f"Error in rect function: {e}")
#             raise

#     return wrapper


# def monkey_patched_fft_kwargs(signal, **kwargs):
#     """
#     Updated patch for fft_kwargs that uses int instead of deprecated np.int

#     Args:
#         signal: Input signal array
#         **kwargs: Additional keyword arguments

#     Returns:
#         dict: Keyword arguments for FFT computation
#     """
#     if helpers._FFT_NEXT_POW2:
#         return {"n": int(2 ** np.ceil(np.log2(len(signal))))}
#     return kwargs


# def monkey_patched_smooth_method(self, W, dt, dj, scales):
#     """Monkey-patch for smooth method in Morlet class, caused by use of
#     deprecated `np.int`"""
#     m, n = W.shape

#     k = 2 * np.pi * fft.fftfreq(fft_kwargs(W[0, :])["n"])
#     k2 = k**2
#     snorm = scales / dt

#     F = np.exp(-0.5 * (snorm[:, np.newaxis] ** 2) * k2)  # Outer product
#     smooth = fft.ifft(
#         F * fft.fft(W, axis=1, **fft_kwargs(W[0, :])),
#         axis=1,  # Along Fourier frequencies
#         **fft_kwargs(W[0, :], overwrite_x=True),
#     )
#     T = smooth[:, :n]  # Remove possibly padded region due to FFT

#     if np.isreal(W).all():
#         T = T.real

#     # Filter in scale. For the Morlet wavelet it's simply a boxcar with
#     # 0.6 width.
#     wsize = DELTAJ0 / dj * 2
#     win = rect(np.int64(np.round(wsize)), normalize=True)
#     T = convolve2d(T, win[:, np.newaxis], "same")  # Scales are "vertical"

#     return T


# # * Modify the original rect function with the wrapper
# original_rect = rect
# rect = rect_wrapper(rect)
# logger.info("Wrapper applied to `rect` function.")

# # * Monkey patch to correct fft_kwargs function from PyCWT
# original_fft_kwargs = fft_kwargs
# helpers.fft_kwargs = monkey_patched_fft_kwargs
# logger.info("Monkey patch applied to `fft_kwargs`.")

# # # Modify the original smooth function since it uses a deprecated numpy type
# Morlet.smooth = types.MethodType(monkey_patched_smooth_method, Morlet)
# logger.info("Monkey-patched wavelet.Morlet.smooth method")
