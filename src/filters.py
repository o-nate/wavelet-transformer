import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def design_fir_filter(cutoff, fs, numtaps=51, window="hamming"):
    """
    Design a Finite Impulse Response (FIR) filter.

    Parameters:
    -----------
    cutoff : float or array_like
        Cutoff frequency in Hz. Can be a single value for lowpass/highpass,
        or an array for bandpass/bandstop.
    fs : float
        Sampling frequency in Hz
    numtaps : int, optional
        Length of the filter (number of coefficients). Must be odd.
        Default is 51.
    window : str, optional
        Windowing method. Default is 'hamming'.

    Returns:
    --------
    numpy.ndarray
        FIR filter coefficients
    """
    nyq = 0.5 * fs

    if isinstance(cutoff, (int, float)):
        # Lowpass or highpass filter
        normalized_cutoff = cutoff / nyq

        # Lowpass filter
        fir_coefs = signal.firwin(numtaps, normalized_cutoff, window=window)
    else:
        # Bandpass or bandstop filter
        normalized_cutoff = np.array(cutoff) / nyq

        fir_coefs = signal.firwin(
            numtaps, normalized_cutoff, pass_zero=False, window=window  # Bandpass
        )

    return fir_coefs


def apply_fir_filter(data, fir_coefs):
    """
    Apply FIR filter to time series data.

    Parameters:
    -----------
    data : numpy.ndarray
        Input time series data
    fir_coefs : numpy.ndarray
        FIR filter coefficients

    Returns:
    --------
    numpy.ndarray
        Filtered time series data
    """
    return signal.lfilter(fir_coefs, 1.0, data)


def visualize_fir_filtering(
    fs, original_signal, lowpass_filtered, bandpass_filtered, fir_coefs
):
    """
    Visualize FIR filtering results and filter characteristics.

    Parameters:
    -----------
    fs : float
        Sampling frequency
    original_signal : numpy.ndarray
        Original time series data
    lowpass_filtered : numpy.ndarray
        Lowpass filtered data
    bandpass_filtered : numpy.ndarray
        Bandpass filtered data
    fir_coefs : numpy.ndarray
        FIR filter coefficients for frequency response
    """
    # Time domain plot
    t = np.linspace(0, 1, fs, endpoint=False)

    plt.figure(figsize=(15, 10))
    plt.suptitle("FIR Filtering Analysis", fontsize=16)

    # Time domain signals
    plt.subplot(2, 2, 1)
    plt.title("Time Domain Signals")
    plt.plot(t, original_signal, label="Original Signal")
    plt.plot(t, lowpass_filtered, label="Lowpass Filtered", alpha=0.7)
    plt.plot(t, bandpass_filtered, label="Bandpass Filtered", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    # Frequency domain (FFT) analysis
    def plot_fft(signal_data, label):
        fft = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data), 1 / fs)
        plt.plot(freqs[: len(freqs) // 2], np.abs(fft[: len(fft) // 2]), label=label)

    plt.subplot(2, 2, 2)
    plt.title("Frequency Domain (FFT)")
    plot_fft(original_signal, "Original Signal")
    plot_fft(lowpass_filtered, "Lowpass Filtered")
    plot_fft(bandpass_filtered, "Bandpass Filtered")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.xlim(0, 150)

    # Filter impulse response
    plt.subplot(2, 2, 3)
    plt.title("FIR Filter Impulse Response")
    plt.stem(fir_coefs, use_line_collection=True)
    plt.xlabel("Tap")
    plt.ylabel("Coefficient Value")

    # Frequency response
    plt.subplot(2, 2, 4)
    w, h = signal.freqz(fir_coefs)
    plt.title("FIR Filter Frequency Response")
    plt.plot(0.5 * fs * w / np.pi, np.abs(h))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain")
    plt.tight_layout()

    plt.show()


def main():
    """
    Example demonstrating FIR filtering workflow with visualization
    """
    # Generate sample time series data
    fs = 1000  # Sampling frequency: 1000 Hz
    t = np.linspace(0, 1, fs, endpoint=False)

    # Create a signal with multiple frequency components
    original_signal = (
        np.sin(2 * np.pi * 10 * t)  # 10 Hz component
        + np.sin(2 * np.pi * 50 * t)  # 50 Hz component
        + np.sin(2 * np.pi * 100 * t)  # 100 Hz component
    )

    # Low-pass filter to remove high-frequency components
    lowpass_coefs = design_fir_filter(cutoff=30, fs=fs, numtaps=51)  # 30 Hz cutoff

    lowpass_filtered = apply_fir_filter(original_signal, lowpass_coefs)

    # Band-pass filter to isolate specific frequency range
    bandpass_coefs = design_fir_filter(
        cutoff=[20, 80], fs=fs, numtaps=101  # 20-80 Hz band
    )

    bandpass_filtered = apply_fir_filter(original_signal, bandpass_coefs)

    # Visualize the results
    visualize_fir_filtering(
        fs, original_signal, lowpass_filtered, bandpass_filtered, bandpass_coefs
    )

    return {
        "original": original_signal,
        "lowpass_filtered": lowpass_filtered,
        "bandpass_filtered": bandpass_filtered,
    }


# Usage example
if __name__ == "__main__":
    filtering_results = main()
