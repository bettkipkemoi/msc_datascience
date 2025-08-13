#Fast Fourier Transform
import numpy as np
import matplotlib.pyplot as plt

def fft(x):
    """
    Compute the Fast Fourier Transform of the input sequence x.
    Input x must have length that is a power of 2.
    Returns the frequency domain representation.
    """
    N = len(x)
    if N <= 1:
        return x
    if N % 2 != 0:
        raise ValueError("Length of input must be a power of 2")

    # Divide into even and odd indices
    even = fft(x[0::2])
    odd = fft(x[1::2])

    # Initialize output array
    X = np.zeros(N, dtype=complex)
    
    # Combine results using twiddle factors
    for k in range(N // 2):
        twiddle = np.exp(-2j * np.pi * k / N)
        X[k] = even[k] + twiddle * odd[k]
        X[k + N // 2] = even[k] - twiddle * odd[k]
    
    return X

# Generate a test signal
def generate_test_signal():
    # Parameters
    fs = 1000  # Sampling frequency (Hz)
    T = 1.0    # Duration (seconds)
    t = np.linspace(0, T, int(fs * T), endpoint=False)  # Time vector
    freq = 50  # Frequency of sine wave (Hz)
    
    # Signal: sum of two sine waves (50 Hz and 120 Hz)
    signal = 1.0 * np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    return t, signal, fs

# Main function to demonstrate FFT
def main():
    # Generate signal
    t, signal, fs = generate_test_signal()
    N = len(signal)
    
    # Ensure signal length is a power of 2
    N = 2**int(np.log2(N))
    signal = signal[:N]
    t = t[:N]
    
    # Compute FFT
    fft_result = fft(signal)
    
    # Compute frequency bins
    freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
    magnitude = np.abs(fft_result)[:N//2] * 2 / N  # Normalize amplitude
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot original signal
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plot frequency spectrum
    plt.subplot(2, 1, 2)
    plt.plot(freqs, magnitude)
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('fft_result.png')
    plt.show()

if __name__ == "__main__":
    main()