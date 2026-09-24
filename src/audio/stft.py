"""STFT and inverse-STFT utilities."""


def compute_stft(audio, n_fft=1024, hop_length=120, win_length=600):
    """Compute a complex STFT.

    TODO: implement using the selected audio library.
    """
    raise NotImplementedError


def inverse_stft(stft, n_fft=1024, hop_length=120, win_length=600):
    """Convert a complex STFT back to waveform."""
    raise NotImplementedError
