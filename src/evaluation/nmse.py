"""NMSE metrics for audio reconstruction."""


def nmse(reference, reconstruction):
    """Compute normalized mean squared error."""
    raise NotImplementedError


def missing_region_nmse(reference, reconstruction, mask):
    """Compute NMSE only on originally missing samples."""
    raise NotImplementedError
