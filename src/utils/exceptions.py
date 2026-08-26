"""Custom exceptions for the Crop Stress Intelligence System."""


class DataCompletenessError(Exception):
    """Raised when an input time series lacks expected temporal continuity or coverage."""
    pass
