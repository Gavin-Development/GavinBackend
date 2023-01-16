class BINFile:
    error_msg = "You're possibly running an unsupported version of Python. Or GavinBackendDatasetUtils was not found."

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(self.error_msg)

    @property
    def MaxNumberOfSamples(self):
        raise NotImplementedError(self.error_msg)

    @property
    def max_number_of_samples(self):
        raise NotImplementedError(self.error_msg)

    def __getitem__(self, item):
        raise NotImplementedError(self.error_msg)
