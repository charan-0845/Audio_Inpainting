"""Single-sample deep-prior optimization."""


def optimize_deep_prior(model, noise, observed, mask, optimizer, epochs):
    """Optimize network parameters for one corrupted audio sample.

    The network output is fitted to the observed regions while the
    missing regions are left for the implicit neural prior to infer.

    TODO: implement optimization loop.
    """
    raise NotImplementedError
