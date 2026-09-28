"""tarski: local decision models.

A frozen encoder (the trunk) reads each message once; every decision is a small branch, trained on
your own labels, that reads the trunk's hidden states at its own depth. See README.md.
"""

__version__ = "0.2.0"
