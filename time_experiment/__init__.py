"""time-experiment: probing how LLMs represent elapsed conversational time.

Sibling of llmoji-study / attractor-study. Reuses saklas for model loading
+ the Mahalanobis whitener and llmoji_study.config for the shared model
registry; everything else (procedural timestamped transcripts, EOT capture,
log-elapsed manifold fit, the 3-way decode) is owned here.
"""

__version__ = "0.0.1"
