"""time-experiment: probing how LLMs represent elapsed conversational time.

Sibling of llmoji-study / attractor-study. Reuses saklas for model loading
+ the Mahalanobis whitener and llmoji_study.config for the shared model
registry; everything else (procedural timestamped transcripts, the elicitation-
slot capture, the single-layer log-elapsed probe, the 3-way decode) is owned
here. The elapsed-time probe is canonicalized as the prefilled answer to a time
elicitation prompt; see DESIGN.md.
"""

__version__ = "0.0.1"
