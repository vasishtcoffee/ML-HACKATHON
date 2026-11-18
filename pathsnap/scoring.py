"""Candidate scoring for the development demo."""

import numpy as np


def score_candidates(rows):
    """Return a display name and one match score in [0, 1] per row, in input order."""
    return "Distance baseline", np.exp(-0.5 * (rows["distance_to_arc_m"].to_numpy() / 20.0) ** 2)
