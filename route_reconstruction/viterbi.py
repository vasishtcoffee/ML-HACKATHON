"""Small log-space Viterbi core; graph transitions are supplied by the caller."""

from math import isfinite, inf


def decode(emissions, transitions):
    """Return candidate indices for the highest-scoring complete sequence.

    emissions[t][j] is a log emission score for candidate j at observation t.
    transitions[t][i][j] connects observations t and t+1. Candidate counts may
    vary. Use negative infinity for impossible candidates or transitions.
    Start states have equal prior weight. Scores need not be normalized.
    Ties select the first candidate in input order. Empty observations and a
    disconnected sequence raise ValueError; callers must split gaps explicitly.
    """
    if len(transitions) != max(0, len(emissions) - 1):
        raise ValueError("Expected one transition matrix between each observation.")
    if not emissions:
        return []

    def valid_score(value):
        return isfinite(value) or value == -inf

    for scores in emissions:
        if not scores or not all(valid_score(value) for value in scores):
            raise ValueError("Each observation needs candidates with finite or -inf scores.")
    for t, matrix in enumerate(transitions):
        if len(matrix) != len(emissions[t]) or any(
            len(row) != len(emissions[t + 1]) for row in matrix
        ):
            raise ValueError(f"Transition matrix {t} has incompatible dimensions.")
        if not all(valid_score(value) for row in matrix for value in row):
            raise ValueError("Transition scores must be finite or -inf.")

    scores = list(emissions[0])
    backpointers = []
    if max(scores) == -inf:
        raise ValueError("No reachable candidate at observation 0.")
    for t, matrix in enumerate(transitions):
        previous = []
        next_scores = []
        for j, emission in enumerate(emissions[t + 1]):
            choices = [score + matrix[i][j] for i, score in enumerate(scores)]
            best = max(range(len(choices)), key=choices.__getitem__)
            previous.append(best)
            next_scores.append(choices[best] + emission)
        if max(next_scores) == -inf:
            raise ValueError(f"No complete route reaches observation {t + 1}.")
        scores = next_scores
        backpointers.append(previous)

    state = max(range(len(scores)), key=scores.__getitem__)
    path = [state]
    for previous in reversed(backpointers):
        state = previous[state]
        path.append(state)
    return path[::-1]
