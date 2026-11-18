# Route reconstruction handoff (Person 3)

Scope: HMM/Viterbi decoding, classifier-only versus decoded evaluation,
FastAPI, a Folium demo, two-laptop setup and the corresponding report sections.

## Inputs from teammates

- Person 1: original directed arc IDs and endpoints, node coordinates and arc
  lengths; observations with IDs, timestamps and gap flags; candidate exports
  with `fraction_along_arc` and `offset_from_arc_start_m`.
- Person 2: one score per `(split, trip_id, obs_id, arc_id)` candidate key,
  model/version identifier and the exact score meaning (probability, log
  probability or uncalibrated decision value). Do not infer meaning from range.

Join scores by the full key, never by row order. Keep trip IDs as strings.
Candidate ranks are distance ranks, not classifier scores. Use the observations
as the complete timeline: an observation may have no candidates.

## First milestone

Use a tiny synthetic directed graph to test decoding across ordinary steps,
GPS gaps, missing candidates and disconnected alternatives. Then select that
synthetic trip in the UI, call FastAPI and display the decoded route with a
clear synthetic-data label. Agree on score conversion and gap/restart rules
before connecting a trained classifier. Never invent an arc to fill a failure.

The backend will calculate evaluation metrics using reference data kept on the
server. Return predictions, metrics and failure counts to the UI, not reference
labels. Final route comparisons and report results must use real trips; synthetic
checks are development evidence only.

`route_reconstruction/viterbi.py` now provides a log-space decoding core with
variable candidate counts and explicit rejection of disconnected sequences.
A separate Person 3 implementation now supplies graph transitions, gap splitting, the API and UI for the development demo; the standalone log-space core remains available.

## Progress and remaining deliverables (Person 3)

Implemented in this development branch:
- Directed shortest-path routing, Viterbi decoding, GPS-gap and unreachable-transition handling.
- FastAPI trip selection/predictions, server-side student-test reference evaluation, and Folium UI.
- Configurable distance, Logistic Regression and KNN probability scoring. The two trained
  options read Person 2's committed student-test score exports once those main-branch
  artifacts are present. Each candidate is joined by `(split, trip_id, obs_id, arc_id)`,
  with duplicate, missing, extra, and invalid scores rejected.
- One keyed score array feeds both independent top-1 and Viterbi comparisons.
- Unit tests for keyed score alignment, invalid scores and score-file mismatches.
- Directed-graph and decoder regression tests for one-way routing, intermediate arcs,
  disconnected components, GPS gaps, missing candidates and invalid inputs.
- Classifier-only top-1 evaluation uses the same deterministic baseline tie-break:
  highest score, then lowest `candidate_rank`, then lowest `arc_id`.

The current API / UI is a **development** demo, not the final evaluation setup.
It does not yet demonstrate that Viterbi improves trained-classifier accuracy.
Do not claim such improvements without recording actual results.

Remaining Person 3 tasks:
1. Run the two trained classifiers through decoding across all ten student-test trips;
   record per-trip and aggregate top-1 and Viterbi accuracy, coverage, route
   continuity, failure/gap statistics and inference times. Report regressions too.
2. Verify model selection and Viterbi transition/gap parameters from permitted
   development data; protect the reserved evaluator split.
3. Add final server-side classifier inference for evaluator-trip candidates;
   precomputed student-test scores cannot be used for these reserved trips.
4. Demonstrate UI on Laptop 2 communicating with Laptop 1's model/API for all
   evaluator trips; keep references server-side, with no hardcoded predictions.
5. Write report sections 5 (HMM/Viterbi), 6 (Folium) and 7 (API/UI/deployment)
   using the team's actual experimental evidence and screenshots.

Run score-loader tests: `python -m unittest discover -s tests -v`.
