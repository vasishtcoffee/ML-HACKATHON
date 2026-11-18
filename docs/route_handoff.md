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
checks are development evidence only. No decoder, API or UI is implemented yet.
