# Team review before submission

The consolidated report is AI-assisted and based on recorded executions. Review
it against the CSVs and code, add the correct team identities if required, and be
prepared to explain these concrete decisions:

1. Why trip-grouped validation prevents observations from the same trip leaking
   into a fit fold and its validation fold.
2. Why an ordered `.route` file cannot be joined row-for-row to GPS samples, and
   how approximate monotone alignment affects the reported denominators.
3. Why direction helps distinguish reverse directed arcs, and why centred
   headings make the current system an offline trip processor.
4. The difference between binary candidate F1 and observation top-1 accuracy.
5. Which classifier won on measured training-CV results, and whether Bayesian
   tuning improved the separate student-test metrics.
6. The Viterbi recurrence, partial-arc directed shortest paths, invalid
   transitions, gap handling and decoder calibration split.
7. Why the live API recomputes features instead of replaying saved scores, and
   where server-only references are first read.
8. What the actual GPU log proves and which models still use CPU.
9. Which performance numbers are warm-cache timings, and which results are
   affected by tuning selection optimism.
10. How to configure the actual reserved archive without using it to tune the
    model, and how to show the receiving server logs on two laptops.

No similarity-check result, reserved-trip score, physical two-laptop test or
unaided student authorship is asserted. GitHub publication is deferred per the
user's instruction.
