# AI use log

Running notes for the AI-use reflection in the report. Add to this as you go:
what the AI helped with, what you checked, and anything it got wrong.

Tool: Claude (Anthropic), via claude.ai.

## Sunday 18 October

**What Claude did**
- Explained the overall project, the prediction contract and the random vs
  grouped split idea in plain language.
- Talked me through creating the GitHub repository (I created it and cloned it
  myself with GitHub Desktop).
- Wrote the first version of the code: `data.py`, `models.py`,
  `evaluation.py`, `plots.py` and `run_all.py`, and ran it to produce the
  results and figures.
- After I shared my own unit summaries (Weeks 1-11), it changed the code to
  use the functions taught in the unit (`cross_validate`, `KFold`,
  `GroupKFold`, `DummyRegressor`) and added "Course link" comments linking
  each part of the code to the relevant week.

**Limitations and errors noticed**
- Claude's workspace could not reach the UCI server, so its preliminary
  results used a copy of the dataset from a GitHub mirror, which it
  reconstructed and checked against the published UCI statistics.
  *To do: re-run `python run_all.py` on my own machine with the official UCI
  download and use those numbers in the report.*
- Claude first wrote its own cross-validation loop instead of the
  `cross_validate` function from my Week 5.5 notes; changed once I shared my
  notes.
- Claude first planned to judge the velocity trend against the U^5 law
  (+17.6 dB). The data showed only ~+4 dB per band (the target is a *scaled*
  SPL), so that benchmark was dropped as a pass mark. The check now compares
  the model with the measured increase.
- Claude's first explanation for degrees 4-5 failing was "overfitting". A rank
  check showed the real cause: velocity has only 4 distinct values, so terms
  like v^4 are redundant and the matrix is singular (condition number ~1e16).

**Decisions that stay mine**
- Project framing, choice of polynomial regression, the 5% selection rule,
  and whether to include log10(frequency) as an improvement.
- Checking that the code really follows my course notes.
