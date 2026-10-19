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

## Monday 19 October

**What I did myself**
- Created the `add-code` branch, committed the code and merged it with a pull
  request (with Claude's guidance on the GitHub Desktop steps).
- Decided to build the Colab re-run notebook myself, for practice.

**What Claude did**
- Wrote 34 pytest tests using a toy dataset with a known answer, a pytest HTML
  report, pdoc HTML documentation and the README.
- Checked the tests can actually fail by breaking the code on purpose twice
  (grouped split replaced by a random split; square root removed from RMSE).
  Both bugs were caught.
- Moved the Week 5.5 metric formulas into a docstring at my request, so they
  appear in the HTML docs.
- Made the column renaming match UCI's column names, not just column order,
  before the first run on the real download.

**Errors and limitations noticed**
- Bug in Claude's own code: the results summary said 22 test runs instead of
  106, because the final-test-set count overwrote the audit's total under the
  same key. Claude spotted it from the summary output and fixed it.
- Claude initially offered to build the Colab notebook for me. I chose to write
  it myself so I understand and can demo it.

## Monday 19 October

**What Claude did**
- Explained creating a branch and merging it with a pull request in GitHub
  Desktop, and good commit messages. I did the Git steps myself.
- Moved the Week 5.5 metric formulas into a docstring so they appear in the
  pdoc documentation.
- Wrote 37 pytest tests, generated the HTML test report and pdoc docs, and
  rewrote the README.
- Checked that the tests can fail by deliberately breaking the code: making
  model selection use the random split was caught by 2 tests; removing the
  StandardScaler was caught by 2 tests (the known-quadratic test failed from
  numerical error alone, showing the conditioning problem).

**Limitations and errors noticed**
- The test report and docs were first generated in Claude's workspace, not
  mine. *To do: regenerate them in my Colab notebook.*

## Decisions that stay mine

- Project framing, choice of polynomial regression, the 5% selection rule,
  and whether to include log10(frequency) as an improvement.
- Checking that the code really follows my course notes.
