# AI use log

Running notes for the AI-use reflection in the report. Add to this as you go:
what the AI helped with, what you checked, and anything it got wrong.

Tool: Claude (Anthropic), via claude.ai. This was the only AI tool used; I
did not use Gemini in Colab or GitHub Copilot.

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
  *Done: I re-ran everything in Colab with the official UCI download on
  Monday, and the report uses those numbers.*
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
- Created the `add-code`, `tests-and-docs`, `official-rerun` and
  `colab-notebook` branches, committed each set of changes and merged them
  into main with pull requests (with Claude's guidance on the GitHub Desktop
  steps and commit messages).
- Wrote the first version of the Colab re-run notebook myself, so I
  understand it and can demo it. Claude had offered to build it; I chose to
  write it.
- Ran the notebook in Colab to produce the official results, test report and
  documentation.

**What Claude did**
- Wrote the pytest tests using a toy dataset with a known answer (34 tests at
  first, 37 in the final version), the pytest HTML report, pdoc HTML
  documentation and the README.
- Checked the tests can actually fail by breaking the code on purpose. Each
  bug was caught:
  - grouped split replaced by a random split in model selection (2 tests);
  - square root removed from RMSE;
  - StandardScaler removed (2 tests; the known-quadratic test failed from
    numerical error alone, showing the conditioning problem).
- Moved the Week 5.5 metric formulas into a docstring at my request, so they
  appear in the pdoc documentation.
- Made the column renaming match UCI's column names, not just column order,
  before the first run on the real download.

**Google Colab notebook**
- Claude cleaned up my notebook into `notebooks/airfoil_noise_colab.ipynb`:
  - The repo had been cloned inside itself
    (`/content/Air-Foil-Noise/Air-Foil-Noise`) because the clone cell ran
    again after `%cd`. The setup cell now starts from `/content` and only
    clones if the folder doesn't exist.
  - Packages were installed before `git pull`, so they came from an old
    `requirements.txt`. The cells are now in the right order.
  - Added pytest and pdoc cells. My first version had none, so the outputs I
    downloaded still contained Claude's old test report and docs.
  - Added headings and explanations for each step, `-q` to quiet the git
    output, and a proper file name instead of `Untitled0`.
- At my request, the install cell lists the packages individually
  (`ucimlrepo`, `pytest-html`, `pdoc`) with a comment for each, instead of
  installing from `requirements.txt`. Installing only what Colab lacks also
  avoids upgrading Colab's built-in packages.
- The results summary cell reads `final_test.csv`, `degree_sweep.csv` and
  `velocity_check.json` directly, so it always matches the current run.
- The last cell zips `results/`, `figures/`, `reports/` and `docs/` and
  downloads them, so I can copy the outputs into the repo and commit them.

**Errors and limitations noticed**
- Bug in Claude's own code: the results summary said 22 test runs instead of
  106, because the final-test-set count overwrote the audit's total under the
  same key. Claude spotted it from the summary output and fixed it.
- The test report and docs were first generated in Claude's workspace (the
  report even listed its computer's details). *Done: regenerated in my Colab
  notebook.*
- My notebook's summary cell opened `results/summary.md`, which `run_all.py`
  did not create at that point, and the cell's comment wrongly said it did.
  Replaced with a cell that reads the result files directly.
- Claude's first notebook didn't handle Colab restarting the session after
  `pip install`. The working folder reset to `/content`, so the figures cell
  failed with `FileNotFoundError`. It worked in Claude's workspace but not in
  Colab. Fixed by starting every cell with `%cd /content/Air-Foil-Noise`.
- Claude removed the GitHub token from the clone cell, assuming the repo was
  public without checking. The repo was still private, so the clone failed
  and every later cell failed with it. Fixed by making the repo public, which
  submission needs anyway.
- The official re-run held back a different 22 runs as the final test set
  (the UCI file has a different row order), so the numbers changed: degree 2
  went from R² 0.33 to 0.51. The conclusions stayed the same. I noted this as
  a limitation in the README: a small test set makes scores sensitive to which
  runs are picked.

## Tuesday 20 October

**What Claude did**
- Added an `input_ranges()` function to `data.py` for the inputs and outputs
  section of my report. `run_all.py` now saves `results/input_ranges.csv`
  (min, max and number of distinct values for each input and the target), and
  the Colab summary cell shows it first.
- Added a test checking the ranges against the data (38 tests in total).
- Pointed out that the distinct-value counts (4 velocities, 6 chords) explain
  why high-degree polynomials break down.

**Errors and limitations noticed**
- Claude's range table came from its reconstructed copy of the data. I used
  the table from my own Colab run with the official data.
- Claude found a `write_summary()` function in `run_all.py` that creates
  `summary.md`, contradicting its earlier statement that `run_all.py` never
  creates that file.

## Decisions that stay mine

- Project framing, choice of polynomial regression, the 5% selection rule,
  and keeping log10(frequency) as a proposed improvement rather than the
  selected model.
- Writing my own Colab notebook rather than having Claude build it.
- Checking that the code really follows my course notes.
