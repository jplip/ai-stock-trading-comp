# Competition notebook and strategy

Start with [the notebook in Colab](https://colab.research.google.com/github/jplip/ai-stock-trading-comp/blob/main/competition-submission/Signal_Lab_Competition.ipynb). It is self-contained: it includes the strategy, training workflow, official data URL, and original export/check functions.

The current team is **Signal Lab**: **Cameron Penaranda** and **Justin Paul Liporada**. The checked-in pickle has those same names. Regenerate it after editing the team cell or strategy.

## Files

- `predict.py`: a short equivalent of the selected 20/100-day trend rule. The shared pickle uses this standalone function; it was checked against the notebook's predictor.
- `strategy.py`: all five candidate strategies, causal features, and fitting code.
- `workflow.py`: training-period comparison and practice evaluation using the official harness.
- `sdsu_harness.py`: the original scoring engine, unchanged.
- `data/`: the exact official CSV and its source/hash record.
- `results/`: preserved comparison tables, practice trades/equity, the unsuccessful first attempt, robustness diagnostics, and a plot.
- `MANIFEST.json`: hashes and verification notes for the shared artifacts.

Running the notebook creates a fresh `trading_competition/` output folder. Git ignores that generated folder; the checked-in historical evidence remains in `results/`.

## Local checks

Python 3.12 was used for verification. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r competition-submission/requirements.txt
python -m unittest discover -s competition-submission -p test_strategy.py -v
```

`requirements-exact.txt` records the full original environment. The submitted predictor uses NumPy and pandas and is packaged with cloudpickle; scikit-learn is needed for the tree challenger during training. Export in the organizer's notebook environment when dependency/Python versions differ.

## Evaluation limits

The selected model is a fixed rule with no fitted coefficients. The ML experiments were trained only through 2022. The 2023 and 2024 windows were reused during development, so they are not claimed as untouched final tests. The current practice result trails buy-and-hold; consult the saved tables rather than treating the model as a proven edge.

The browser dashboard is separate. Only the generated **`.pkl` plus executed `.ipynb`** go into the organizer's submission folder.
