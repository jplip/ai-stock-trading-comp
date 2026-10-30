# AI Stock Trading Competition

Choose what you want to do:

| Goal | Start here |
| --- | --- |
| Try the visual dashboard | [Dashboard](https://signal-lab-camski-1030.stripymule.chatgpt.site/) or download this repo and double-click **Signal-Lab.html** |
| Run the competition notebook | [Open in Google Colab](https://colab.research.google.com/github/jplip/ai-stock-trading-comp/blob/main/competition-submission/Signal_Lab_Competition.ipynb) |
| Understand the current strategy | [Short prediction function](competition-submission/predict.py) |
| Inspect the experiments and scores | [Recorded results](competition-submission/results/) |

## Run the dashboard

Use the dashboard link above, or select **Code → Download ZIP** on GitHub, unzip it, and double-click **Signal-Lab.html**. The local HTML file is self-contained and needs no Python installation.

The dashboard starts with synthetic demo prices. Upload a single-stock CSV and choose **Run experiment** to try your own data. Its models and long/cash simulator are a separate learning playground. Its CSV/JSON exports are not the official competition submission. Changing the dashboard does not change the strategy in the notebook.

## Prepare and submit the competition files

1. Open the competition notebook in Colab using the link above.
2. Check the team cell: **Signal Lab**, **Cameron Penaranda**, **Justin Paul Liporada**.
3. Select **Runtime → Run all**. The notebook loads the official data, compares strategies, and runs the organizer's checks.
4. The final `save_submission(predict)` cell downloads the `.pkl` file.
5. Select **File → Download → Download .ipynb** to save the executed notebook.
6. Upload **both files** to the folder the organizers provide. Uploading to GitHub does not submit your entry.

The saved `.pkl` in this repo has the current team metadata. After changing the code or team details, regenerate it in the notebook before submitting. The notebook includes a two-minute explanation outline and questions for both teammates to rehearse.

## What the current model does

The selected strategy is the organizer's **20-day / 100-day trend rule**:

- Return `1.0` when the 20-day average closing price is above the 100-day average.
- Otherwise return `0.0` and target cash.
- Return cash when fewer than 100 days are available.

This rule has no learned coefficients. Ridge and Extra Trees were also tried but lost the recorded training-period comparison. On the four-stock 2023 practice run, the selected rule's Sharpe was **0.14**, compared with **1.19** for buying and holding. The results do not establish an advantage over that benchmark.

The fitted experiments use 2015–2022 data. **2023 and 2024 were reused development checks**, not untouched final tests. No real prices after 2024 were used. The organizer's secret rounds remain unseen.

## Repository map

```text
Signal-Lab.html                         Browser dashboard
AI_Trading_Competition_Simple.ipynb     Original organizer starter
competition-submission/
  Signal_Lab_Competition.ipynb         Our notebook and presentation guide
  Signal_Lab_submission.pkl            Saved strategy and team metadata
  predict.py                          Readable equivalent of the selected rule
  strategy.py / workflow.py           Full experiments and evaluation
  sdsu_harness.py                     Unchanged organizer scoring engine
  test_strategy.py                    Contract and causality checks
  data/                              Official CSV and its provenance
  results/                           Recorded scores, trades, and plots
docs/                                Dashboard preview
dashboard/                           Editable dashboard source and checks
```

For local Python setup and file details, see [competition-submission/README.md](competition-submission/README.md).
