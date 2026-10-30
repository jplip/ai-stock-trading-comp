# Dashboard source

To use the dashboard, open **Signal-Lab.html** in the repository root. That self-contained file needs no installation.

This folder preserves the editable source formerly packed inside the ZIP:

- `dist/index.html`, `style.css`, `app.js`, `engine.js`: the browser interface and its model engine.
- `tests/`: browser-engine checks that run with Node.js.
- `python/`: an optional educational NumPy companion, separate from the competition submission.

The dashboard starts on synthetic demo data. Its models, long/cash allocation rules, time split, and approximate simulator differ from the actual competition notebook. Its exported CSV/JSON files are diagnostics, not official submissions.

Run the browser checks from the repository root:

```sh
node --test dashboard/tests/engine.test.cjs dashboard/tests/integration.test.cjs
```

The official notebook, saved predictor, and organizer's scoring harness are in `../competition-submission/`.
