# APEX — Fourier Racing Line Lab

A retro-80s-F1 telemetry web viewer for physics-constrained minimum-lap-time
racing lines, computed for every circuit in the dataset via progressive
Fourier optimisation.

```
optimise_all_tracks.py     ← run locally: solves every *.csv, writes site/data/*.json
site/
  index.html               ← the viewer (static, no build step)
  style.css
  app.js
  data/
    index.json             ← track list + F1 records
    <Track>.json            ← bounds + 11 progressive-resolution solutions
```

## 1 — Generate the data (local, one time)

Put `optimise_all_tracks.py` in your `track_optim/` folder next to all the
`*.csv` files, then:

```bash
uv run optimise_all_tracks.py
```

This loops every circuit. For each it runs the **two nested loops**:

* OUTER — resolution schedule `N ∈ [1,2,4,8,16,32,48,64,96,128,192]`,
  each level warm-starting from the previous (zero-padded coefficients).
* INNER — progressive penalty `μ ∈ [1e0 … 1e8]`, each warm-starting the next.
* SOLVER — `jaxopt.ScipyMinimize` (L-BFGS) + JAX autodiff + JIT, innermost.

It writes `site/data/<Track>.json` (one per circuit) plus `site/data/index.json`.

> Tip: `FAST_TEST=1 uv run optimise_all_tracks.py` does one track with a
> reduced schedule — use this to sanity-check before the full multi-hour run.

> The full run is long (26 tracks × 11 resolutions × 8 penalties). Run it
> overnight, or trim `RES_SCHEDULE` / `PENALTY_SCHEDULE` at the top of the file.

## 2 — Preview locally

```bash
cd site
python3 -m http.server 8000
# open http://localhost:8000
```

## 3 — Deploy to GitHub Pages

GitHub Pages serves static files straight from a repo — no build needed.

### Option A — `/docs` folder on `main` (simplest)

```bash
# from your repo root
mv site docs                       # Pages can serve from /docs
git add docs
git commit -m "Add APEX racing-line viewer"
git push origin main
```

Then on GitHub:

1. Repo → **Settings** → **Pages**
2. **Source**: *Deploy from a branch*
3. **Branch**: `main`  •  **Folder**: `/docs`
4. **Save**

Wait ~1 minute. Your site is live at:

```
https://<your-username>.github.io/<repo-name>/
```

### Option B — dedicated `gh-pages` branch

```bash
git subtree push --prefix site origin gh-pages
```

Then Settings → Pages → Source: branch `gh-pages`, folder `/ (root)`.

### Option C — GitHub Actions (auto-redeploy on every push)

Create `.github/workflows/pages.yml`:

```yaml
name: Deploy Pages
on:
  push: { branches: [main] }
permissions:
  contents: read
  pages: write
  id-token: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with: { path: ./site }
      - id: deployment
        uses: actions/deploy-pages@v4
```

Then Settings → Pages → Source: **GitHub Actions**. Every `git push` to
`main` now rebuilds the live site automatically.

## 4 — Updating data

Re-run `optimise_all_tracks.py`, then commit the regenerated
`site/data/*.json` (or `docs/data/*.json`) and push. Pages redeploys.

## Notes

* The F1 lap-record table is in `optimise_all_tracks.py` (`F1_RECORDS`).
  Edit values there — they only feed the "vs F1 record" comparison line.
  Non-F1 layouts (BrandsHatch, Norisring, etc.) show "N/A".
* Each line stores 600 decimated points (smooth for animation, small JSON).
* Playback advances by **real arc-length ÷ speed**, so the car moves at
  near-true relative pace — slow through corners, fast on straights.
* Keyboard: `Space` play/pause, `←`/`→` frame step.
