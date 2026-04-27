## Dashboard screenshots

Capture these from the running Streamlit dashboard (`streamlit run app.py`)
and drop the PNGs in this directory:

- `progression.png` — Progression tab. Shows best val_bpb per run over time
  (1.6656 → 1.310 trajectory across the four submission milestones).
- `sweep_heatmap.png` — Sweep heatmaps tab. Both heatmaps in one shot:
  the stride × n-gram grid (best cell stride=64, ngram=False, val_bpb=1.36165)
  and the α × max_order grid (best cell α=0.15, order=5, val_bpb=1.36345).
- `run_audit.png` — Run audit tab. Top of the run inventory table — rank 1
  is `overnight_int8_best` at 1.34168 BPB; the v3 seed-314 sweep cluster
  sits at ranks 3–7.
- `leaderboard_gap.png` — Leaderboard tab. The current OpenAI leaderboard
  alongside this project's best submission (1.310), making the
  Spark-vs-8×H100 gap explicit.

How to capture:

1. Start the dashboard: `streamlit run app.py` (port 8501 by default).
2. Open each tab in the browser and take a full-tab screenshot
   (Cmd/Ctrl + Shift + S in most browsers, or use the OS screenshot tool).
3. Save with the filenames above into this directory.
4. Commit with: `git add screenshots/*.png && git commit -m "Add dashboard screenshots"`

Streamlit also has a built-in "Print" → "Save as PDF" if you want a single
PDF of all tabs instead of per-tab PNGs.
