# Current detector application

This directory contains the selected refined YOLOv10n 1024 browser application. Start with the [repository README](../README.md). The [technical paper](../TECHNICAL_PAPER.md) documents the model and its measured development performance.

## Run from this directory

```bash
node scripts/check-deployment.cjs
node --test tests/*.test.cjs
python3 -m http.server 8765 --bind 127.0.0.1 --directory dist
```

Open http://localhost:8765. Use the initial confidence of 0.174 to reproduce the evaluated operating cutoff. The browser shows image-level candidate totals. Offline per-cell evaluation used supplied cell polygons.

See [DEPLOYMENT.md](DEPLOYMENT.md) for Vercel and Netlify settings. The [metrics directory](metrics/) contains the model comparison and per-cell records. Training and analysis workflows require the original research data layout described in the technical paper.

The active application contains only `dist/refined1024.onnx`. The previous repository tree is preserved under [`../archive/legacy-before-1024-release/`](../archive/legacy-before-1024-release/). Original project materials use [GPL-3.0-only](../LICENSE), subject to [third-party notices](../THIRD_PARTY_NOTICES.md).
