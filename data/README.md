# `data/` — runtime and bundled data

| Path | Tracked? | What it is |
|---|---|---|
| `recipes.json` | **yes** | The 200-recipe South-Indian dataset (`recipe_catalog.py` loads it). |
| `fastembed/` | no (git-ignored) | Downloaded local embedding-model cache (~240 MB). Recreated on first boot. |
| `hf_cache/` | no | Leftover HuggingFace download cache from an earlier one-off model fetch. Not referenced by any code. Safe to delete. |

Large model weights live in `../models/` (Vosk) and in `../infra/data/` (Docker volumes).
