# engage: the code

Setup: `bash setup.sh`, then `source .venv/bin/activate` in every new terminal.
Windows (Python 3.12): `powershell -ExecutionPolicy Bypass -File setup.ps1`, then `..\.venv\Scripts\Activate.ps1`.

| Command | What it does |
|---|---|
| `python check_orbbec.py` | Test the Orbbec camera (colour + depth, centre distance) |
| `python run_live.py --source orbbec --take NAME --scenario stationary --location lab --commander P03 --people P01,P03` | Record a take into `takes/NAME/` |
| `python run_live.py --source take:takes/NAME --take NAME_replay --headless` | Re-run a recorded take offline |
| `python label_episodes.py takes/NAME [--annotator B]` | Label each gesture: r / n / x |
| `python check_dataset.py` | Check all takes, write `dataset_report.md` |
| `python evaluate.py takes/*` | Results table + figures into `results/` |

Full recording procedure: `../HANDOVER_DATA_COLLECTION.md`. Data format: `engage/takeio.py` and `../CLAUDE.md`.

Code: `engage/perception.py` (people, hands, gestures, distance/facing + σ) · `engage/fuzzy.py`
(Type-1 / IT2 engagement + decision) · `engage/methods.py` (policies) · `engage/orbbec.py` (camera) ·
`engage/takeio.py` (raw data I/O) · `engage/go2_link.py` (WebRTC control, untested).
