# GenEvac-Sentinel trajectory dataset pipeline

## Tracks
- Real-world: ETH, HOTEL, UNIV, ZARA1, ZARA2 and optionally Stanford Drone Dataset (SDD).
- Synthetic: GenEvac evacuation runs captured from the authoritative SimulationEngine.
Keep benchmark metrics and synthetic evacuation metrics separate. ETH/UCY and SDD are ordinary pedestrian-motion data, not labelled emergency-evacuation data.

## ETH/UCY timing
The repository dataset audit states 25 FPS source video and annotations every 10 frames. Thus one source frame is 0.04 s and successive annotations are 0.4 s apart. The loader default is `seconds_per_source_frame=0.04`. The old adapter default of 0.4 incorrectly treated a frame as 0.4 seconds.

## Inventory
```bash
python scripts/build_dataset.py --inventory-only --raw-root data/raw
```

## Prepare an ETH/UCY source file
```bash
python scripts/build_dataset.py --dataset ETH --file data/raw/eth/test/biwi_eth.txt --scene eth_test --output data/processed/datasets
```
The generic preparation command uses deterministic agent-disjoint splitting. For formal benchmark comparisons, preserve the dataset publisher's official train/validation/test files and prepare each partition independently; do not merge official splits and randomly re-split them.

## SDD
```bash
python scripts/build_dataset.py --dataset SDD --file data/raw/stanford/train_trajnet.pkl --trust-pickle
```
Pickle deserialization can execute arbitrary code. Only enable it for a verified, trusted local file. SDD pickle layouts vary; if the loader rejects the structure, export a verified table containing track_id, frame, x, y rather than guessing its schema.

## Synthetic evacuation capture
At simulation setup:
```python
from genevac.prediction.dataset_pipeline import SimulationDatasetRecorder
recorder = SimulationDatasetRecorder(observation_interval=0.1)
```
After every authoritative engine step call `recorder.capture(engine)`. At the end:
```python
recorder.export("data/processed/evacuation", scenario_id="hallway", seed=123, run_id="hallway_seed123")
```
This recorder observes engine state; it does not alter simulation movement. It must be explicitly wired into the run loop to collect data.

## Outputs and limitations
The exporter writes timestamped JSONL history/future windows and a manifest with split assignments and counts. Splits are agent-disjoint to reduce overlapping-window leakage. Coordinates retain source units; do not compare ADE/FDE across sources until coordinate scale and sampling interval are normalized. Dataset licences are independent of the code licence; verify current publisher terms before redistribution or deployment.
