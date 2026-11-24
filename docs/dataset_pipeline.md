# Real-world pedestrian trajectory dataset pipeline

## Scope

This pipeline prepares only real-world pedestrian trajectory data from:

- ETH
- HOTEL
- UNIV
- ZARA1
- ZARA2
- Stanford Drone Dataset (SDD)

The five ETH/UCY subsets are handled as individual sources. No synthetic trajectory generator, GenEvac simulation recorder, or synthetic-data export path is included in this pipeline.

ETH/UCY/SDD describe ordinary pedestrian motion. They are not emergency-evacuation ground truth. Use them for pedestrian trajectory training, prediction benchmarking, and external validation; do not describe their metrics as evacuation-performance validation.

## 1. Dataset acquisition and folder organization

Acquire each dataset from its official research/project distribution and review the applicable terms before use or redistribution. This repository pipeline does not download datasets automatically and does not bundle the raw data.

Suggested local organization:

```text
data/
  raw/
    eth/
      train/  test/
    hotel/
      train/  test/
    univ/
      train/  test/
    zara1/
      train/  test/
    zara2/
      train/  test/
    sdd/
      <scene>/
  processed/
    real_world/
```

Keep source scene and official partition boundaries. Do not combine scenes and randomly split the combined data if the goal is a benchmark comparison.

Inventory files:

```bash
python scripts/build_dataset.py --inventory-only --raw-root data/raw
```

Inventory is a file listing only. It does not mean the dataset was parsed or validated.

## 2. ETH/UCY ingestion

Expected row format:

```text
frame_id pedestrian_id x y
```

At 25 source frames per second, one source frame is 0.04 seconds. If annotations are 10 frames apart, adjacent observations are 0.4 seconds apart. The loader calculates timestamps from frame-index differences; it does not assume every row is one 0.4-second step.

Example:

```bash
python scripts/build_dataset.py \
  --dataset ETH \
  --file data/raw/eth/test/biwi_eth.txt \
  --scene eth_test \
  --source-split test \
  --output data/processed/real_world
```

Repeat for HOTEL, UNIV, ZARA1 and ZARA2 using their respective files and scene identifiers. Use `--source-split train`, `validation` or `test` to preserve the source partition. If the publisher's release does not define that partition, omit the option and the pipeline creates deterministic agent-disjoint train/validation/test splits within that scene.

## 3. SDD ingestion and coordinate handling

SDD annotations are image/scene-specific. Raw pixel coordinates must not be passed directly to this project as metres. Before ingestion:

1. Read the SDD scene/video metadata.
2. Use the matching scene homography to transform pedestrian image coordinates into world coordinates.
3. Convert the result into a normalized CSV or TSV with columns `track_id,frame,x,y` or `track_id,timestamp,x,y`.
4. Confirm that x/y are world coordinates in metres.

The loader deliberately rejects SDD unless the caller explicitly confirms the metre conversion. It does not guess a homography, frame rate, or raw SDD annotation schema.

Example for a normalized frame-index table:

```bash
python scripts/build_dataset.py \
  --dataset SDD \
  --file data/raw/sdd/scene1_tracks_world.csv \
  --scene sdd_scene1 \
  --source-split test \
  --sdd-seconds-per-frame 0.0333333333 \
  --sdd-coordinates-in-metres \
  --output data/processed/real_world
```

Use the actual frame rate for the corresponding SDD video; the example value is illustrative, not a universal SDD setting. If the normalized table already contains timestamps in seconds, omit `--sdd-seconds-per-frame`.

## 4. Validation and window preparation

The pipeline:
- checks finite coordinates and timestamps;
- rejects duplicate/non-increasing timestamps within a track;
- creates observed trajectory records;
- generates configurable history/future windows (default 8 history observations and 12 future observations);
- keeps agents disjoint across automatically generated splits within each scene;
- writes separate train, validation and test JSONL files;
- writes a manifest with source, coordinate/time units, split assignments and window counts.

The current implementation preserves source observation timing. It does not resample irregular trajectories onto a common temporal grid. Before training, verify the sampling interval expected by the MID model and resample consistently if required.

## 5. Outputs

For each dataset/scene, output files are:

- `<dataset>_<scene>_train.jsonl`
- `<dataset>_<scene>_validation.jsonl`
- `<dataset>_<scene>_test.jsonl`
- `<dataset>_<scene>_manifest.json`

The manifest marks the source as `real_world_observed`. The pipeline does not claim that files exist until they have been acquired locally and successfully processed.

## 6. Important limitations / next verification

- Dataset acquisition and licence review are manual.
- The SDD path requires a verified homography conversion step outside this loader.
- Automatic splitting is agent-disjoint only within one scene; it is not a substitute for official benchmark partitions.
- Different datasets/scenes can have different coordinate frames and sampling intervals. Do not pool raw coordinates or compare ADE/FDE until the coordinate/time conventions are aligned.
- This pipeline prepares data windows; it does not train MID or establish model performance.
- No synthetic trajectory data is generated or exported here.
