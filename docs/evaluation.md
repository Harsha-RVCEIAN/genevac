# Evaluation Protocol and Metrics

## 1. Evacuation Performance Metrics
- **Total Evacuation Time (TET)**: Time until all agents have either exited or reached safety.
- **Time to 50% Evacuation (T50)**: Time required for half of the occupants to exit.
- **Mean / Max Density**: Average and peak density (pedestrians per square meter) across grid cells.
- **Exit Throughput (Flow Rate)**: Number of pedestrians passing an exit per second ($ped/s$).
- **Bottleneck Severity**: Duration and density of congestion zones at choke points.

## 2. AI Prediction Accuracy Metrics
- **Average Displacement Error (ADE)**:
  $$\text{ADE} = \frac{1}{N \cdot T} \sum_{i=1}^N \sum_{t=1}^T \| p_i^{(t)} - \hat{p}_i^{(t)} \|_2$$
- **Final Displacement Error (FDE)**:
  $$\text{FDE} = \frac{1}{N} \sum_{i=1}^N \| p_i^{(T)} - \hat{p}_i^{(T)} \|_2$$
- **Collision Rate**: Percentage of predicted trajectories that intersect obstacles or other pedestrians.

---

## 3. Part 4 Prediction Evaluation Subsystem

GenEvac-Sentinel provides a modular evaluation subsystem (`src/genevac/prediction/evaluation.py`):

### Architecture & Isolation Guarantee
```text
           Authoritative Observation Sequence
                     |
       +-------------+-------------+
       |                           |
[History Sequence]         [Future Ground Truth]
(t <= t_reference)          (t > t_reference)
       |                           |
  to_snapshot()                    |
       |                           |
       v                           |
[Predictor Model]                  |
       |                           |
[Predicted Trajectory]             |
       |                           |
       +------------+--------------+
                    | (Matched strictly by agent_id and timestamp)
                    v
          [PredictionEvaluator]
                    |
          ADE, FDE, Horizon-wise errors, Cohort metrics
```

### Components
1. **`EvaluationSample`**: Encapsulates a single evaluation window. Enforces strict temporal separation: historical observations cannot exceed `reference_timestamp`, guaranteeing zero future-data leakage into predictor snapshots.
2. **`SampleGenerator`**: Generates sliding-window evaluation samples across trajectories with configurable `history_length`, `prediction_horizon`, and `stride`.
3. **`PredictionEvaluator`**: Runs the predictor (`ConstantVelocityPredictor` by default) against each sample, matches coordinates with ground-truth strictly by timestamp, and computes ADE, FDE, and horizon-wise displacement errors.
4. **`DatasetAdapter`**: Loads real benchmark datasets (ETH/UCY format) and GenEvac JSON sample files into validated `TrajectoryObservation` sequences.
5. **`SimulationEvaluationAdapter`**: Integrates live simulation runs into the evaluation pipeline without mutating simulation state.

### Running the Evaluation Script
```bash
# Evaluate on ETH benchmark (8 history steps, 12 prediction horizon steps)
python scripts/evaluate_predictions.py --data data/raw/eth/test/biwi_eth.txt --format eth --history 8 --horizon 12

# Evaluate on GenEvac sample JSON
python scripts/evaluate_predictions.py --data data/sample/sample_trajectories.json --format json --history 2 --horizon 2
```
Outputs are written to `data/processed/evaluation_report.json` and `data/processed/evaluation_results.csv`.
