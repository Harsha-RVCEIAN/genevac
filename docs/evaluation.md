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
