# AI Prediction Engine

## Objective
The prediction module forecasts future 2D trajectories $(\hat{x}_{t+1}, \hat{y}_{t+1}), \dots, (\hat{x}_{t+H}, \hat{y}_{t+H})$ of pedestrians based on their historical positions and environmental context.

## Models
1. **Linear Extrapolation Baseline**: Constant velocity / directional heading extrapolation.
2. **Social-Aware Heuristics**: Extrapolation incorporating exit attraction and obstacle repulsion.
3. **Learned Predictors (ML / Neural Networks)**:
   - Sequence-to-Sequence (LSTM / GRU / Transformer)
   - Graph Neural Networks (GNN) modeling pedestrian-pedestrian interactions.

## Inputs & Outputs
- **Input**:
  - Observation tensor: `(N, T_obs, 2)` - Past positions of `N` agents over `T_obs` steps.
  - Environmental map grid: Obstacle occupancy grid `(H, W)`.
- **Output**:
  - Prediction tensor: `(N, T_pred, 2)` - Predicted future positions over horizon `T_pred`.
