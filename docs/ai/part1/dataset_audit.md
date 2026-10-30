# Phase C: Dataset Audit and Licensing Review

## 1. Inventory of Available Datasets

All dataset collections were audited under `data/raw/` in the local checkout (commit `f6eaed5`).

| Dataset Collection | Subdirectories | Primary Test File | File Size | Line Count | Format / Delimiter | Coordinate Units |
|---|---|---|---|---|---|---|
| **ETH** | `test/`, `train/`, `val/` | `test/biwi_eth.txt` | 122.5 KB | 5,492 | TSV (`\t`) | Meters (rectified world) |
| **HOTEL** | `test/`, `train/`, `val/` | `test/biwi_hotel.txt` | 134.8 KB | 6,543 | TSV (`\t`) | Meters (rectified world) |
| **UNIV** | `test/`, `train/`, `val/` | `test/students001.txt`<br>`test/students003.txt` | 756.8 KB<br>645.5 KB | 21,813<br>17,953 | TSV (`\t`) | Meters (homography) |
| **ZARA1** | `test/`, `train/`, `val/` | `test/crowds_zara01.txt` | 175.4 KB | 5,153 | TSV (`\t`) | Meters (homography) |
| **ZARA2** | `test/`, `train/`, `val/` | `test/crowds_zara02.txt` | 313.6 KB | 9,722 | TSV (`\t`) | Meters (homography) |
| **Stanford Drone (SDD)** | `stanford/` | `test_trajnet.pkl`<br>`train_trajnet.pkl` | 2.38 MB<br>7.14 MB | N/A (binary pickle) | Pickle (`.pkl`) | Pixels / Scaled Meters |

---

## 2. Structural & Formatting Audit

### 2.1 Tabular Schema (`.txt` files)
Each text file across the ETH/UCY collections adheres to the standard 4-column TrajNet / Social-GAN annotation structure:
1. `frame_id`: Integer or float frame number (typically increments by 10 frames = 0.4 seconds at 25 fps).
2. `track_id`: Integer or float pedestrian identifier.
3. `pos_x`: Float representing Cartesian X coordinate in meters.
4. `pos_y`: Float representing Cartesian Y coordinate in meters.

### 2.2 Integrity & Quality Findings
- **File Readability**: All 38 text files in `data/raw/` are readable, non-empty, and free of syntax corruption.
- **Sampling Interval**: Consecutive frames for a given track increment by $\Delta \text{frame} = 10$ ($dt = 0.4$ s, 2.5 Hz).
- **Scale Discrepancies in Research Literature**:
  - In `research/MID/process_data.py:133`, `biwi_eth` test positions are multiplied by $0.6$ ($pos \times 0.6$) and subsequently divided by $0.6$ during evaluation (`mid.py:95`). This historical scaling quirk originated in early Social-GAN benchmarks. Any future comparison on ETH must explicitly account for this scaling convention to avoid erroneous ADE/FDE metrics.
- **Preprocessed Pickles**: `data/raw/` contains only raw annotations; no preprocessed `.pkl` scene graphs exist for ETH/UCY in `data/` or `research/MID/`.

---

## 3. Provenance and Licensing Review

### 3.1 Code vs. Data License Separation
- **MID Source Code**: Released under the permissive **MIT License** (see [`research/MID/README.md`](file:///c:/Users/harsh/projects/genevac/research/MID/README.md#L78-L80)).
- **Dataset Collections**: Do **not** inherit the MIT License of the code repository. Each dataset originates from an independent academic institution.

### 3.2 Dataset Provenance & Terms of Use

#### 1. ETH Walking Pedestrians Dataset (BIWI)
- **Publisher**: Computer Vision Laboratory (CVL), ETH Zurich (Pellegrini et al., ICCV 2009).
- **Original Source**: `https://data.vision.ee.ethz.ch/cvl/aass/`
- **License / Terms**: Granted strictly for academic, non-commercial research. Commercial redistribution or use in proprietary products is prohibited without institutional permission.
- **Status in GenEvac-Sentinel**: **Unresolved for commercial deployment; Permitted for non-commercial academic research and benchmarking.**

#### 2. UCY Crowds-by-Example Dataset (Zara01, Zara02, University)
- **Publisher**: Department of Computer Science, University of Cyprus (Lerner et al., Eurographics 2007).
- **Original Source**: `https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data`
- **License / Terms**: Restricted to non-commercial academic research. Redistribution without citation or attribution is unauthorized.
- **Status in GenEvac-Sentinel**: **Unresolved for commercial deployment; Permitted for internal academic validation.**

#### 3. Stanford Drone Dataset (SDD)
- **Publisher**: Computational Vision and Geometry Lab, Stanford University (Robicquet et al., ECCV 2016).
- **License / Terms**: Research and educational use only.
- **Status in GenEvac-Sentinel**: **Academic research use only.**

---

## 4. Transferability to Evacuation Simulations

| Dimension | Real-World Benchmark (ETH/UCY) | GenEvac-Sentinel Evacuation Simulator |
|---|---|---|
| **Context** | Open plazas, campus paths, sidewalks. | Confined architectural interiors, doorways, bottlenecks. |
| **Agent Goal** | Diverse origins and destinations, wandering, grouping. | Unidirectional flow toward emergency exits. |
| **Speed Distribution** | Nominal walking speeds ($1.0 - 1.4\text{ m/s}$). | Elevated speeds, stress-induced acceleration ($1.3 - 2.5\text{ m/s}$). |
| **Density & Interaction** | Low to moderate density ($< 1.0\text{ ped/m}^2$). | Extreme density ($2.0 - 5.0\text{ ped/m}^2$), high physical repulsion. |
| **Sampling Rate** | 2.5 Hz ($dt = 0.4\text{ s}$). | 10 Hz ($dt = 0.1\text{ s}$). |

> [!WARNING]
> Models trained exclusively on ETH/UCY cannot be assumed to generalize directly to panic or evacuation regimes without domain adaptation, synthetic fine-tuning on simulation logs, or hybrid physical constraints.

