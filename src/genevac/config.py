"""Application configuration and settings management."""

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class SimulationConfig:
    timestep: float = 0.1
    max_steps: int = 1000
    random_seed: int = 42
    default_pedestrian_speed: float = 1.34  # m/s
    pedestrian_radius: float = 0.3          # meters


@dataclass
class AppConfig:
    environment: str = os.getenv("GENEVAC_ENV", "development")
    log_level: str = os.getenv("GENEVAC_LOG_LEVEL", "INFO")
    data_dir: Path = Path(os.getenv("GENEVAC_DATA_DIR", "data"))
    output_dir: Path = Path(os.getenv("GENEVAC_OUTPUT_DIR", "output"))
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


config = AppConfig()
