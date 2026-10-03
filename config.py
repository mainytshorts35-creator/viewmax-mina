import os
import logging
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] ViewMaxPro: %(message)s"
)
logger = logging.getLogger("ViewMaxPro")

@dataclass
class AppConfig:
    workspace_dir: str = os.path.join(os.getcwd(), "viewmax_workspace")
    sample_rate: int = 44100
    fps: int = 24
    font_path: str = "Impact.ttf"
    
    def __post_init__(self):
        os.makedirs(self.workspace_dir, exist_ok=True)
