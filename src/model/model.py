import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.params import MAX_ROSTER_SIZE, CONTEXT_WINDOW, FEATURES

class Model():   
    def __init__(self, hidden_size: int | None = 128):

        self.hidden_size = hidden_size

        self.lstm = torch.lstm(batch_size=MAX_ROSTER_SIZE,
                               input_size=FEATURES,
                               hidden_size=hidden_size,
                               batch_first=True)
        
        self.mlp = 