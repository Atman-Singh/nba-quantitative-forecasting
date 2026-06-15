import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.espn_client import ESPNClient
from data_pipeline.raw_data_aggregator import RawDataAggregator
from data_pipeline.dataset_builder import DatasetBuilder

def main():
    trend_builder = DatasetBuilder(timing=True)
    trend_builder.build_dataset(save_step=5)
    # trend_builder.build_dataset(reload=True, save_step=1)
    # print(DataAggregator.game_log)
    # trend_builder.build_player_trends(game_id=401873343, poi_id=4222252)[1]
    trend_builder.load_dataset()
    dataset = trend_builder.dataset
    dataset = dataset.flatten(start_dim=0, end_dim=1)
    total = 0
    zeroes = 0
    for sample in dataset:
        if not torch.any(sample[1]):
            print(sample[1].shape)
            zeroes += 1
            print(1)
        if not torch.any(sample[2]):
            zeroes += 1
            print(2)
        total += 2
    print(zeroes / total)
            
    

if __name__ == "__main__":
    main()