import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.espn_client import ESPNClient
from data_pipeline.raw_data_aggregator import RawDataAggregator
from data_pipeline.dataset_builder import DatasetBuilder

def main():
    trend_builder = DatasetBuilder(timing=True)
    trend_builder.build_dataset(save_step=100)
    # trend_builder.build_dataset(reload=True, save_step=100)
    # print(DataAggregator.game_log)
    # trend_builder.build_player_trends(game_id=401873343, poi_id=4222252)[1]
    trend_builder.load_dataset()
    dataset = trend_builder.dataset
    # full= 0
    # for player in dataset:
    #     total = 0
    #     zeroes = 0
    #     for sample in player:
    #         if not torch.any(sample[1]):
    #             zeroes += 1
    #         if not torch.any(sample[2]):
    #             zeroes += 1
    #         total += 2
    #     if zeroes == 0:
    #         full += 1
    # print(f'{full}/{len(dataset)} players have full datasets')

    with open('data/datasets/diagnostic.txt', 'w') as f:
        for i, player in enumerate(dataset):
            no_teammate = sum(1 for sample in player if torch.any(sample[0]) and not torch.any(sample[1]))
            no_opponent = sum(1 for sample in player if torch.any(sample[0]) and not torch.any(sample[2]))
            if no_teammate or no_opponent:
                f.write(f'player {i}: {no_teammate} games missing teammate data, {no_opponent} missing opponent data\n')
    

if __name__ == "__main__":
    main()