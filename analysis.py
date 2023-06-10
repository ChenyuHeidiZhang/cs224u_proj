import os
import json
import utils

for dataset in ['c4', 'yelp']:
    for option in ['frequency_only', 'smoothed']:
        print(f"Dataset: {dataset}, option: {option}")
        folder = f'{dataset}/cluster_outputs_{option}/n_clusters200_distance_threshold_None/'
        loss_file = 'deactivate_mean_cluster_id_to_average_MLM_loss.json'
        tokens_file = 'top_30_tokens.txt'
        with open(os.path.join(folder, loss_file), 'r') as f:
            cluster_id_to_losses = json.load(f)
        cluster_to_tokens = utils.read_top_activating_tokens(os.path.join(folder, tokens_file))

        # find the cluster that has the highest difference between cluster_turned_off loss and nothing_turned_off loss
        diffs = [(cluster_id, cluster_id_to_losses[cluster_id]['cluster_turned_off'] - cluster_id_to_losses[cluster_id]['nothing_turned_off']) for cluster_id in cluster_id_to_losses]
        diffs = sorted(diffs, key=lambda x: x[1], reverse=True)

        # print the top 30 and top 10 tokens for the 2 cluster with the highest difference
        for cluster_id, diff in diffs[:5]:
            print(f"Cluster {cluster_id} has difference: {diff}")
            print("Top 30 tokens:")
            print(cluster_to_tokens[str(cluster_id)][:30])
            print("Top 10 tokens:")
            print(cluster_to_tokens[str(cluster_id)][:10])
            print()
