"""
    0.1 Caliculate Modularity with Gephi.
    0.2 Save and upload result.csv to server.
    1. Concat result.csv with metadata.csv
    0.3 View Tensorboard.
"""
import os
import pprint
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig

@hydra.main(config_name="config_gephi")
def main(cfg: DictConfig):
    gephi_df = pd.read_csv(f"{cfg.path.load_dir}/herm_full_edgelist.csv")
    metadata_df = pd.read_csv(f"{cfg.path.load_dir}/metadata.tsv", sep="\t")

    gephi_df = gephi_df.loc[:, ["Id", "modularity_class"]]
    gephi_df.columns = ["Class", "modularity_class"]

    metadata_df = metadata_df.loc[:, ["Time", "Date", "Dist", "Class"]]
    metadata_df = metadata_df.reset_index()
    print(metadata_df.columns)

    df = pd.merge(
        metadata_df.loc[:, ],
        gephi_df,
        how="outer"
        )

    df = df.sort_values("index")
    df = df.fillna(9.0)
    print(set(df.loc[:, "modularity_class"].values))
    df.to_csv(f"{cfg.path.load_dir}/{cfg.path.save_name}", header=True, index=False, sep="\t")

if __name__ == '__main__':
    main()