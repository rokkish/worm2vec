import pandas as pd

path_to_metadata = "../outputs/2021-01-27/plt_onewindow_3k/plt_test_win_20/projector/after_test/metadata.tsv"
path_to_8vector_labels = "./tmp/metadata_8vector/metadata_win20_j2.0_test.csv"
path_to_clustering_labels = "./tmp/metadata_clustering/metadata_win20_test_kmeans_class10.csv"
save_path = "../outputs/2021-01-27/plt_onewindow_3k/plt_test_win_20/projector/add_8vector_kmeans_scalar_labels/metadata.tsv"

def main():
    metadata = pd.read_csv(path_to_metadata, sep="\t")
    label = pd.read_csv(path_to_8vector_labels, index_col=0)
    label2 = pd.read_csv(path_to_clustering_labels, index_col=0)

    metadata = metadata.reset_index(drop=True)
    label = label.reset_index(drop=True)
    label2 = label2.reset_index(drop=True)

    df = pd.concat([metadata, label, label2], axis=1)
    print(df.head())

    df.to_csv(save_path, header=True, index=False, sep="\t")

if __name__ == "__main__":
    main()
