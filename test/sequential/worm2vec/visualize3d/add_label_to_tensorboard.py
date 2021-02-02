import pandas as pd

path_to_metadata = "../outputs/2021-01-27/plt_onewindow_3k/plt_test_win_20/projector/after_test/metadata.tsv"
path_to_labels = "./tmp/metadata/metadata_win20_j2.0.csv"
save_path = "../outputs/2021-01-27/plt_onewindow_3k/plt_test_win_20/projector/add_8labels_j2/metadata.tsv"
def main():
    metadata = pd.read_csv(path_to_metadata, sep="\t")
    label = pd.read_csv(path_to_labels, index_col=0)

    metadata = metadata.reset_index(drop=True)
    label = label.reset_index(drop=True)

    df = pd.concat([metadata, label], axis=1)
    print(df.head())

    df.to_csv(save_path, header=True, index=False, sep="\t")

if __name__ == "__main__":
    main()
