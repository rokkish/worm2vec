import numpy as np
import pandas as pd


def main():
    df = pd.read_csv("./odor_data/odor_data.csv").astype("float32")
    df["time"] = df["time"]*10

    df_10fps = pd.DataFrame(list(np.arange(10, df.shape[0]*10)), columns=["time"])

    df_10fps = pd.merge(df_10fps, df, how="outer")
    df_10fps.sort_values("time")

    df_10fps = df_10fps.interpolate(method="linear")
    print(df_10fps.head())

    df_10fps.to_csv("./odor_data/odor_data_10fps.csv", index=False)

if __name__ == '__main__':
    main()