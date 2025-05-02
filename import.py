import os
import struct
import numpy as np
import pandas as pd

def load_aida_to_csv(fname, n_channels, math_n_channels=0, csv_out="out.csv"):
    # 1) file size and channel count check
    filesize = os.path.getsize(fname)
    if filesize <= 1024 or ((filesize - 1024) % (4 * n_channels)) != 0:
        raise ValueError(f"Not a valid AIDA file for {n_channels} channels")
    n_samples = (filesize - 1024) // (4 * n_channels)

    # 2) read raw int16 data
    with open(fname, "rb") as f:
        f.seek(1024)  # skip header
        raw = np.fromfile(f, dtype="<i2", count=2 * n_channels * n_samples)

    # 3) reshape: each row = [w1_ch0, w2_ch0, w1_ch1, w2_ch1, ...]
    raw = raw.reshape(n_samples, 2 * n_channels)

    # 4) build DataFrame dict with channel X=word2, Y=word1
    data = {}
    for ch in range(n_channels):
        w1 = raw[:, 2 * ch    ]  # this was your 'xx'
        w2 = raw[:, 2 * ch + 1]  # this was your 'yy'
        data[f"ch{ch}_X"] = w2.astype(float)
        data[f"ch{ch}_Y"] = w1.astype(float)

    # 5) compute math channels
    base_X = data["ch0_X"]  # <- now a numpy array
    base_Y = data["ch0_Y"]
    for j in range(math_n_channels):
        mx = np.zeros(n_samples, dtype=float)
        my = np.zeros(n_samples, dtype=float)
        # diffs from sample 11 onward:
        diffs_X = (base_X[11:] - base_X[:-11]) / 5.0
        diffs_Y = (base_Y[11:] - base_Y[:-11]) / 5.0
        # place into rows 6 through n_samples-6:
        mx[6 : n_samples - 5] = diffs_X
        my[6 : n_samples - 5] = diffs_Y
        data[f"math{j}_X"] = mx
        data[f"math{j}_Y"] = my

    # 6) dump to CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_out, index=False)
    print(f"Wrote {n_samples} samples × {df.shape[1]} columns → {csv_out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert AIDA .bin → .csv")
    parser.add_argument("binfile", help="input .aida binary filename")
    parser.add_argument("-n", "--nchannels", type=int, required=True,
                        help="number of raw channels")
    parser.add_argument("-m", "--math",    type=int, default=0,
                        help="number of math channels to add")
    parser.add_argument("-o", "--output",  default="out.csv",
                        help="output CSV filename")
    args = parser.parse_args()

    load_aida_to_csv(args.binfile, args.nchannels, args.math, args.output)
