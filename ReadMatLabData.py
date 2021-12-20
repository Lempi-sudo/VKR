import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


if __name__ == '__main__':
    dataframe= pd.read_parquet("im")
    df = pd.read_table("im",header=None, sep=' ' )
    df2=df.isna()
    df3=df.dropna(axis=1)
    image_np = df3.to_numpy()

    img = Image.fromarray(image_np.astype(np.uint8))



    img.show()

    print(df)