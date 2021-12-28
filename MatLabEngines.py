import matlab.engine
import numpy as np
from ImageWork import LoadNamesImage, LoadImage
from skimage.io import imread
import matlab
import pandas as pd
import numpy as np

if __name__ == '__main__':

    df = pd.DataFrame.from_dict({'bit': [], 'feature': []})
    print(df)

    arr = np.array([1, 2, 3, 4])

    for i in range(10):
        new_row = {"bit": int(i), "feature": arr}
        df = df.append(new_row, ignore_index=True)

    print(df)
    df.to_csv('feature_vec/Image_with_water.txt')



