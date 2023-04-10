import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], header=None)

print(df.describe())
