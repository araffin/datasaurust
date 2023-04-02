import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], header=None)

print(df.describe())

# Note: we use biased estimator in the rust code
# print(df[1].std(ddof=0))

# import ipdb

# ipdb.set_trace()
