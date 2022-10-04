import numpy as np
import weylpointprocesses as wpp
from datetime import datetime

#%%
start = datetime.now()

numstat = wpp.obtain_42_wp_configs(number_of_random_systems = 10, n = 3, filename = "testtest.txt")

print(datetime.now()-start)