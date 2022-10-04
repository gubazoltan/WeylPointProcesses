import numpy as np
import wpp2
from datetime import datetime

#%%
start = datetime.now()

numstat = wpp2.obtain_42_wp_configs(number_of_random_systems = 2000, n = 3, filename = "testv2_6.txt")

print(datetime.now()-start)