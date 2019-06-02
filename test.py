import time, sys
import pandas as pd

def read_csvfile(filepath):
	update_progress("Loading Dataframe Started",0.5)
	df=pd.read_csv(filepath)
	update_progress("Loading Dataframe", 1)


def update_progress(job_title, progress):
    length = 20 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()

# Test
# for i in range(100):
#     time.sleep(0.1)
#     update_progress("Some job", i/100.0)
# update_progress("Some job", 1)
read_csvfile("./cv_server_data.csv")




# import numpy as np
# import sesd
# import matplotlib as ml
# import matplotlib.pyplot as plt

# ts = np.random.random(100)
# index=[]
# for i in range(100):
# 	index.append(i)
# # Introduce artificial anomalies
# ts[14] = 9
# ts[83] = 10


# outliers_indices = sesd.seasonal_esd(ts, hybrid=True, max_anomalies=2)
# for idx in outliers_indices:
#     print ("Anomaly index: {0}, anomaly value: {1}".format(idx, ts[idx]))
# # y_values=[]
# # for item in outliers_indices:
# # 	y_values.append(ts[item])
# # plt.scatter(index, ts)

# # plt.show()
# plt.plot(index, ts, label='some graph')
# plt.plot(index,ts,markevery=outliers_indices, ls="", marker="o", label="points")
# plt.show()
