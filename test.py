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
