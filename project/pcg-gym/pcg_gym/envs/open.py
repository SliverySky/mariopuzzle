import os
rootpath = os.path.abspath(os.path.dirname(__file__))
import subprocess
from atexit import register
def clear(process):
    pass#process.terminate()
agent = subprocess.Popen(["java", "-jar",rootpath + "/Mario-AI-Framework-master.jar", rootpath + "/" + str(10000)]) #, rootpath + "/" + str(10000)
register(clear,agent)
float("")