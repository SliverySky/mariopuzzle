import os
import argparse
parser = argparse.ArgumentParser(description='play')
parser.add_argument(
    '--path',
    help='the location of the level')
parser.add_argument(
    '--mode',
    default = 'human',
    help='human:human play; agent:A* agent to play')
args = parser.parse_args()
if args.mode == "human": #+os.path.abspath(os.path.dirname(__file__)) 
    os.system("java -jar "+ "./Mario-AI-Framework-master.jar "+args.path+" 1")
elif args.mode == "agent":
    os.system("java -jar "+ "./Mario-AI-Framework-master.jar "+args.path+" 0")