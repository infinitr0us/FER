import argparse

from FER import demo
from Training import train

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("func",default="demo", type=str, help="<Demo> or <model>")
  args = parser.parse_args()
  func = args.func

  if func == "demo":
    demo()
  if func == "train":
    train()

if __name__ == '__main__':
  main()
