#!/usr/bin/env python

import argparse
import random

def main():
    parser = argparse.ArgumentParser(description="Simple 'cross-platform' utility to shuffle and trim file.")
    parser.add_argument("file")
    parser.add_argument("limit", type=int)
    args = parser.parse_args()

    # read file
    data = []
    with open(args.file, 'r') as f:
        data = f.readlines()

    # shuffle file
    #random.shuffle(data)

    # trim file
    data = data[:args.limit]

    # overwrite file
    with open(args.file, 'w') as f:
        for line in data:
            f.write(line)

if __name__ == '__main__':
    main()
