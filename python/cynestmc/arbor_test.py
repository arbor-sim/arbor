#!/bin/bash/python3

#import py_arbor

import sys
import getopt
import json

def main(argv):
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:", ["ifile="])
    except getopt.GetoptError:
        print("arbor_test.py -i <inputfile>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("arbor_test.py -i <inputfile>")
            sys.exit()
        elif opt in ("-i","--ifile"):
            inputfile = arg
            with open(inputfile) as parameter:
                input_para = json.load(parameter)
                num_cell = input_para["cells"]
                print(num_cell)




if __name__ == '__main__':
    main(sys.argv[1:])





