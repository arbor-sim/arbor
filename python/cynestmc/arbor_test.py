#!/bin/bash/python3

import py_arbor

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
                # py_arbor(input_para)
                recipe = py_arbor.py_recipe(input_para)
                decomp = py_arbor.domain_decomposition(recipe, rules)
                model = py_arbor.py_model(recipe, decomp)
                dt = input_para["dt"]
                tfinal = input_para["tfinal"]
                model.run(dt,tfinal)
                print("Number spikes: ", model.num_spikes())
        else:
            print("No input json file, use default settings instead.")
            default_input ={
                "cells": 1000,
                "synapses": 500,
                "compartments": 50,
                "dt": 0.02,
                "tfinal": 100.0,
                "all_to_all": False
            }
            recipe = py_arbor.py_recipe(default_input)
            decomp = py_arbor.domain_decomposition(recipe, rules)
            model = py_arbor.py_model(recipe, decomp)
            dt = default_input["dt"]
            tfinal = default_input["tfinal"]
            model.run(dt, tfinal)
            print("Number spikes: ", model.num_spikes())

if __name__ == '__main__':
    main(sys.argv[1:])





