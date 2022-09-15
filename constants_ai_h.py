#!/usr/bin/env python3
import sys, os

BASE_DIR_GIT = r'C:\Users\erikw\git'
DIR_HELPER = os.path.join(BASE_DIR_GIT, 'helper')

sys.path.insert(1, DIR_HELPER)
import helper as h


RANDOM_STATE = 42
BASE_DIR_AI = r'C:\ai'

DIR_AI_HELPER = h.j(BASE_DIR_GIT, 'ai_helper')

DIR_RESULTS = h.j(BASE_DIR_AI, 'results')

