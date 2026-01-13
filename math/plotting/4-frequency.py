#!/usr/bin/env python3
""" A script that plot a histogram of student scores for a project"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """ A function that plot a histogram of student scores for a project"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    plt.hist(student_grades, bins=10, range=(0, 100), edgecolor='black')
    plt.title('Project A')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xticks(np.arange(0, 101, step=10))
