# A simple torch style logger
# (C) Wei YANG 2017
from __future__ import absolute_import
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

__all__ = ['Logger', 'LoggerMonitor', 'savefig']

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)
    
def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        # for _, name in enumerate(names):
        #     x = np.arange(len(numbers[name]))
        #     plt.plot(x, np.asarray(numbers[name]))
        # plt.legend([self.title + '(' + name + ')' for name in names])
        # plt.grid(True)
        fig, axes = plt.subplots(3, 1, figsize=(6, 12))  # 3行1列

        # Learning Rate
        x = np.arange(len(numbers["Learning Rate"]))
        axes[0].plot(x, numbers["Learning Rate"], label="Learning Rate")
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_title("Learning Rate")

        # Train Loss & Valid Loss
        for name in ["Train Loss", "Valid Loss"]:
            x = np.arange(len(numbers[name]))
            axes[1].plot(x, numbers[name], label=name)
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_title("Loss")

        # Train Acc. & Valid Acc.
        for name in ["Train Acc.", "Valid Acc."]:
            x = np.arange(len(numbers[name]))
            axes[2].plot(x, numbers[name], label=name)
        axes[2].legend()
        axes[2].grid(True)
        axes[2].set_title("Accuracy")

    def close(self):
        if self.file is not None:
            self.file.close()

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)
                    
if __name__ == '__main__':

    paths = {
    }

    field = ['Valid Acc.']

    monitor = LoggerMonitor(paths)
    monitor.plot(names=field)
    savefig('test.eps')