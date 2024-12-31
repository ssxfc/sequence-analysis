import unittest

import torch
import torch.nn as nn


class TestYield(unittest.TestCase):
    def my_ge(self):
        for i in range(10):
            yield 1
        print("after")

    def test_yield(self):
        x = self.my_ge()
        for i in x:
            print(i)
