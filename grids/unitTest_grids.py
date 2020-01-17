# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:55:31 2020

@author: Arthur
"""
import unittest
import numpy as np
from grids import RectangularGrid, RectangularData


class TestRectangularData(unittest.TestCase):
    def test1_coarse_grain(self):
        data = RectangularData(data=np.array([1, 2, 3, 4, 5, 6, 7, 8]))
        coarse = data.coarse_grain(2)
        self.assertTrue((coarse.data == np.array([1.5, 3.5, 5.5, 7.5])).all())

    def test2_coarse_grain(self):
        data = RectangularData(data=np.array([[1,2,3], [4,5,6]]))
        coarse = data.coarse_grain(2)
        test = coarse.data == np.array([[3,]])
        self.assertTrue(test.all())

    def test3_coarse_grain(self):
        data = RectangularData(data=np.array([[1,2,3,4], [5,6,7,8]]))
        coarse = data.coarse_grain(2)
        test = coarse.data == np.array([[3.5, 5.5]])
        self.assertTrue(test.all())

    def test4_coarse_grain(self):
        data = RectangularData(data=np.array([[1,2,3,4], [5,6,7,8],
                                              [9,10,11,12], [13,14, 15, 16]]))
        coarse = data.coarse_grain(2, dims=(1,))
        self.assertEqual(coarse.data.shape, (4, 2))
        test = coarse.data == np.array([[1.5, 3.5], [5.5, 7.5], 
                                        [9.5, 11.5], [13.5, 15.5]])
        self.assertTrue(test.all())


if __name__ == '__main__':
    unittest.main()
