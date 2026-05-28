"""
Module: models.py
Description:
    This module contains functions for performing various machine learning models and calculations.
"""

from typing import Union, List


class LinearRegression:
    def __init__(self, weight: Union[int, float]=1, intercept: Union[int, float]=0):
        """
        Initializes the LinearRegression model.

        Parameters
        ----------
        weight : int or float, optional
            The weight (slope) of the regression line. Default is 1.
        intercept : int or float, optional
            The intercept (y-intercept) of the regression line. Default is 0.
        """
        self.weight = weight
        self.intercept = intercept

    def _calculate(self, value: Union[int, float]) -> Union[int, float]:
        """
        Internal method to perform the linear regression calculation.

        Parameters
        ----------
        value : int or float
            The input value for the regression.

        Returns
        -------
        int or float
            The result of the linear regression calculation.
        """
        return value * self.weight + self.intercept

    def predict(
        self,
        data: List[Union[int, float]]
    ) -> List[Union[int, float]]:
        """
        Performs a linear regression calculation.

        Parameters
        ----------
        data : list of int or float
            The input data for the regression.

        Returns
        -------
        list of int or float
            The result of the linear regression calculation.
        """
        return [self._calculate(value) for value in data]

class Dummy:
    pass

class Forest:
    pass

class KNN:
    pass
