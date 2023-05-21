"""
RANSAC for 2d lines
Algorythm:

I Hypotesys generation Stage
1. Sample 2d points (1. 2 ponts; 2. 5 points)
2. Model estimation (1. analytics; 2. MSE estimation)

II Hypotesys evaluation Stage

3. Inlier counting (%inlinear > threshold) 
    if True -> best params
    if False -> 1.
4. # iter > num_iter?

"""
import numpy as np
from line import Line

class RANSAC:
    def __init__(self) -> None:
        self.iter_num: int = 100
        self.inlin_thrsh: float = 0.8
        self.epsilon: float = 0.1
        self.best_params: dict = {}
        self.inlinears: list = []
        self.outliers: list = []
        self.score: int = 0
        self.points: np.ndarray = None;

    def set_case(self, case_params) -> None:
        pass

    def clear_case(self) -> None:
        pass

    def fit(self):
        pass

    def draw(self):
        pass