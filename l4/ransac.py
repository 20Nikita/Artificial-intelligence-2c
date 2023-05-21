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
import matplotlib.pyplot as plt
import line

class RANSAC:
    def __init__(self) -> None:
        self.iter_num: int = 100
        self.n_pointsy: int = 2
        self.inlin_thrsh: float = 0.8
        self.epsilon: float = 0.1
        self.best_params: dict = {}
        self.inlinears_x: list = []
        self.inlinears_y: list = []
        self.outliers_x: list = []
        self.outliers_y: list = []
        self.score: int = 0
        self.x: np.ndarray = None
        self.y: np.ndarray = None

    def set_case(self, case_params) -> None:
        if 'iter_num' in case_params.keys():
            self.iter_num = case_params['iter_num']
        if 'n_pointsy' in case_params.keys():
            self.n_pointsy = case_params['n_pointsy']
        if 'inlin_thrsh' in case_params.keys():
            self.inlin_thrsh = case_params['inlin_thrsh']
        if 'epsilon' in case_params.keys():
            self.epsilon = case_params['epsilon']
        if not ('x' in case_params.keys() and 'y' in case_params.keys()):
            raise ValueError(f"case_params обязан включать в себя ключи 'x' и 'y'")
        self.x = case_params['x']
        self.y = case_params['y']

    def clear_case(self) -> None:
        pass

    def fit(self):
        for i in range(self.iter_num):
            ind = range(len(self.x))
            ind_sampl = np.random.choice(ind, self.n_pointsy)
            x_sampl = self.x[ind_sampl]
            y_sampl = self.y[ind_sampl]
            Line = line.Line(x_sampl,y_sampl)
            Line.estimate_params()
            inliers_x, inliers_y, outliers_x, outliers_y = Line.devide_points(self.x, self.y, self.epsilon)
            score = len(inliers_x) / len(self.x)
            if score > self.score:
                k, b = Line.get_params()
                self.best_params = {'k': k, 'b': b}
                self.score = score
                self.inlinears_x = inliers_x
                self.inlinears_y = inliers_y
                self.outliers_x = outliers_x
                self.outliers_y = outliers_y

    def draw(self):
        plt.plot(self.inlinears_x, self.inlinears_y, 'o', label='inlinears')
        plt.plot(self.outliers_x, self.outliers_y, 'o', label='outliers')
        plt.plot(self.x, self.best_params['k']*self.x + self.best_params['b'], 'r', label='Fitted line')
        plt.legend()
        plt.show()