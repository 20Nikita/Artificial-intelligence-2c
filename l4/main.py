import data
import line
import ransac
import matplotlib.pyplot as plt

def main():
    point_generator = data.Point_Generator(100, 0.3)
    x, y = point_generator.generate_case(None,None,eps=0.1)
    Line = line.Line(x,y)
    Line.estimate_params()
    inliers_x, inliers_y, outliers_x, outliers_y = Line.devide_points(x, y, eps = 0.1)
    k, b = Line.get_params()
    plt.plot(inliers_x, inliers_y, 'o', label='inliers')
    plt.plot(outliers_x, outliers_y, 'o', label='outliers')
    plt.plot(x, k*x + b, 'r', label='Fitted line')
    plt.legend()
    print("Без RANSAC")
    plt.show()
    Ransac =  ransac.RANSAC()
    case_params = {'x': x, 'y': y}
    Ransac.set_case(case_params)
    Ransac.fit()
    print("RANSAC c Sample = 2")
    Ransac.draw()
    Ransac =  ransac.RANSAC()
    case_params = {'x': x, 'y': y, 'n_pointsy': 5}
    Ransac.set_case(case_params)
    Ransac.fit()
    print("RANSAC c Sample = 5")
    Ransac.draw()

if __name__ == "__main__":
    main()