import numpy as np
import matplotlib.pyplot as plt


class picture:
    def build_grapth(self, x, y, x_leftPoint, x_rightPoint, y_leftPoint, y_rightPoint, color):
        # *******************************************************************
        # Настройки графика
        xnumbers = np.linspace(x_leftPoint, x_rightPoint, 10)
        ynumbers = np.linspace(y_leftPoint, y_rightPoint, 10)
        # ----------------------------------------------
        # plt.subplot(2, 2, 1)
        plt.plot(x, y, color)  # r - red colour
        # ------------------------------------------
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Function")
        plt.xticks(xnumbers)
        plt.yticks(ynumbers)
        # plt.legend(['sin'])
        plt.grid(True)
        plt.axis([x_leftPoint, x_rightPoint, y_leftPoint, y_rightPoint])  # [xstart, xend, ystart, yend]
