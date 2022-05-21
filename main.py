import matplotlib.pyplot as plt
import random

import math
import Grapths as gr
import numpy as np
from scipy import interpolate

N = 5


def function_x(x):
    y = x
    return y


def function_sin(x):
    y = []
    for ind in range(0, len(x)):
        y.append(math.sin(x[ind]))
    return y


def function_2_x(x):
    y = 2 ** x
    return y


def function_1_125(x):
    y = 1 / (1 + 25 * (x ** 2))
    return y


class SplineTuple:
    def __init__(self, a, b, c, d, x):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x = x


def Sweep(a, b, c, d):
    answer = [0] * len(d)
    gamma = []
    alpha = []
    betta = []

    gamma.append(b[0])
    alpha.append(-c[0] / gamma[0])
    betta.append(d[0] / gamma[0])

    for k in range(1, (len(b) - 1)):
        gamma.append(b[k] + a[k] * alpha[k - 1])
        alpha.append(-c[k] / gamma[k])
        betta.append((d[k] - a[k] * betta[k - 1]) / gamma[k])

    gamma.append(b[len(b) - 1] + a[len(b) - 1] * alpha[len(b) - 2])
    betta.append((d[len(b) - 1] - a[len(b) - 1] * betta[len(b) - 2]) / gamma[len(b) - 1])

    answer[len(d) - 1] = betta[len(d) - 1]
    for k in range(len(d) - 1, 0, -1):
        answer[k - 1] = alpha[k - 1] * answer[k] + betta[k - 1]

    return answer


def BuildSpline(x, y, n):
    splines = [SplineTuple(0, 0, 0, 0, 0) for _ in range(0, n)]
    for i in range(0, n):
        splines[i].x = x[i]
        splines[i].a = y[i]

    splines[0].c = splines[n - 1].c = 0.0

    # Решение СЛАУ относительно коэффициентов сплайнов c[i] методом прогонки для трехдиагональных матриц
    # Вычисление прогоночных коэффициентов - прямой ход метода прогонки

    A = [0] * (n)
    B = [0] * (n)
    C = [0] * (n)
    D = [0] * (n)

    B[0] = 1
    B[len(B) - 1] = 1

    for i in range(1, (len(B) - 1)):
        hi = x[i] - x[i - 1]
        hi1 = x[i + 1] - x[i]
        A[i] = hi
        B[i] = 2.0 * (hi + hi1)
        C[i] = hi1
        D[i] = 3.0 * ((y[i + 1] - y[i]) / hi1 - (y[i] - y[i - 1]) / hi)

    # Нахождение решения - обратный ход метода прогонки

    answer = Sweep(A, B, C, D)

    for i in range(1, n-1):
        splines[i].c = answer[i]

    # По известным коэффициентам c[i] находим значения b[i] и d[i]
    for i in range(1, n):
        hi = x[i] - x[i - 1]
        splines[i].d = (splines[i].c - splines[i - 1].c) / (3 * hi)
        splines[i].b = hi * (2.0 * splines[i].c + splines[i - 1].c) / 3.0 + (y[i] - y[i - 1]) / hi

    return splines


# Вычисление значения интерполированной функции в произвольной точке
def Interpolate(splines, x):
    if not splines:
        return None  # Если сплайны ещё не построены - возвращаем NaN

    n = len(splines)
    s = SplineTuple(0, 0, 0, 0, 0)

    if x <= splines[0].x:  # Если x меньше точки сетки x[0] - пользуемся первым эл-тов массива
        s = splines[0]
    elif x >= splines[n - 1].x:  # Если x больше точки сетки x[n - 1] - пользуемся последним эл-том массива
        s = splines[n - 1]
    else:  # Иначе x лежит между граничными точками сетки - производим бинарный поиск нужного эл-та массива
        i = 0
        j = n - 1
        while i + 1 < j:
            k = i + ((j - i) // 2)
            if x <= splines[k].x:
                j = k
            else:
                i = k
        s = splines[j]

    dx = x - s.x
    # Вычисляем значение сплайна в заданной точке
    return (s.a + s.b * dx + s.c * dx * dx + s.d * dx * dx * dx)

obj = SplineTuple(0, 0, 0, 0, 0)

x = np.linspace(-10, 10, 100000)
x_temp = x
y = function_1_125(x)

x_spline = np.linspace(-2, 2, N)  # Настройка сплайна
print(x_spline)
y1 = function_1_125(x_spline)
spline = BuildSpline(x_spline, y1, len(y1))
y_spline = []
for i in range(len(x_temp)):
    y_spline.append(Interpolate(spline, x_temp[i]))

print('len of y_spline')
print(len(y_spline))

spisokA = []
spisokB = []
spisokC = []
spisokD = []
spisokX = []
for ind in range(0, len(spline)):
    obj = spline[ind]
    spisokA.append(obj.a)
    spisokB.append(obj.b)
    spisokC.append(obj.c)
    spisokD.append(obj.d)
    spisokX.append(obj.x)

print("SpisokA")
print(spisokA)
print("SpisokB")
print(spisokB)
print("SpisokC")
print(spisokC)
print("SpisokD")
print(spisokD)
print("SpisokX")
print(spisokX)

obj = gr.picture()
obj.build_grapth(x_temp, y_spline, -2, 2, -0.2, 2, 'g')
obj.build_grapth(x, y, -2, 2, -0.2, 2, 'r')
# tick = interpolate.splrep(x, y)
# obj.build_grapth(tick[0], tick[1], -5, 5, -5, 5, 'g')
# -------------------------------------------------------------------------------
plt.show()
