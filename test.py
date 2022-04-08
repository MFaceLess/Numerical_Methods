from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt

EPS = 0.00001
H = 10**(-3)
plt.figure(figsize=(12, 7))

number_of_iteration = 0

def Modification(f,x0,leftPoint,rightPoint):
    global number_of_iteration
    number_of_iteration = 0
    df = 0
    if x0 == leftPoint:
        df = (f(x0)-f(x0-H))/H
    elif x0 == rightPoint:
        df = (f(x0+H)-f(x0))/H
    else:
        df = (f(x0+H)-f(x0-H))/(2*H)
    countingX = []
    countingX.append(x0)
    while True:
        number_of_iteration += 1
        value = f(x0)
        tempX = x0
        x0 = x0 - value/df
        countingX.append(x0)
        if (abs(x0 - tempX)<EPS):
            return x0,countingX

def search(f, negPoint, posPoint):
    global number_of_iteration
    number_of_iteration += 1
    midPoint = average(negPoint, posPoint)
    plt.subplot(2, 2, 1)
    plt.plot(midPoint, 0,'o')
    if closeEnough(negPoint, posPoint):
        print("Число итераций метода половинного деления:")
        print(number_of_iteration)
        number_of_iteration = 0
        return midPoint
    testValue = f(midPoint)
    if testValue > 0:
        return search(f,negPoint, midPoint)
    elif testValue < 0:
        return search(f,midPoint,posPoint)
    print("Число итераций метода половинного деления:")
    print(number_of_iteration)
    return midPoint

def Newtons_method(f,x0,leftPoint,rightPoint):
    global number_of_iteration
    number_of_iteration = 0
    df = 0
    countingX = []
    countingX.append(x0)
    while True:
        number_of_iteration += 1
        value = f(x0)
        if x0 == leftPoint:
            df = (f(x0)-f(x0-H))/H
        elif x0 == rightPoint:
            df = (f(x0+H)-f(x0))/H
        else:
            df = (f(x0+H)-f(x0-H))/(2*H)
        tempX = x0
        x0 = x0 - value/df
        countingX.append(x0)
        if (abs(x0 - tempX)<EPS):
            return x0,countingX

def derivative(f,x0,leftPoint,rightPoint):
    df = 0
    if (x0 == leftPoint):
        df = (f(x0)-f(x0-H))/H
    elif (x0 == rightPoint):
        df = (f(x0+H)-f(x0))/H
    else:
        df = ((f(x0+H)-f(x0-H))/(2*H))
    return df

def average(x,y):
    return float((x+y)/2.)

def closeEnough(x,y):
    if abs(x-y)<EPS:
        return 1
    return 0

def halfIntervalMethod(f,a,b):
    aVal = f(a)
    bVal = f(b)
    if aVal > 0 and bVal < 0:
        return search(f,b,a)
    elif aVal<0 and bVal>0:
        return search(f,a,b)
    else:
        print ("У аргументов разные знаки")
        return 0

def secant(f,x0,x1):
    global number_of_iteration
    number_of_iteration = 0
    listOfPointsX1 = []
    listOfPointsX0 = []
    listOfPointsX0.append(x0)
    listOfPointsX1.append(x1)
    while(True):
        number_of_iteration += 1
        f1 = f(x1)
        f0 = f(x0)
        temp = x1
        x1 = x1-f1/(f1-f0)*(x1-x0)
        x0 = temp
        listOfPointsX0.append(x0)
        listOfPointsX1.append(x1)
        if (abs(x1 - x0)<EPS):
            return x0 ,listOfPointsX0,listOfPointsX1
#***********************************************************
def myFunction1(x):
    y = (2**(x-0.1))-1
    return y 
#***********************************************************
def myFunction2(x):
    y = (x-0.2)**(3)
    return y
#____________________________Тело программы_______________________________
x = np.linspace(0,1,100)
y = (2**(x-0.1))-1
z = 0*x

t1 = time.time()
print("Ответ по методу половинного деления")
print(halfIntervalMethod(myFunction1,0,1))
t2 = time.time()

print("Абсолютная погрешность метода половинного деления:")
print(abs(halfIntervalMethod(myFunction1,0,1)-0.1))

print("Время выполнения Дихотомического поиска")
print("%12.4e"%(t2-t1))

y1 = (x-0.2)**(3)

t1 = time.time()
print("Ответ по методу половинного деления")
print(halfIntervalMethod(myFunction2,0,1))
t2 = time.time()

print("Время выполнения метода половинного деления")
print("%12.4e"%(t2-t1))

x2 = Newtons_method(myFunction1,0.5,0,1)[1]
y2 = [None]*(len(x))

x3 = Newtons_method(myFunction2,0.5,0,1)[1]
y3 = [None]*(len(x))

x4 = Modification(myFunction1,0.5,0,1)[1]
y4 = [None]*(len(x))

x50 = secant(myFunction1,0,1)[1]
x51 = secant(myFunction1,0,1)[2]
y5 = [None]*(len(x))

for suction in range(len(x50)):
    plt.subplot(2, 2, 4)
    plt.plot(x50[suction],0,'o')
    plt.plot(x51[suction],0,'o')

for iteration in range(len(x50)):    
    for ind in range(len(x)):
        y5[ind] = (myFunction1(x51[iteration])-myFunction1(x50[iteration]))*(x[ind]-x50[iteration])/(x51[iteration]-x50[iteration])+myFunction1(x50[iteration])
    plt.subplot(2, 2, 4)
    plt.plot(x,y5,'m')

for iteration in range(len(x4)):    
    for ind in range(len(x)):
        y4[ind] = (myFunction1(x4[iteration])+(x[ind]-x4[iteration])*derivative(myFunction1,x4[0],0,1))
    plt.subplot(2, 2, 3)
    plt.plot(x,y4,'m')

for suction in range(len(x4)):
    plt.subplot(2, 2, 3)
    plt.plot(x4[suction],0,'o')


for iteration in range(len(x3)):    
    for ind in range(len(x)):
        y3[ind] = (myFunction2(x3[iteration])+(x[ind]-x3[iteration])*derivative(myFunction2,x3[iteration],0,1))
    plt.subplot(2, 2, 2)
    plt.plot(x,y3,'m')

for suction in range(len(x3)):
    plt.subplot(2, 2, 2)
    plt.plot(x3[suction],0,'o')

# for suction in range(len(x2)):
#     plt.subplot(2, 2, 2)
#     plt.plot(x2[suction],0,'o')

# for iteration in range(len(x2)):    
#     for ind in range(len(x)):
#         y2[ind] = (myFunction1(x2[iteration])+(x[ind]-x2[iteration])*derivative(myFunction1,x2[iteration],0,1))
#     plt.subplot(2, 2, 2)
#     plt.plot(x,y2,'m')



print("Ответ по методу Ньютона")
t3 = time.time()
print(Newtons_method(myFunction1,0.9,0,1)[0])
t4 = time.time()
print("Число итераций метода Ньютона:")
print(number_of_iteration)


print("Абсолютная погрешность метода Ньютона:")
print(abs(Newtons_method(myFunction1,0.9,0,1)[0]-0.1))

print("Время выполнения алгоритма Ньютона")
print("%12.4e"%(t4-t3))

print("Ответ по методу Ньютона(Упрощенный)")
t1 = time.time()
print(datetime.now())
print(Modification(myFunction1,0.9,0,1)[0])
print(datetime.now())
t2 = time.time()
print("Число итераций метода Ньютона(Упрощенный):")
print(number_of_iteration)

print("Время выполнения алгоритма Ньютона(Упрощенный)")
print("%12.4e"%(t2-t1))

print("Абсолютная погрешность метода Ньютона(Упрощенный):")
print(abs(Modification(myFunction1,0.9,0,1)[0]-0.1))

print()

print("Проведем исследование метода секущих")
print()
print("Приближенный ответ метода секущих")
t1 = time.time()
print(secant(myFunction1,0,1)[0])
t2 = time.time()

print("Время выполнения метода секущих")
print("%12.4e"%(t2-t1))

print("Число итераций:")
print(number_of_iteration)

print("Абсолютная погрешность метода секущих:")
print(abs(secant(myFunction1,0,1)[0]-0.1))
#*******************************************************************
#Настройки графика
xnumbers = np.linspace(0, 1, 10)
ynumbers = np.linspace(-0.2, 1, 11)
#----------------------------------------------
plt.subplot(2, 2, 1)
plt.plot(x, y, 'r',x,z,'g') # r - red colour
plt.subplot(2, 2, 1)
plt.plot(x,y1,'b')
#------------------------------------------
plt.xlabel("x")
plt.ylabel("y")
plt.title("Function")
plt.xticks(xnumbers)
plt.yticks(ynumbers)
#plt.legend(['sin'])
plt.grid()
plt.axis([0, 1, -0.2, 1]) # [xstart, xend, ystart, yend]
#---------------------------------------------------------
plt.subplot(2, 2, 2)
plt.plot(x, y, 'r',x,z,'g') # r - red colour
plt.subplot(2, 2, 2)
plt.plot(x,y1,'b')
plt.subplot(2, 2, 2)
plt.plot(x,y2,'m')
#------------------------------------------
plt.xlabel("x")
plt.ylabel("y")
plt.title("Function")
plt.xticks(xnumbers)
plt.yticks(ynumbers)
#plt.legend(['sin'])
plt.grid()
plt.axis([0, 1, -0.2, 1]) # [xstart, xend, ystart, yend]
#---------------------------------------------------------
plt.subplot(2, 2, 3)
plt.plot(x, y, 'r',x,z,'g') # r - red colour
plt.subplot(2, 2, 3)
plt.plot(x,y1,'b')
#------------------------------------------
plt.xlabel("x")
plt.ylabel("y")
plt.title("Function")
plt.xticks(xnumbers)
plt.yticks(ynumbers)
#plt.legend(['sin'])
plt.grid()
plt.axis([0, 1, -0.2, 1]) # [xstart, xend, ystart, yend]
#---------------------------------------------------------
plt.subplot(2, 2, 4)
plt.plot(x, y, 'r',x,z,'g') # r - red colour
plt.subplot(2, 2, 4)
plt.plot(x,y1,'b')
#------------------------------------------
plt.xlabel("x")
plt.ylabel("y")
plt.title("Function")
plt.xticks(xnumbers)
plt.yticks(ynumbers)
#plt.legend(['sin'])
plt.grid()
plt.axis([0, 1, -0.2, 1]) # [xstart, xend, ystart, yend]
#---------------------------------------------------------

plt.show()




