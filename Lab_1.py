import numpy as np
import matplotlib.pyplot as plt
import math
import os
import scipy.integrate as integrate


# Межі
li = -np.pi
lf = np.pi
# Порядок ряду Фур'є
n = 50

# Створюємо масив x значень від -π до π з кроком
x = np.arange(li, lf, 0.05)

# Обчислюємо значення функції f(x)
y = x**22 * np.exp(-x**2/22)

#Побудова графіка
plt.plot(x, y)

plt.xlim([li, lf])
plt.ylim([-10, 70]) #межі

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Графік функції на інтервалі [-π, π]')

plt.show()


# Визначення функції для обчислення коефіцієнтів Фур'є та наближення
def fourier(li, lf, x, n):
    l = (lf - li) / 2 #піддовжина інтервалу
    m = n
    # нульовий коефіцієнт 
    a0 = (2.0 / l) * (integrate.quad(lambda x: (x**22 * np.exp(-x**2 / 22)), 0, l))[0]
    # косинусові коефіцієнти
    an = []
    an.append(a0 / 2.0)
# Обчислення an для кожної гармоніки
    for i in range(1, n + 1):
        an_i = (2.0 / l) * (integrate.quad(lambda x: (x**22 * np.exp(-x*2 / 22)) * np.cos(i * np.pi * x / l), 0, l))[
            0]
        an.append(an_i)
# Обчислення наближеного значення функції в точці x на основі ряду Фур'є
    fx = a0
    s = sum(an[i] * math.cos((i * x * np.pi) / l) for i in range(1, n + 1))
    fx += s

    return an,fx

# Обчислення ряду Фур'є в якійсь конкретній точці
x = math.pi / 2
#коефіцієнти Фур'є і наближене значення функції в точці x
an, fx = fourier(li, lf, x, n)

print('Коефіцієнти Фур\'є a_n =' + str(an))

print("Наближення рядом Фур'є з точністю до порядку", n, "в точцi x =", x, "дорівнює", fx)


# Створення масиву частот k для гармонік
k_n = np.arange(0, n + 1)

# Побудова графіків гармонік та відповідних функцій an
plt.subplot(2, 1, 1)
plt.stem(k_n, an)
plt.title('Графік гармоніки an')

plt.show()


def f(x):
    return x**22 * np.exp(-x**2 / 22)

# Створення масиву x значень від -π до π з кроком 0.1 для аналізу
x = np.arange(li, lf, 0.1)
# Ініціалізація списків для зберігання наближених та точних значень функції
y_approx = [] 
y_exact = [] 
y_approx_all = []
# Обчислення наближених значень функції для кожного x
for i in x:
    an, fx = fourier(li, lf, i, n) 
    y_approx.append(fx)

# Обчислення наближених значень функції для кожного порядку ряду Фур'є від 1 до n
for j in range(1,n):
    for i in x:
        an_all, fx_all = fourier(li, lf, i, j)
        y_approx_all.append(fx_all)
        # Побудова графіка для кожного порядку ряду Фур'є
    plt.plot(x, y_approx_all)
    # Очищення списку для наступного порядку
    y_approx_all = []

# Обчислення точних значень функції для кожного x
for i in x:
    y_exact.append(f(i))


# Обчислення відносної похибки наближення
erorr = []
# Обчислення похибки для кожного значення x
for i in range(0, len(y_exact)):
    erorr.append((y_approx[i] - y_exact[i]) / y_exact[i])

relative_error = np.abs(erorr) # Обчислення абсолютного значення відносної похибки


# Побудова графіка точних та наближених значень функції
plt.plot(x, y_exact, label='Точне значення функції')
plt.plot(x, y_approx, label='Наближане значення розкладу Фур\' є')
print(y_approx_all)

plt.legend()
plt.show()

# Графік відносної похибки
plt.plot(x, relative_error)
plt.title('Відносна похибка наближення')
plt.show()

print('Відносна похибка наближення:', relative_error) # Виведення відносної похибки наближення


# відкриття файлу для запису
with open('output.txt', 'w') as file:
    file.write("Порядок:" + str(n))
    file.write("\nОбчислені коефіцієнти an:"+str(an))
    file.write("\nОбчислені похибки відхилень:" + str(relative_error))
    file.close()


