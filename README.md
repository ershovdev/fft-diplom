# fateco


## Описание

Программный продукт **fateco** (**FA**st **Te**nzorized **CO**nvolutions) реализован в рамках дипломной работы `Исследование распараллеливания
вычисления “быстрой свёртки”`. В состав программы входит набор методов для построения малоранговой (малопараметрической) аппроксимации тензорного поезда и методов быстрого вычисления сверток с использованием данной аппроксимации.


## Установка

1. Установить интерпретатор [python](https://www.python.org) (версия >= 3.7; предпочтительно использовать сборку [anaconda](https://www.anaconda.com), в которую входит интерпретатор языка и набор полезных библиотек);
2. Установить python библиотеки [numpy](https://numpy.org), [scipy](https://www.scipy.org), [matplotlib](https://matplotlib.org/) и [seaborn](https://seaborn.pydata.org):
    ```bash
    pip install numpy scipy matplotlib seaborn
    ```


## Использование

- Демонстрационный расчет с малопараметрическим приближением модельных многомерных аналитических функций:
    ```python
    python fateco.py demo
    ```
    > Результат будет отображен непосредственно в консоли.

- Исследование зависимости степени сжатия и точности аппроксимации модельных многомерных аналитических функций от размерности задачи:
    ```python
    python fateco.py dims
    ```
    > Результат будет представлен в виде ряда графиков в папке `results`.

- Демонстрационный расчет с малопараметрическим сложением модельных многомерных аналитических функций:
    ```python
    python fateco.py demo_add
    ```
    > Результат будет отображен непосредственно в консоли.

- Исследование зависимости степени сжатия и точности при сложении модельных многомерных аналитических функций от размерности задачи:
    ```python
    python fateco.py dims_add
    ```
    > Результат будет представлен в виде ряда графиков в папке `results`.

- Демонстрационный расчет с малопараметрическим умножением модельных многомерных аналитических функций:
    ```python
    python fateco.py demo_mul
    ```
    > Результат будет отображен непосредственно в консоли.

- Исследование зависимости степени сжатия и точности при умножении модельных многомерных аналитических функций от размерности задачи:
    ```python
    python fateco.py dims_mul
    ```
    > Результат будет представлен в виде ряда графиков в папке `results`.

- Демонстрационный расчет с вычислением многомерных сверток при использовании TT-разложения:
    ```python
    python fateco.py demo_conv
    ```
    > Результат будет отображен непосредственно в консоли.

- Исследование зависимости эффективности алгоритма вычисления многомерных сверток при использовании TT-разложения:
    ```python
    python fateco.py dims_conv
    ```
    > Результат будет представлен в виде ряда графиков в папке `results`.