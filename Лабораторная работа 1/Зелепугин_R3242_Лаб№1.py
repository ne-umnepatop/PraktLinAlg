import numpy as np
import random


def zamena(stg):
    """
    :return: массив чисел при шифровке и ключе 1 и стринг при нет
    """
    global alf
    if type(stg) is str:
        stg = list(stg)
        nw = list()
        for i in stg:
            nw.append(alf.find(i))
    else:
        nw = ''
        for i in stg:
            nw += alf[int(i) % len(alf)]  # чтобы замкнуть
    return nw


def obr(a):
    """Ищем обратную матрицу по модулю"""
    global module
    dt = round(np.linalg.det(a), 4)
    a = (np.linalg.inv(a) * dt) % module
    new_matrix = []  # устрою принудительнцю модулярность
    for i in a:
        new_string = []
        for j in i:
            elem = round(j, 4)
            new_string.append(elem)
        new_matrix.append(new_string)
    a = np.array(new_matrix) % module
    for i in range(1, module):  # обратное по модулю
        if (i * dt) % module == 1:
            break  # немного опасно, но компенсируется
    a *= i
    b = list()
    for j in a:
        string = list()
        for i in j:
            string.append(int((i % module)))
        b.append(string)
    return b


def shifr(stg, matrix):
    """Шифрую"""
    global module
    stg = zamena(stg)
    nw = list()
    n = np.array(matrix).shape[0]  # размерность исходной квадратной матрицы
    for i in range(0, len(stg), n):
        sh = np.array(stg[i:i + n]) % module
        hs = np.dot(matrix, sh) % module
        for x in hs:
            nw.append(int(round(x, 4) % module))  # некоторые ограничения возможны из-за 4, но красиво
    return zamena(nw)


def zadanie1(*args):
    print('\033[1m\033[4mДля первого задания\033[0m\033[22m'.center(100))
    print(f'\n  Я взял русский алфавит из строчных, заглавных и добил символов до 63,\n'
          f'потому что это наиболее близкое простое число к 64 как 2 в 6 степени,\n'
          f'в 31 символ не интересно укладываться, ведь тогда не будет даже пробелов,\n'
          f'а в 127 тупо лень.\n')
    print(f'Алфавит получился: {alf}\n')
    print(f'Ценным сообщением послужило \033[4m{cs}\033[0m.\n'
          f'')
    for a in args:
        print(f'Используя матрицу')
        print(' ' + str(a)[1:-1])
        print(f'С определителем {round(np.linalg.det(a), 4)}')
        шифрован = shifr(cs, a)
        print(f'шифрован: {" " * 14}{шифрован}')
        дешифрован = shifr(шифрован, obr(a))
        print(f'дешифрован: {" " * 12}{дешифрован}')
        испорчен1 = 'жЖ-' + шифрован[3:]
        print(f'испорчен: {" " * 14}{испорчен1}')
        дешифрован = shifr(испорчен1, obr(a))
        print(f'дешифрован испорченный: {дешифрован}')
        испорчен2 = 'ж' + шифрован[1:4] + 'Ж' + шифрован[5:7] + '-' + шифрован[8:]
        print(f'испорчен снова и иначе: {испорчен2}')
        дешифрован = shifr(испорчен2, obr(a))
        print(f'дешифрован испорченный: {дешифрован}\n')
    print(f'    Следует обратить внимание на то, что в результате внешнего вмешательсва менялся не один символ,\n'
          f'а комбинация из символов, по длине равная стороне шифрующей матрицы.\n'
          f'Это связано с матричным умножением, которое лежит в основе метода.\n'
          f'Таким образом, если бы я, шифруя матрицей 4х4, портил 1-й, 5-й, 9-й символы, потенциально оказались бы \n'
          f'не подверженными восстановлению группы 1-4, 5-8, 9-12 символы, то есть всё сообщение.\n\n'
          f'    На 5 строке я определил функцию, превращающую слово в число.\n'
          f'На 22 - нахождение обратной матрицы по модулю, воспользовавшись тем,\n'
          f'что от обычного нахождения обратной она отличается только определителем\n'
          f'На 48 - само шифрование Хилла, основанное на матричном умножении\n')


def check(matrix):
    flag = 888
    global module
    if round(np.linalg.det(matrix), 4) % module == 0: return False
    for i in obr(matrix):
        for j in i:
            if int(j % module) != j % module:
                return False
    if cs != shifr(shifr(cs, matrix), obr(matrix)): return False  # БЕЗ Этой проверки я чуть не помер
    return True


def gen_key():
    global module
    matrix = [[random.randint(0, 17), random.randint(0, 17), random.randint(0, 17)] for _ in range(3)]
    # Если поставить 63 он часто с определителем 0 или кратным выдаёт, из-за этого страдает глубина рекурсии
    while round(np.linalg.det(matrix), 4) <= 0 or not check(matrix) or round(np.linalg.det(matrix), 4) > 100:
        matrix = gen_key()
    #     Это тупая проверка на то, чтобы можно было раскодировать после этой матицы, то есть на обратимость
    return np.array(matrix)


def SLAY_salvation(B, C):
    B = B.T
    answered = False
    for i in range(15):
        for j in range(15):
            for k in range(15):
                C1 = np.array([C[0] + module * i, C[1] + module * j, C[2] + module * k])
                responce = np.linalg.solve(B, C1) % module
                flag = 0
                if responce is not None:
                    for e in responce:
                        if int(e) == round(e, 4):
                            flag += 1
                    if flag != 0:
                        answered = True
                        return responce % module
    if not answered: raise Exception(f'Между прочим, автор не умеет решать СЛАУ компьютером'
                                     f'\n Перезапустите.\n Не смог решить при \n{a4}')


def zadanie2():
    global module
    print('\033[1m\033[4mДля второго задания\033[0m\033[22m'.center(100))
    cs1 = 'Это уже длиннее'  # Нам известно это исходное сообщение
    __cs2 = ''  # Это случайно и поэтому неизвестно
    for i in range(15):  # можно было сделать генератором в одну строку, но упала бы читаемость
        __cs2 += alf[random.randint(0, 62)]  # заполнение той самой случайной строки

    # print(f'Было 1: {" " * 3}{cs1}')
    шифрован1 = shifr(cs1, a4)
    # print(f'шифрован1: {шифрован1}')
    шифрован2 = shifr(__cs2, a4)
    # print(f'шифрован2: {шифрован2}\n')
    print(f'\n  Представил, что у меня на руках два зашифрованных сообщения\n'
          f'\033[4m{шифрован1}\033[0m')
    print(f'и'.center(15))
    print(f'\033[4m{шифрован2}\n\033[0m\n'
          f'    Известно, что в них использовался шифр Хилла с одним и тем же ключом,\n'
          f'который мне неизвестен и генерируется функций на строке 110.\n'
          f'Использовал по назначению алфавит из предыдущего задания.\n'
          f'Но, чтобы было сильно интереснее, познавательнее, применимее к жизни,\n'
          f'Ключ-то мой размера 3 на 3!\n'
          f'Тем не менее, в статье Википедии напрямую сказано, что была механическая машина для \n'
          f'ключа 6 на 6, но мне как будто делать нечего.\n'
          f'    Соответственно, и сообщения я придумал не из 12, а из 15 символов, так снова веселее:\n'
          f'\033[4m{cs1}\033[0m')
    print('и'.center(15))
    print(f'\033[4m{__cs2}\033[0m\n'
          f'    Второе сообщение генерируется случайным образом в строке 143-145\n'
          f'и используется только для задания сообщения и вывода в самом конце (ну и в этом тексте).\n'
          f'В этом можно убедиться поиском по коду с помощью ctrl+f для __cs2\n'
          f'Таким образом я имитировал потерю одного из исходных сообщений.\n'
          f'')
    maTRIX, aNTIMATIXXX = np.array(list()), np.array(list())
    x = np.array(zamena(cs1))  # Хотя в реальной жизни мы не знаем алфавита для шифра, но тут...
    y = np.array(zamena(шифрован1))
    x = x.reshape(5, 3)
    y = y.reshape(5, 3)
    if round(np.linalg.det(x[:3]), 4) != 0:
        x = x[:3].T
        y = y[:3].T
    elif round(np.linalg.det(x[1:4]), 4) != 0:
        x = x[1:4].T
        y = y[1:4].T
    for i in range(3):
        maTRIX = np.append(maTRIX, SLAY_salvation(x, y[i]))
    np.set_printoptions(suppress=True)  # а то его вывод мне надоел
    maTRIX = np.array(maTRIX.reshape(3, 3)) % module
    aNTIMATIXXX = np.array(obr(maTRIX)) % module
    print(f'    Далее остаётся только понять, что шифрование - это Ах=у,\n'
          f'что х и у нам известны, а найти нужно А.')
    print(f'Матрица находится через транспонирование одной, что становится практически очевидным, если вручную '
          f'провести процесс шифровки. \nЭто вызвано тем, что обычно в СЛАУ Ах=у находится х, а у нас А.\n'
          f'Учитывая то, что длина сообщения, которое я шифровал больше, чем квадрат размерности матрицы, то есть боль'
          f'ше 9 в моём случае, данных у меня даже с избытком.\n'
          f'Также упомяну, что матрицу я нахожу построчно (повекторно)\n'
          f'в силу того, что превращаю наборы Ах=у в наборы X*a1=y, \n'
          f'а решением этой штуки станет вектор а1, у меня строка матрицы А.\n'
          f'Так я обнаружил прямую матрицу, шифрующую исходное сообщение. Функция нахождения обратной у меня уже есть, '
          f'так что остаётся только ее применить:\n')
    print(f'Получил, что исходно было {shifr(шифрован2, aNTIMATIXXX)}')
    print(
        f'А на самом деле было {" " * 5}{__cs2}\n(Повторюсь, эта переменная взята из скрытых и не использовалась при '
        f'подсчёте матриц - см строки 168-171).\n'
        f'То есть успешно проведены оперативно-розыскные работы, обнаружена ключ-матрица\n'
        f'и дешифровано неизвестное сообщение.\n')


def zadanie3():
    print('\033[1m\033[4mДля третьего задания\033[0m\033[22m'.center(100))
    print(f'\nАлфавит из 32 букв: {alf0}\n')
    Interesting_World = 'мило'

    G = np.array([[1, 0, 0, 0, 0, 1, 1],
                  [0, 1, 0, 0, 1, 0, 1],
                  [0, 0, 1, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1, 1, 1]])
    H = np.array([[0, 1, 1, 1, 1, 0, 0],
                  [1, 0, 1, 1, 0, 1, 0],
                  [1, 1, 0, 1, 0, 0, 1]])
    print(f'    Матрица G - на самом деле её строки должны быть базисом пространства кодовых слов.\n'
          f'То есть линейные комбинации строк G дают все возможные правильные (без ошибок) слова Хэмминга.\n'
          f'Таким образом, мной была выбрана такая порождающая матрица\n{G}\n'
          f'    Замечу, что на самом деле левая часть матрицы представляет все возможные позиции 1 на 4 разрядах.\n'
          f'Соответственно, последние три бита - соответствующие этим единицам проверочные биты.\n'
          f'Я решил писать их в конец, потому что могу.\n'
          f'Если бы писал в начало, то \033[4mправая часть\033[0m 4 на 4 могла бы быть единичной матрицей.\n'
          f'Могла БЫ потому, что\n'
          f'    Можно было бы заняться извращениями и избрать другой базис для всех 4-буквенных комбинаций.\n'
          f'И тогда я бы посчитал для них проверочные биты, и приписал бы каждой строке соответсвующую комабинацию\n'
          f'Причём не обязательно в начало, можно разместить как угодно, но удобно - как удобно.\n\n'
          f'    Матрица H - в случае Хэмминга 7, 4 - 7 на 3. У меня такая:\n'
          f'{H}\n'
          f'    Снова обратим внимание на то, что справа (то есть там, где я пишу проверочные биты) образовалась\n'
          f'единичная матрица 3 на 3, что легко трактуется как базис всех возможных комбинаций проверочных битов\n'
          f'Ну а слева я по приколу раскидал какие-то циферки. )\n'
          f'На самом деле, конечно, это символы, соответствующие сообщениям, которым, в свою очередь,\n'
          f'соответствуют данные проверочные символы.\n'
          f'    Опять же, если бы писал проверочные биты не в конец, то единичная матрица не стояла бы в конце,\n'
          f'Например, матрица H запросто могла бы выглядеть так, если бы моим сообщением \n'
          f'Были 2, 4, 6 и 7 биты, а пров. символами 1, 3 и 5, причем в качестве базиса проверочных битов я бы выбрал \n'
          f'{np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])}\n'
          f'Тогда моя матрица H\n'
          f'{np.array([[1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 1, 0], [0, 1, 0, 1, 1, 0, 1]])}\n'
          f'Аналогичный пример можно придумать для G, но я не буду, мне лень: в ней больше цифр.\n'
          f'')

    def zamena(to_code, key=1):
        global alf0
        result = ''
        if to_code[0] != '1' and to_code[0] != '0':
            for i in to_code:
                new = bin(alf0.find(i))[2:]
                new = '0' * (5 - len(new)) + new
                result += new
            return result
        else:
            for i in range(0, 20, 5):
                i = to_code[i:i + 5]
                new = alf0[int(i, 2)]
                result += new
            return result

    def shifr(a, key=1):
        a = list(a)
        result = list()
        if key == 1:
            for i in range(0, 20, 4):
                b = np.array([int(i) for i in a[i:i + 4]]).T
                """Вычислены контрольные биты и возвращено закодированное слово"""
                result.append(encode_hamming(b))
            # return f'{"".join(str(i) for i in result)}'
            return np.array(result)
        else:
            for b in a:
                """Контрольные биты, позиция ошибки, исправить, декодировать"""
                result.append(decode_hamming(b))
            return result

    def encode_hamming(b):
        # Преобразуем список битов в вектор-столбец (numpy array)
        b = np.array(b).reshape((4, 1))

        # Вычисляем закодированный вектор, умножая матрицу-генератор на вектор данных
        eb = np.dot(b.T, G) % 2

        # Преобразуем вектор обратно в список битов
        eb = list(eb.flatten())

        return eb

    def decode_hamming(eb):
        # Преобразуем список битов в вектор-столбец (numpy array)
        eb = np.array(eb).reshape((7, 1))
        # print(eb.T)

        # Вычисляем синдром (сумма по модулю 2 от произведения матрицы проверки четности на закодированный вектор)
        syndrome = np.dot(eb.T, H.T) % 2
        # print(syndrome)

        # Задаем таблицу исправления ошибок для декодирования
        error_table = {
            (0, 0, 0): -1,  # Нет ошибки
            (0, 0, 1): 6,  # Ошибка в 7-м бите
            (0, 1, 0): 6,  # Ошибка в 6-м бите
            (0, 1, 1): 0,  # Ошибка в 5-м бите
            (1, 0, 0): 4,  # Ошибка в 4-м бите
            (1, 0, 1): 1,  # Ошибка в 3-м бите
            (1, 1, 0): 2,  # Ошибка в 2-м бите
            (1, 1, 1): 3  # Ошибка в 1-м бите
        }

        # Исправляем ошибку, если она есть
        error_position = error_table[tuple(syndrome.flatten())]
        # print(error_position)
        # print(eb.T)
        if error_position != -1:
            eb[error_position] = 1 - eb[error_position]
            # print(f'333 {eb[error_position]}')

        # Декодируем данные, удаляя проверочные биты
        # print(eb.T)
        decoded_b = eb[:4].flatten().tolist()

        return decoded_b

    print(f'Итого в качестве интересного 4-х буквенного слова я выбрал: \033[4m{Interesting_World}\033[0m')
    coDEAD = zamena(Interesting_World)
    Hide_Interest = shifr(coDEAD)  # Кодируем данные
    print("Закодированное при помощи G слово с проверочными битами:")
    print(f'{Hide_Interest.flatten()}')
    print(f'На строке 336 успешно вредоносно вмешался в пятый символ, теперь слово такое:')
    Hide_Interest1 = Hide_Interest
    Hide_Interest1[0][4] = 1 - Hide_Interest1[0][4]
    print(f'{Hide_Interest1.flatten()}')
    secret_becomes_clear = ''
    for i in shifr(Hide_Interest1, 234):
        for j in i:
            secret_becomes_clear += str(j)
    print(f'Декодировал это вмешанное сообщение, получил: {zamena(secret_becomes_clear)}\n')
    print(f'На строке 341 инвертировал два символа, теперь слово такое:')
    Hide_Interest1 = Hide_Interest
    Hide_Interest1[0][4] = 1 - Hide_Interest1[0][4]
    Hide_Interest1[3][3] = 1 - Hide_Interest1[3][3]
    print(f'{Hide_Interest1.flatten()}')
    secret_becomes_clear = ''
    for i in shifr(Hide_Interest1, 234):
        for j in i:
            secret_becomes_clear += str(j)
    print(f'Декодировал это вмешанное сообщение, получил {zamena(secret_becomes_clear)}\n')
    print(f'А тут (352) инвертировал два подряд, слово такое:')
    Hide_Interest1 = Hide_Interest
    Hide_Interest1[0][4] = 1 - Hide_Interest1[0][4]
    Hide_Interest1[0][3] = 1 - Hide_Interest1[0][3]
    print(f'{Hide_Interest1.flatten()}')
    secret_becomes_clear = ''
    for i in shifr(Hide_Interest1, 234):
        for j in i:
            secret_becomes_clear += str(j)
    print(f'Декодировал это вмешанное сообщение, получил {zamena(secret_becomes_clear)}\n')
    print(f'Далее по заданию сменил 3 символа, слово такое:')
    Hide_Interest1 = Hide_Interest
    Hide_Interest1[0][4] = 1 - Hide_Interest1[0][4]
    Hide_Interest1[1][3] = 1 - Hide_Interest1[1][3]
    Hide_Interest1[4][3] = 1 - Hide_Interest1[4][3]
    print(f'{Hide_Interest1.flatten()}')
    secret_becomes_clear = ''
    for i in shifr(Hide_Interest1, 234):
        for j in i:
            secret_becomes_clear += str(j)
    print(f'Декодировал это вмешанное сообщение, получил {zamena(secret_becomes_clear)}\n')
    print(f'Естественно, было бы крайне интересно проверить подряд 3 символа (строки 373-375), слово такое:')
    Hide_Interest1 = Hide_Interest
    Hide_Interest1[0][0] = 1 - Hide_Interest1[0][0]
    Hide_Interest1[0][1] = 1 - Hide_Interest1[0][1]
    Hide_Interest1[0][2] = 1 - Hide_Interest1[0][2]
    print(f'{Hide_Interest1.flatten()}')
    secret_becomes_clear = ''
    for i in shifr(Hide_Interest1, 234):
        for j in i:
            secret_becomes_clear += str(j)
    print(f'Декодировал это вмешанное сообщение, получил {zamena(secret_becomes_clear)}\n')
    print(f'Было бы странно остановиться, так что 4 символа подряд, слово такое:')
    Hide_Interest1 = Hide_Interest
    Hide_Interest1[0][0] = 1 - Hide_Interest1[0][0]
    Hide_Interest1[0][1] = 1 - Hide_Interest1[0][1]
    Hide_Interest1[0][2] = 1 - Hide_Interest1[0][2]
    Hide_Interest1[0][3] = 1 - Hide_Interest1[0][3]
    print(f'{Hide_Interest1.flatten()}')
    secret_becomes_clear = ''
    for i in shifr(Hide_Interest1, 234):
        for j in i:
            secret_becomes_clear += str(j)
    print(f'Декодировал это вмешанное сообщение, получил {zamena(secret_becomes_clear)}\n')
    print(f'Меня сильно удивило то, что сообщение декодировалось правильно, попробую ещё раз')
    Hide_Interest1 = Hide_Interest
    Hide_Interest1[1][0] = 1 - Hide_Interest1[1][0]
    Hide_Interest1[1][2] = 1 - Hide_Interest1[1][2]
    Hide_Interest1[1][3] = 1 - Hide_Interest1[1][3]
    Hide_Interest1[1][4] = 1 - Hide_Interest1[1][4]
    print(f'{Hide_Interest1.flatten()}')
    secret_becomes_clear = ''
    for i in shifr(Hide_Interest1, 234):
        for j in i:
            secret_becomes_clear += str(j)
    print(f'Декодировал это вмешанное сообщение, получил {zamena(secret_becomes_clear)}\n')
    print(f'    В результате приведённых выше изысканий выяснил, что "внезапно" если в каждом из блоков кодировки,\n'
          f'то есть группах по 7 (для (7,4)),\n'
          f'допущено не более ошибки, код, предназначенный для исправления не более одной ошибки,\n'
          f'исправляет эти ошибки.\n'
          f'    При этом если допустить более 1 ошибки на блок, то код перестанет корректно расшифровывать сообщение,\n'
          f'И, если портить два символов в двух блоках (по заданию нельзя портить больше 4), можно испортить\n'
          f'две буквы, хотя задействовать два блока для этого не обязательно, как показано в послднем примере,\n'
          f'где "сломались" две буквы, но корректировался только один блок.\n'
          f'Это вызвано тем, что буквы кодируются 5 знаками, то есть блок может кодировать "половину" одной буквы\n'
          f'и "половину другой". Почему при порче 4 знаков предпоследний раз я получил исходное слово,\nмне пока нет времени'
          f' анализировать, назову это "чудом"\n'
          f'    Если вы внимательно смотрели мой код, то выяснили,\n'
          f'что функции для 3-го задания я задал в пространстве его имён.\n'
          f'Так и надо.\n')


def zadanie4():
    print('\033[1m\033[4mДля четвёртого задания\033[0m\033[22m'.center(100))
    print('ЭссеСеСер'.center(82))
    print(f'\n    Итак, дано: поле 64х64, ключ, два состояния каждой клетки.\n'
          f'Очевидно, поле представимо в виде слова из 64, то есть 2 в 6 степени, цифр 0 или 1.\n'
          f'Известно, что код Хэмминга ищет ошибку. Для кодирования будут задействоваться 6 битов кодирования.\n'
          f'Очевидно, так и хочется принять позицию ключа за позицию ошибки, раз уж код её ищет.\n'
          f'В хорошем случае (что я предварительно обсужу со вторым, раз уж задача допускает)\n'
          f'будет перевёрнуто 0 или 64 монеты. Тогда я инвертирую одну, и это ответ.\n'
          f'')
alf0 = 'абвгдежзиклмнопрстуфхцчшыэюя'
alf = alf0 + alf0.upper() + '/., -01'
alf0 += ' .,-'
module = len(alf)
cs = 'Капец Ценное'
a1 = np.array([[3, 2], [1, 2]])
a2 = np.array([[3, 2, 1], [1, 2, 3], [1, 1, 2]])
a3 = np.array([[2, 4, 1, 2], [1, 2, 3, 1], [2, 1, 4, 3], [2, 3, 2, 3]])
a4 = gen_key()
if __name__ == '__main__':
    print()
    # zadanie1(a1, a2, a3)  # должно быть закомментировано для скрытия отчёта по заданию 1
    # zadanie2()  # должно быть закомментировано для скрытия отчёта по заданию 2
    # zadanie3()  # должно быть закомментировано для скрытия отчёта по заданию 3
    zadanie4()
