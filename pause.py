#Создаем необходимые структуры данных в нужном файле
#И далее используем их в функциях модуля через global
PAUSE = True

def pause_unpause():
    global PAUSE
    old_PAUSE = PAUSE
    PAUSE = False if PAUSE else True
    print("Switch PAUSE from", old_PAUSE, "to", PAUSE, sep=" ")
    