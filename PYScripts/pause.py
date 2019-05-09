from logger import pause_logger

#Создаем необходимые структуры данных в нужном файле
#И далее используем их в функциях модуля через global
PAUSE = True

def pause_unpause():
    global PAUSE
    
    if PAUSE:
        pause_logger.info("Algorithm pause")
    else:
        pause_logger.info("Algorithm unpause ")
    
    old_PAUSE = PAUSE
    PAUSE = False if PAUSE else True
    print("Switch PAUSE from", old_PAUSE, "to", PAUSE, sep=" ")
    
    return PAUSE
    