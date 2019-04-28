import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
import logging.config

import pause

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("mainLogger")


def main(list_Ball,list_TeamBlue,list_TeamYellow,list_BallisInside):
    logger.info("Start main script")
    #В global передаем имя как есть
    #Далее используем квалифицированное имя
    #Заменяя обычный импорт этим from <name module> import <...> можно использовать 
    #неквалифицированные имена
    global PAUSE   
    
    print(sys.argv)
    print(list_Ball)
    print(list_TeamBlue)
    print(list_TeamYellow)
    print(list_BallisInside)
    #print(sys.path)
    """
    scalar = np.dot(vector1, vector2)
    print(scalar)
    t = np.arange(0.0, 0.1, 0.0001)
    s = np.cos(scalar*t)
    plt.plot(t, s)
    plt.grid(True)
    plt.savefig("result.png")
    c = vector1 + vector2
    """
    q = [1]*7 if not pause.PAUSE else [0]*7
    c = [q]*4
    print(type(c))
    print(c)
    return c
    