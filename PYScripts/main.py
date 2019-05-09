import numpy as np
import matplotlib.pyplot as plt
import sys

from logger import main_logger
import pause

import datetime

def main(list_Ball,list_TeamBlue,list_TeamYellow,list_BallisInside):
    main_logger.info("Start main script")
    
    #main_logger.info("QQQQ")
    print(datetime.datetime.now())
    #print(sys.argv)
    #print(sys.path)
    #print(list_Ball)
    
    #print(list_TeamBlue)
    #print(list_TeamYellow)
    #print(list_BallisInside)


    #Используем квалифицированное имя
    #Заменяя обычный импорт этим from <name module> import <...> можно использовать 
    #неквалифицированные имена
    
    q = [1]*7 if not pause.PAUSE else [0]*7
    c = [q]*12
    #print(type(c))
    #print(c)
    
    return c
    