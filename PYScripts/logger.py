import logging
import logging.config
import os

#Конфиг файл для логгера лежит там же где и все скрипты - в PYScripts
path_to_logger_config = os.path.join(os.path.dirname(os.path.abspath(__file__))
                                                         ,"logging.conf")

logging.config.fileConfig(path_to_logger_config)

main_logger = logging.getLogger("mainLogger")
pause_logger = logging.getLogger("pauseLogger")

