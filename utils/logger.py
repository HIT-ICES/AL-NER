# Python Standard Modules
import time
import logging.config
import os.path
from colorama import Fore, Style


class Logger(object):
    def __init__(self,  logger_name=__name__, config=None):
        """
        Configure log through LOGGING_DIC
        it supports printing to terminals and files
        """
        standard_format = '[%(asctime)s][%(message)s]'
        simple_format = '[%(levelname)s][%(asctime)s]%(message)s'
        
        if config is None :
            logfile_dir = os.path.join(os.getcwd(), logger_name.split('.')[0])
        else:
            logdir_prefix = config.param("LOGGER", "logdir_prefix", type="dirpath")

            word_embedding = config.param("WORDEMBEDDING", "method")
            corpus_name = config.param(word_embedding, "courpus_name", type="string")
            model_train = config.param("MODELTRAIN", "method")
            al_strategy = config.param("ActiveStrategy", "strategy", required=False)

            method = '_'.join(list(filter(None, [word_embedding, model_train, al_strategy])))
            logfile_dir = os.path.join(logdir_prefix, corpus_name, method)
        
        if not os.path.isdir(logfile_dir):
            os.makedirs(logfile_dir)

        logfile_prefix = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        logfile_path = os.path.join(logfile_dir, logfile_prefix + ".log")

        LOGGING_DIC = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': standard_format
                },
                'simple': {
                    'format': simple_format
                },
            },
            'filters': {},
            'handlers': {
                'console': {
                    'level': 'DEBUG',
                    'class': 'logging.StreamHandler',
                    'formatter': 'simple'
                },
                'default': {
                    'level': 'DEBUG',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'formatter': 'standard',
                    'filename': logfile_path,
                    'maxBytes': 1024 * 1024 * 5,  
                    'backupCount': 5,
                    'encoding': 'utf-8',
                },
            },
            'loggers': {
                '': {
                    'handlers': ['default', 'console'],
                    'level': 'DEBUG',
                    'propagate': True,
                },
            },
        }

        logging.config.dictConfig(LOGGING_DIC)
        self.logger = logging.getLogger(__name__)

    def debug(self, msg):
        """
        Defines the color of the output: debug--white
        :param msg:Text for log output
        :return:
        """
        self.logger.debug(Fore.WHITE + "DEBUG - " + str(msg) + Style.RESET_ALL)

    def info(self, msg):
        """
        Defines the color of the output: info--green
        :param msg:Text for log output
        :return:
        """
        self.logger.info(Fore.GREEN + "[INFO] - " + str(msg) + Style.RESET_ALL)

    def warning(self, msg):
        """
        Defines the color of the output: warning--red
        :param msg:Text for log output
        :return:
        """
        self.logger.warning(Fore.RED + "[WARNING] - " + str(msg) + Style.RESET_ALL)

    def error(self, msg):
        """
        Defines the color of the output: error--red
        :param msg:Text for log output
        :return:
        """
        self.logger.error(Fore.RED + "[ERROR] - " + str(msg) + Style.RESET_ALL)

    def critical(self, msg):
        """
        Defines the color of the output: critical--red
        :param msg:Text for log output
        :return:
        """
        self.logger.critical(Fore.RED + "[CRITICAL] - " + str(msg) + Style.RESET_ALL)
