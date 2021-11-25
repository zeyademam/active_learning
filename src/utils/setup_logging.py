import logging
import datetime as dt
import os


class LoggingFormatter(logging.Formatter):
    converter = dt.datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s,%03d" % (t, record.msecs)
        return s


def setup_logging(directory, filename):
    print(f"Logging filename: {filename}")
    logger = logging.getLogger("ActiveLearning")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=os.path.join(directory, filename), mode="w+")
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    formatter = LoggingFormatter(fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d,%H:%M:%S.%f")
    file_handler.setFormatter(formatter)

    return logger
