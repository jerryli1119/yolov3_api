import time
import logging
import numpy as np
from config import logger_cfg


def f1(x, y):
    print(x)
    time.sleep(1)
    print(y)

def f2(y):
    print(y)
    time.sleep(1)


class TimerWrapper:
    """
    for test
    """
    def __init__(self):
        self.time_dict = {}
        self.logger = logging.getLogger(logger_cfg['name'])

    def __call__(self, f, **kwargs):
        t1 = time.time()
        res = f(**kwargs)
        t2 = time.time()
        time_interval = t2 - t1
        try:
            self.time_dict[f.__name__] += time_interval
        except KeyError:
            self.time_dict[f.__name__] = time_interval

        return res

        # if f.__name__ not in self.time_dict:
        #     self.time_dict[f.__name__] = time_interval
        # else:
        #     self.time_dict[f.__name__] += time_interval

    def analysis(self, num_files):
        # total = self.time_dict['tracking']
        total = np.sum(list(self.time_dict.values()))




        for k, v in self.time_dict.items():
            self.logger.info('%s, %s', k, v/total)

        # logging.info('number of files=%s', average_count)
        # logging.info('len = %s', len(self.time_dict))
        # logging.info('total count = %s', self.total_count)
        total = total - self.time_dict['update_preprocess']
        self.logger.info('fps=%s', num_files / total)



    def show_info(self):
        for k, v in self.time_dict.items():
            self.logger.info('%s, %s', k, v)


timer_wrapper = TimerWrapper()


def main():
    tw = TimerWrapper()
    tw(f1, x=123, y=100)
    # tw.run(f2, y=666)
    # tw.run(f1, x=123)

    # tw.run(f1)
    print(tw.time_dict)

if __name__ == "__main__":
    main()

