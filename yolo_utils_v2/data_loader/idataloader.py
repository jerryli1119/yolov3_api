import abc


class IDataLoader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def frame_iter(self):
        return NotImplementedError

    # @abc.abstractmethod
    # def pre_load(self):
    #     return NotImplementedError


class IFormatLoader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def imread(self):
        return NotImplementedError
