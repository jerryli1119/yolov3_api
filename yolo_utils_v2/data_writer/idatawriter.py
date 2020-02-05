import abc


class IDataWriter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def write_data(self):
        return NotImplementedError
