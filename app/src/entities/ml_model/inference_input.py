from typing import Any


class InferenceInput:
    def __init__(self, data: Any):        
        self.__data = data

    @property
    def data(self) -> Any:
        return self.__data

    @data.setter 
    def data(self, value: Any):
        self.__data = value

