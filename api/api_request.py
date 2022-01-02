import typing as t

from enum import Enum


class Type(int, Enum):
    Pick_up   = 0,
    Drop_off  = 1,
    Incentive = 2


class ApiRequest:
    def __init__(self, data: t.Dict, _type: Type):
        self.data = data
        self.type = _type

    @staticmethod
    def parse(_request: t.Dict):
        _data = _request['data']
        _type = Type(_request['type'])

        return ApiRequest(_data, _type)
