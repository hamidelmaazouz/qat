from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class PassResultKey:
    ir_id: int
    pass_id: str


class PassResultSet:
    """
    Models a collection of pass results with caching and aggregation capabilities.

    Passes that merely compute analyses on the IR must not invalidate prior results. Passes that mutate any IR
    units are likely to invalidate previous results. These result caching complexities could be closely modelled
    around some lazy execution style of the pass pipeline as well as lazy construction and evaluation of the pass
    dependency graph.

    Today's needs are very simple and I will be adding features as time goes. For now it's just a set of
    pass results
    """

    def __init__(self, *tuples):
        self._data: Dict[PassResultKey, Any] = {}
        for t in tuples:
            self.add_result(t[0], t[1], t[2])

    @property
    def data(self):
        return self._data

    def update(self, other_rs):
        if not isinstance(other_rs, PassResultSet):
            raise ValueError(
                f"Invalid type, expected PassResultSet, but got {type(other_rs)}"
            )
        self._data.update(other_rs._data)

    def add_result(self, ir_id, pass_id, value):
        return self._data.setdefault(PassResultKey(ir_id, pass_id), value)

    def get_result(self, pass_id):
        key = next((rk for rk in self._data if rk.pass_id == pass_id))
        return self._data[key]


class PassConcept(ABC):
    """
    Base class describing the abstraction of a pass.
    """

    @abstractmethod
    def run(self, ir, *args, **kwargs) -> PassResultSet:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class PassModel(PassConcept):
    """
    Implement the polymorphic pass API.
    A wrapper for any object providing a run() method that accepts some unit of IR.
    """

    def __init__(self, pass_obj):
        self._pass = pass_obj

    def run(self, ir, *args, **kwargs):
        return self._pass.run(ir, args, kwargs)

    def name(self):
        return self._pass.name()


class PassInfoMixin:
    def id(self):
        pass

    def name(self):
        pass


class PassManager(PassInfoMixin):
    """
    The pass manager acts as a pass itself. In doing so, it runs a sequence of passes over some unit of IR
    and aggregates results from them.

    Result aggregation could potentially involve caching and result invalidation concepts as described in
    PassResultCache.

    However, for today's needs, it just accumulates and returns a set of results from the passes.
    """

    def __init__(self):
        self.passes: List[PassModel] = []

    def run(self, ir, *args, **kwargs):
        global_rs = PassResultSet()
        for p in self.passes:
            pass_rs = p.run(ir, args, kwargs)
            global_rs.update(pass_rs)
        return global_rs

    def add(self, pass_obj):
        self.passes.append(PassModel(pass_obj))
