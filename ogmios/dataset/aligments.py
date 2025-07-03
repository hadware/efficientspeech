from typing import Iterable

from pydantic import BaseModel


class Interval(BaseModel):
    start : float
    end : float
    annot: str

class Tier(BaseModel):
    type: str
    entries : list[tuple[float, float, str]]

    @property
    def intervals(self) -> Iterable[Interval]:
        prev_interval: Interval = None
        for e in self.entries:
            interval = Interval(start=e[0], end=e[1], annot=e[2].strip())
            if prev_interval is not None:
                if prev_interval.end > interval.start:
                    interval.start = prev_interval.end
                if prev_interval.end < interval.start:
                    yield Interval(start=prev_interval.end, end=interval.start, annot="sil")
            yield interval
            prev_interval = interval

class AlignmentFile(BaseModel):
    start: float
    end: float
    tiers: dict[str, Tier]

    @property
    def words(self) -> Tier:
        return self.tiers['words']

    @property
    def phones(self) -> Tier:
        return self.tiers['phones']
