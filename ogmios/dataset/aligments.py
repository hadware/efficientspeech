from pydantic import BaseModel, validator, field_validator


class Interval(BaseModel):
    start : float
    end : float
    annot: str

class Tier(BaseModel):
    type: str
    entries : list[tuple[float, float, str]]

    @property
    def intervals(self) -> list[Interval]:
        return [Interval(start=e[0], end=e[1], annot=e[2].strip()) for e in self.entries]

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
