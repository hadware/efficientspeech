from pydantic import BaseModel

class Interval(BaseModel):
    start : float
    end : float
    annot: str

class Tier(BaseModel):
    type: str
    entries : list[Interval]

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
