from typing import Literal
from pydantic import BaseModel, Field, model_validator

Label = Literal['supported', 'contradicted', 'unknown']

class Claim(BaseModel):
    quote: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    label: Label
    evidence_fact_ids: list[str] = Field(default_factory=list)
    explanation: str = ''

class Prediction(BaseModel):
    answer_id: str
    version: Literal['v1', 'v2']
    claims: list[Claim]

    @model_validator(mode='after')
    def validate_spans(self):
        for c in self.claims:
            if self.answer_id and c.end_char <= c.start_char:
                raise ValueError('Invalid span length')
            if c.end_char - c.start_char != len(c.quote):
                raise ValueError('Quote length must equal end_char-start_char')
        return self
