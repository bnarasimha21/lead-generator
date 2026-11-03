"""The strict output schema for a lead (issue 2).

Every field carries its value next to the verbatim quote from the transcript
that justifies it. A field with no evidence must have a null value: the
schema itself rules out "confident but unquoted" answers. app/verify.py then
checks the quote is really a substring of the transcript before the value is
trusted.
"""

from typing import Optional

from pydantic import BaseModel, Field, model_validator

FIELD_NAMES = (
    "name",
    "company",
    "role",
    "email",
    "phone",
    "interest",
    "next_step",
    "budget_mentioned",
    "urgency",
)


class EvidencedField(BaseModel):
    value: Optional[str] = Field(
        default=None, description="The extracted value, or null if not stated."
    )
    evidence: Optional[str] = Field(
        default=None,
        description="A verbatim substring of the transcript that supports value, or null.",
    )

    @model_validator(mode="after")
    def _value_requires_evidence(self) -> "EvidencedField":
        if self.value is not None and (self.evidence is None or self.evidence == ""):
            raise ValueError("a non-null value must carry non-empty evidence")
        return self


class Lead(BaseModel):
    name: EvidencedField = Field(default_factory=EvidencedField)
    company: EvidencedField = Field(default_factory=EvidencedField)
    role: EvidencedField = Field(default_factory=EvidencedField)
    email: EvidencedField = Field(default_factory=EvidencedField)
    phone: EvidencedField = Field(default_factory=EvidencedField)
    interest: EvidencedField = Field(default_factory=EvidencedField)
    next_step: EvidencedField = Field(default_factory=EvidencedField)
    budget_mentioned: EvidencedField = Field(default_factory=EvidencedField)
    urgency: EvidencedField = Field(default_factory=EvidencedField)

    def to_display_dict(self, rejected: Optional[list] = None) -> dict:
        rejected = rejected or []
        out = {}
        for name in FIELD_NAMES:
            field = getattr(self, name)
            out[name] = {
                "value": field.value,
                "evidence": field.evidence,
                "rejected": name in rejected,
            }
        return out

    @classmethod
    def empty(cls) -> "Lead":
        return cls()
