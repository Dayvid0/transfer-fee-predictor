"""Public API contracts. Values are EUR and dates describe the supplied profile."""
from datetime import date, datetime, timezone
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import config

League = Literal['GB1', 'ES1', 'IT1', 'L1', 'FR1']
Position = Literal['GK', 'DEF', 'MID', 'ATT']


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class PlayerProfile(Contract):
    """Age at as_of_date and match totals in [as_of_date - 365 days, as_of_date)."""
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)

    age_years: Annotated[float, Field(ge=15, le=46)]
    height_cm: Annotated[float, Field(ge=140, le=220)] | None = None
    minutes_365: Annotated[int, Field(ge=config.MIN_MINUTES, le=9000)]
    appearances_365: Annotated[int, Field(ge=1, le=100)]
    goals_365: Annotated[int, Field(ge=0, le=100)]
    assists_365: Annotated[int, Field(ge=0, le=100)]
    yellow_365: Annotated[int, Field(ge=0, le=200)]
    red_365: Annotated[int, Field(ge=0, le=100)]
    league: League
    position_group: Position

    @model_validator(mode='after')
    def consistent_totals(self):
        if self.minutes_365 > self.appearances_365 * 130:
            raise ValueError('Minutes cannot exceed 130 per appearance, including extra time')
        if self.red_365 > self.appearances_365:
            raise ValueError('Red cards cannot exceed appearances')
        if self.yellow_365 > 2 * self.appearances_365:
            raise ValueError('Yellow cards cannot exceed twice the appearances')
        return self


class PredictionRequest(PlayerProfile):
    as_of_date: date = Field(description='YYYY-MM-DD; age is measured on this date, stats exclude this day.')

    @field_validator('as_of_date', mode='before')
    @classmethod
    def iso_date_only(cls, value):
        if type(value) is date:
            return value
        if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            raise ValueError('Use an ISO date (YYYY-MM-DD), without a time or timestamp')
        return date.fromisoformat(value)

    @field_validator('as_of_date')
    @classmethod
    def supported_date(cls, value):
        if not date.fromisoformat(config.FIRST_VALUATION_DATE) <= value <= datetime.now(timezone.utc).date():
            raise ValueError(f'as_of_date must be from {config.FIRST_VALUATION_DATE} through today (UTC)')
        return value


class PredictionRange(Contract):
    lower_eur: Annotated[int, Field(ge=0)]
    upper_eur: Annotated[int, Field(ge=0)]
    nominal_coverage: Annotated[float, Field(gt=0, lt=1)]


class ModelContext(Contract):
    version: str
    trained_through: date
    range_calibrated_through: date


class FeatureChange(Contract):
    feature: Literal['age_years', 'goals_365', 'assists_365', 'minutes_365', 'appearances_365']
    before: float
    after: float


class WhatIfFactor(Contract):
    factor: Literal['age', 'goals', 'assists', 'playing_time']
    changes: list[FeatureChange]
    estimate_eur: Annotated[int, Field(ge=0)]
    difference_eur: int
    direction: Literal['higher', 'lower', 'unchanged']


class Explanation(Contract):
    text: str
    source: Literal['fallback', 'groq']
    fallback_reason: Literal['disabled', 'not_configured', 'daily_limit', 'service_busy',
                            'provider_error', 'invalid_output', 'budget_unavailable',
                            'explanation_unavailable'] | None = None
    factors: list[WhatIfFactor] = Field(default_factory=list)


class PredictionResponse(Contract):
    target: Literal['Transfermarkt listed market value'] = 'Transfermarkt listed market value'
    currency: Literal['EUR'] = 'EUR'
    estimate_eur: Annotated[int, Field(ge=0)]
    prediction_range: PredictionRange
    as_of_date: date
    profile_source: Literal['submitted_profile', 'historical_lookup']
    model: ModelContext
    notes: list[str]
    explanation: Explanation | None = None


class PlayerSummary(Contract):
    player_id: Annotated[int, Field(gt=0)]
    name: str | None
    league: League
    position_group: Position
    age_years: float
    as_of_date: date


class SearchResponse(Contract):
    query: str
    total: int
    limit: int
    offset: int
    lookup_as_of: date
    profile_source: Literal['historical_lookup'] = 'historical_lookup'
    players: list[PlayerSummary]


class PlayerEstimateResponse(Contract):
    player: PlayerSummary
    profile: PredictionRequest
    prediction: PredictionResponse
    listed_value_eur: Annotated[int, Field(gt=0)]
    listing_minus_estimate_eur: int = Field(description='Positive means the historical listing exceeds the estimate.')


class HealthResponse(Contract):
    status: Literal['ok', 'unavailable']
    model_loaded: bool
    lookup_loaded: bool
    players: int
    model: ModelContext | None = None
    lookup_as_of: date | None = None


class ErrorDetail(Contract):
    field: str
    message: str
    type: str


class ErrorBody(Contract):
    code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)


class ErrorResponse(Contract):
    error: ErrorBody
