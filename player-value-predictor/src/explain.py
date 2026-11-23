"""Grounded explanation facts and small, explicitly non-causal what-if checks.

The language service chooses among approved phrasings of computed facts. It
cannot introduce facts, recalculate numbers, or invent causal explanations.
"""
from collections import Counter
import json
import re

import pandas as pd

from schemas import Explanation, FeatureChange, PredictionRequest, WhatIfFactor

NUMBER = re.compile(r'\d+(?:,\d{3})*(?:\.\d+)?')


def money(value):
    return f'€{int(value):,}'


def scenarios(profile):
    """Local probes, not forecasts: keep all other submitted inputs fixed.

    Minute changes also change appearances to preserve a plausible profile.
    Totals remain fixed, so the shared feature transform recomputes per-90 rates.
    """
    raw = profile.model_dump()
    proposed = [
        ('age', {'age_years': profile.age_years + (1 if profile.age_years <= 45 else -1)}),
        ('goals', {'goals_365': profile.goals_365 + (1 if profile.goals_365 < 100 else -1)}),
        ('assists', {'assists_365': profile.assists_365 + (1 if profile.assists_365 < 100 else -1)}),
    ]
    if profile.minutes_365 <= 8910 and profile.appearances_365 < 100:
        proposed.append(('playing_time', {'minutes_365': profile.minutes_365 + 90,
                                          'appearances_365': profile.appearances_365 + 1}))
    valid = []
    for factor, changes in proposed:
        candidate = PredictionRequest(**{**raw, **changes})
        valid.append((factor, candidate, [FeatureChange(feature=field, before=raw[field], after=value)
                                          for field, value in changes.items()]))
    return valid


def compute_factors(profile, estimate_eur, predictor):
    probes = scenarios(profile)
    rows = pd.DataFrame([candidate.model_dump(exclude={'as_of_date'}) for _, candidate, _ in probes])
    predictions = predictor(rows)
    if len(predictions) != len(probes):
        raise ValueError('Wrong number of what-if predictions')
    factors = []
    for (factor, _, changes), predicted in zip(probes, predictions):
        value = round(float(predicted))
        difference = value - estimate_eur
        factors.append(WhatIfFactor(
            factor=factor, changes=changes, estimate_eur=value, difference_eur=difference,
            direction='higher' if difference > 0 else 'lower' if difference < 0 else 'unchanged',
        ))
    return sorted(factors, key=lambda item: (-abs(item.difference_eur), item.factor))[:3]


def factor_condition(factor):
    if factor.factor == 'playing_time':
        return '90 more minutes and 1 more appearance, with the recorded goals, assists and cards held fixed'
    change = factor.changes[0]
    more = change.after > change.before
    return {
        'age': 'an age 1 year older' if more else 'an age 1 year younger',
        'goals': '1 more goal' if more else '1 fewer goal',
        'assists': '1 more assist' if more else '1 fewer assist',
    }[factor.factor]


def make_facts(prediction, factors, listed_value=None):
    """Construct an allowlist from typed numerical outputs, never request text.

    No name, player ID, date of birth, date string, free text, or search query is
    accepted here. Each fact has two complete, truthful sentence variants.
    """
    estimate = money(prediction.estimate_eur)
    bounds = prediction.prediction_range
    lower, upper = money(bounds.lower_eur), money(bounds.upper_eur)
    coverage = f'{bounds.nominal_coverage * 100:g}%'
    facts = [
        {'id': 'estimate', 'variants': [
            f'The model estimates the Transfermarkt listed market value at {estimate}.',
            f'The estimated Transfermarkt listed market value is {estimate}.',
        ]},
        {'id': 'range', 'variants': [
            f'The prediction range is {lower} to {upper}, targeting {coverage} coverage across valuations.',
            f'The range runs from {lower} to {upper} and targets {coverage} coverage across valuations.',
        ]},
        {'id': 'range_limit', 'variants': [
            'This range is not a guarantee for this player and does not measure their true worth.',
            'The range does not guarantee this player’s value or establish their true worth.',
        ]},
    ]
    if listed_value is not None:
        difference = listed_value - prediction.estimate_eur
        relation = 'above' if difference > 0 else 'below'
        comparison = (f'{money(abs(difference))} {relation} the estimate' if difference else 'equal to the estimate')
        facts.append({'id': 'listing', 'variants': [
            f'The historical listing is {money(listed_value)}, {comparison}.',
            f'The recorded listing of {money(listed_value)} is {comparison}.',
        ]})
    for factor in factors:
        condition = factor_condition(factor)
        change = (f'{money(abs(factor.difference_eur))} {factor.direction}'
                  if factor.direction != 'unchanged' else 'unchanged after rounding')
        new_value = money(factor.estimate_eur)
        facts.append({'id': f'what_if_{factor.factor}', 'variants': [
            f'With {condition}, the model estimate would be {new_value}, {change}.',
            f'For a profile with {condition}, the model returns {new_value}, {change}.',
        ]})
    if factors:
        facts.append({'id': 'sensitivity_limit', 'variants': [
            'These comparisons change only the stated inputs; they are model sensitivities, not causal effects, forecasts or a complete explanation of the estimate.',
            'Only the stated inputs change in these comparisons; they show local model sensitivity, not causation, future value or a full attribution of the estimate.',
        ]})
    if prediction.profile_source == 'historical_lookup':
        facts.append({'id': 'historical', 'variants': [
            'This uses a saved historical profile, not current player data.',
            'The profile is historical and does not describe the player today.',
        ]})
    if prediction.as_of_date <= prediction.model.range_calibrated_through:
        facts.append({'id': 'retrospective', 'variants': [
            'The model or its range used data from at or after this profile date, so this is not an independent backtest.',
            'This is a retrospective demonstration, not an independent backtest, because model or range development reaches this profile date or later.',
        ]})
    else:
        facts.append({'id': 'frozen_model', 'variants': [
            'The model and its range are frozen and do not adjust for subsequent market changes.',
            'Later market changes are not incorporated into this frozen model and range.',
        ]})
    return {'facts': facts}


def fallback_explanation(facts, factors, reason):
    return Explanation(text=' '.join(fact['variants'][0] for fact in facts['facts']),
                       source='fallback', fallback_reason=reason, factors=factors)


def basic_fallback(prediction):
    """Emergency path if what-if/facts orchestration fails; numbers still return."""
    return fallback_explanation(make_facts(prediction, []), [], 'explanation_unavailable')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate response key')
        result[key] = value
    return result


def validate_wording(content, facts):
    """Reject changed numbers, swapped relations, omitted caveats, or new claims.

    Free paraphrases are intentionally not accepted: numeric checks alone cannot
    catch reversed comparisons or unsupported causal statements. Groq may select
    a wording variant for each fact; the final paragraph follows canonical order.
    """
    if not isinstance(content, str) or len(content) > 20000:
        raise ValueError('Invalid response length')
    result = json.loads(content, object_pairs_hook=_unique_object)
    if not isinstance(result, dict) or set(result) != {'sentences'} or not isinstance(result['sentences'], list):
        raise ValueError('Invalid response schema')
    expected = {fact['id']: fact['variants'] for fact in facts['facts']}
    selected = {}
    for item in result['sentences']:
        if not isinstance(item, dict) or set(item) != {'fact_id', 'text'}:
            raise ValueError('Invalid sentence schema')
        key, text = item['fact_id'], item['text']
        if not isinstance(key, str) or key not in expected or key in selected or not isinstance(text, str):
            raise ValueError('Unknown or duplicate fact')
        if Counter(NUMBER.findall(text)) != Counter(NUMBER.findall(expected[key][0])):
            raise ValueError('Numbers changed')
        if text not in expected[key]:
            raise ValueError('Unsupported wording or claim')
        selected[key] = text
    if set(selected) != set(expected):
        raise ValueError('Required facts omitted')
    return ' '.join(selected[key] for key in expected)
