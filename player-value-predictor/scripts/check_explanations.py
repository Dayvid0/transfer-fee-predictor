"""Mocked wording/faithfulness review using synthetic profiles and frozen model.

Run from root: .venv/bin/python scripts/check_explanations.py --output report.json
No API key, outbound requests, training-table access or model evaluation. This
is not a live Groq validation. Each sample includes exact model-computed facts,
an accepted wording, and adversarial responses that must be rejected.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import api
from explain import compute_factors, make_facts, validate_wording
from llm import ExplanationEngine
from schemas import PredictionRequest
from train import ARTIFACT_DIR, predict_values


def wording(facts, variant):
    return json.dumps({'sentences':[{'fact_id':fact['id'],'text':fact['variants'][variant]}
                                   for fact in facts['facts']]}, ensure_ascii=False)


def review(output, artifact_dir=ARTIFACT_DIR):
    service=api.PredictionService.load(artifact_dir)
    profiles=[
        ('young_attacker', {'age_years':19,'position_group':'ATT','goals_365':18}),
        ('older_keeper', {'age_years':39,'position_group':'GK','goals_365':0,'assists_365':0}),
        ('missing_height', {'height_cm':None,'position_group':'DEF','goals_365':1}),
        ('minimum_minutes', {'minutes_365':450,'appearances_365':5,'goals_365':1,'red_365':0}),
        ('midfielder', {'position_group':'MID','assists_365':14,'league':'ES1'}),
        ('bounded_inputs', {'age_years':46,'goals_365':100,'assists_365':100,
                            'minutes_365':9000,'appearances_365':100}),
    ]
    results=[]
    for label,changes in profiles:
        profile=PredictionRequest(**{
            'as_of_date':'2025-06-15','age_years':24,'height_cm':180,'minutes_365':1800,
            'appearances_365':24,'goals_365':8,'assists_365':5,'yellow_365':3,'red_365':1,
            'league':'GB1','position_group':'ATT',**changes})
        prediction=service.predict(profile)
        factors=compute_factors(profile,prediction.estimate_eur,lambda rows:predict_values(service.bundle['model'],rows))
        facts=make_facts(prediction,factors)
        accepted=[validate_wording(wording(facts,v),facts) for v in (0,1)]
        malicious=[]
        for change in ('extra_claim','changed_number','omitted_caveat','reversed_direction'):
            body=json.loads(wording(facts,0))
            if change=='extra_claim': body['sentences'][0]['text']+=' An injury caused the lower value.'
            elif change=='changed_number': body['sentences'][0]['text']='The estimated value is €999,999,999.'
            elif change=='omitted_caveat': body['sentences']=[r for r in body['sentences'] if r['fact_id']!='range_limit']
            else:
                item=next(r for r in body['sentences'] if r['fact_id'].startswith('what_if_'))
                if 'higher' in item['text']: item['text']=item['text'].replace('higher','lower')
                elif 'lower' in item['text']: item['text']=item['text'].replace('lower','higher')
                else: item['text']=item['text'].replace('unchanged after rounding','higher')
            try:
                validate_wording(json.dumps(body),facts)
                raise AssertionError(f'{change} was accepted')
            except ValueError:
                malicious.append(change)
        result=ExplanationEngine().explain(facts,factors)
        results.append({'profile':label,'input':profile.model_dump(mode='json'),
                        'prediction':prediction.model_dump(mode='json'),
                        'factors':[f.model_dump() for f in factors],
                        'accepted_wordings':accepted,'rejected_mutations':malicious,
                        'fallback_text':result.text})
    report={'mode':'mocked_only','live_groq_tested':False,'profiles':len(results),
            'valid_wordings_accepted':2*len(results),'adversarial_outputs_rejected':4*len(results),
            'samples':results}
    path=Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"Checked {len(results)} profiles: {2*len(results)} valid wordings accepted, {4*len(results)} adversarial responses rejected; no live Groq calls.")
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,default=ARTIFACT_DIR)
    args=parser.parse_args()
    review(args.output,args.artifacts)
