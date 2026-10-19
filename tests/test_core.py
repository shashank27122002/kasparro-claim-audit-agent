from pathlib import Path
import pandas as pd
from agent.facts import BrandFacts
from agent.evaluate import evaluate
ROOT=Path(__file__).resolve().parents[1]
def test_csv_shape_and_splits():
    a=pd.read_csv(ROOT/'data/answers.csv')
    assert len(a)==50 and (a.split=='development').sum()==40 and (a.split=='check').sum()==10
def test_gold_spans_are_exact():
    a=pd.read_csv(ROOT/'data/answers.csv').set_index('answer_id')
    g=pd.read_csv(ROOT/'data/development_gold.csv')
    for r in g.itertuples():
        s=a.loc[r.answer_id,'ai_answer']
        assert s[int(r.start_char):int(r.end_char)]==r.quote, r.claim_id
def test_fact_lookup():
    f=BrandFacts(ROOT/'data/brand_facts.csv')
    rows=f.get_brand_facts('cleartrail','warranty','2026-10-01')
    assert rows and all(x['brand_id']=='cleartrail' for x in rows)
def test_extraction_metric_penalizes_missing_and_extra():
    # This deliberately incomplete prediction should incur one FN and one FP.
    g=pd.read_csv(ROOT/'data/development_gold.csv')
    a=pd.read_csv(ROOT/'data/answers.csv')
    row=g.iloc[0]
    p=[{'answer_id':row.answer_id,'claims':[{'start_char':999999,'end_char':1000000,'quote':'invented','label':'unknown'}]}]
    m=evaluate(p,ROOT/'data/development_gold.csv',ROOT/'data/answers.csv')
    assert m['extraction']['fp']>=1 and m['extraction']['fn']>=1
