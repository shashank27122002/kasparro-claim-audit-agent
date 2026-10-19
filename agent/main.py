import os, json, uuid, asyncio
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .facts import BrandFacts
from .llm import Agent
from .evaluate import evaluate
from .db import init_db, save_run

ROOT=Path(__file__).resolve().parents[1]
ANS=os.getenv('ANSWERS_CSV',str(ROOT/'data/answers.csv')); FACTS=os.getenv('BRAND_FACTS_CSV',str(ROOT/'data/brand_facts.csv')); GOLD=os.getenv('GOLD_CSV',str(ROOT/'data/development_gold.csv'))
app=FastAPI(title='KASPARRO Claim Audit Agent',version='1.0.0')
facts=BrandFacts(FACTS); agent=Agent(facts)

class RunRequest(BaseModel):
    version: str='v1'
    split: str='development'
    limit: int|None=None

@app.on_event('startup')
def startup(): init_db()

@app.get('/health')
def health(): return {'status':'ok','model':agent.model,'database_configured':bool(os.getenv('DATABASE_URL'))}

@app.post('/run')
async def run(req: RunRequest):
    if req.version not in ('v1','v2'): raise HTTPException(400,'version must be v1 or v2')
    df=pd.read_csv(ANS).fillna(''); df=df[df.split==req.split]
    if req.limit: df=df.head(req.limit)
    if df.empty: raise HTTPException(404,'No answers found for split')
    results=await asyncio.gather(*(agent.predict(r,req.version) for r in df.itertuples(index=False)))
    preds=[p.model_dump() for p in results]
    metrics=evaluate(preds,GOLD,ANS) if req.split=='development' else {}
    run_id=f'{req.version}-{req.split}-{uuid.uuid4().hex[:10]}'
    save_run(run_id,req.version,req.split,agent.model,metrics,df.to_dict('records'),preds)
    out=ROOT/'artifacts'; out.mkdir(exist_ok=True)
    with open(out/f'{run_id}.json','w',encoding='utf8') as f: json.dump({'run_id':run_id,'version':req.version,'split':req.split,'model':agent.model,'metrics':metrics,'predictions':preds},f,ensure_ascii=False,indent=2)
    return {'run_id':run_id,'version':req.version,'split':req.split,'count':len(preds),'metrics':metrics,'artifact':f'artifacts/{run_id}.json'}
