import asyncio, json, os, uuid
from pathlib import Path
import pandas as pd
from agent.facts import BrandFacts
from agent.llm import Agent
from agent.evaluate import evaluate
from agent.db import init_db, save_run

ROOT=Path(__file__).resolve().parent
async def main():
    split=os.getenv('SPLIT','development'); version=os.getenv('VERSION','v1')
    answers=pd.read_csv(ROOT/'data/answers.csv').fillna(''); subset=answers[answers.split==split]
    facts=BrandFacts(ROOT/'data/brand_facts.csv'); agent=Agent(facts)
    async def one(row):
        pred=await agent.predict(row,version)
        print(f"{row.answer_id}: {len(pred.claims)} claims",flush=True)
        return pred.model_dump()
    # Agent semaphore bounds concurrent requests; gather all rows for efficient async execution.
    preds=await asyncio.gather(*(one(row) for row in subset.itertuples(index=False)))
    metrics=evaluate(preds,ROOT/'data/development_gold.csv',ROOT/'data/answers.csv') if split=='development' else {}
    run_id=f'{version}-{split}-{uuid.uuid4().hex[:8]}'
    out=ROOT/'artifacts'; out.mkdir(exist_ok=True)
    artifact={'run_id':run_id,'version':version,'split':split,'model':agent.model,'metrics':metrics,'predictions':preds}
    path=out/f'{run_id}.json'; path.write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf8')
    init_db(); save_run(run_id,version,split,agent.model,metrics,subset.to_dict('records'),preds)
    print(json.dumps({'run_id':run_id,'artifact':str(path),'metrics':metrics},indent=2))
if __name__=='__main__': asyncio.run(main())
