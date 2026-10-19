import os, json
from sqlalchemy import create_engine, text

def engine():
    url=os.getenv('DATABASE_URL')
    if not url: return None
    return create_engine(url, pool_pre_ping=True)

def init_db():
    e=engine()
    if not e: return False
    with e.begin() as c:
        c.execute(text('''CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, version TEXT NOT NULL, split TEXT NOT NULL, model TEXT, started_at TIMESTAMPTZ DEFAULT now(), metrics JSONB)'''))
        c.execute(text('''CREATE TABLE IF NOT EXISTS answers (answer_id TEXT PRIMARY KEY, split TEXT NOT NULL, brand_id TEXT NOT NULL, question TEXT, answer_text TEXT NOT NULL)'''))
        c.execute(text('''CREATE TABLE IF NOT EXISTS predictions (id BIGSERIAL PRIMARY KEY, run_id TEXT REFERENCES runs(run_id), answer_id TEXT REFERENCES answers(answer_id), version TEXT NOT NULL, claims JSONB NOT NULL, created_at TIMESTAMPTZ DEFAULT now())'''))
    return True

def save_run(run_id, version, split, model, metrics, answers, predictions):
    e=engine()
    if not e: return False
    with e.begin() as c:
        c.execute(text('INSERT INTO runs(run_id,version,split,model,metrics) VALUES(:r,:v,:s,:m,:x) ON CONFLICT(run_id) DO UPDATE SET metrics=EXCLUDED.metrics'), {'r':run_id,'v':version,'s':split,'m':model,'x':json.dumps(metrics)})
        for a in answers:
            c.execute(text('INSERT INTO answers(answer_id,split,brand_id,question,answer_text) VALUES(:id,:s,:b,:q,:t) ON CONFLICT(answer_id) DO UPDATE SET answer_text=EXCLUDED.answer_text'), {'id':str(a['answer_id']),'s':str(a['split']),'b':str(a['brand_id']),'q':str(a['question']),'t':str(a['ai_answer'])})
        for p in predictions:
            c.execute(text('INSERT INTO predictions(run_id,answer_id,version,claims) VALUES(:r,:a,:v,:c)'), {'r':run_id,'a':p['answer_id'],'v':version,'c':json.dumps(p['claims'])})
    return True
