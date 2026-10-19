import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
LABELS=['supported','contradicted','unknown']

def evaluate(predictions: list[dict], gold_csv: str, answers_csv: str):
    gold=pd.read_csv(gold_csv).fillna(''); answers=pd.read_csv(answers_csv).fillna('')
    answer_text=dict(zip(answers.answer_id.astype(str),answers.ai_answer.astype(str)))
    gmap={str(a):grp.to_dict('records') for a,grp in gold.groupby('answer_id')}
    tp=fp=fn=0; y_true=[]; y_pred=[]; span_details=[]
    for p in predictions:
        aid=str(p['answer_id']); gt=gmap.get(aid,[]); pred=p.get('claims',[])
        # Exact-span matching; an invented/misquoted claim is FP, missed gold span is FN.
        g_by={(int(x['start_char']),int(x['end_char']),str(x['quote'])):x for x in gt}
        p_by={(int(x['start_char']),int(x['end_char']),str(x['quote'])):x for x in pred}
        matched=set(g_by)&set(p_by)
        tp += len(matched); fp += len(set(p_by)-set(g_by)); fn += len(set(g_by)-set(p_by))
        for key in matched:
            y_true.append(str(g_by[key]['expected_label'])); y_pred.append(str(p_by[key].get('label','unknown')))
        span_details.append({'answer_id':aid,'gold_claims':len(gt),'pred_claims':len(pred),'exact_matches':len(matched),'missed':len(set(g_by)-set(p_by)),'extra':len(set(p_by)-set(g_by))})
    ep=tp/(tp+fp) if tp+fp else 0; er=tp/(tp+fn) if tp+fn else 0
    ef=2*ep*er/(ep+er) if ep+er else 0
    # Joint score gives extraction errors consequences: missed/extra spans count against extraction, classification only on exact matches.
    return {'extraction':{'tp':tp,'fp':fp,'fn':fn,'precision':ep,'recall':er,'f1':ef},'classification_on_exact_spans':{'matched_claims':len(y_true),'accuracy':accuracy_score(y_true,y_pred) if y_true else 0,'macro_f1':f1_score(y_true,y_pred,labels=LABELS,average='macro',zero_division=0) if y_true else 0,'per_class_f1':{lab:f1_score(y_true,y_pred,labels=[lab],average='macro',zero_division=0) if y_true else 0 for lab in LABELS}},'per_answer':span_details}
