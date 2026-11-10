import copy,hashlib,json
from trace_stream import TraceReader
def normalized(event):
 args=event.get('args',{})
 if 'fag_direct' not in args and 'fag' not in args:
  if 'original-name' in args:raise ValueError('original-name without documented fag_direct annotation')
  return event,False
 key='fag_direct' if 'fag_direct' in args else 'fag'
 meta=args[key];original=args.get('original-name')
 if not isinstance(meta,dict) or not isinstance(original,str) or event.get('name')!=meta.get('label','')+' | '+original:raise ValueError('invalid documented FAG annotation')
 result=copy.deepcopy(event);result['name']=result['args'].pop('original-name');result['args'].pop(key)
 return result,True
def digest(path,filter_counters=False,inverse=False):
 reader=TraceReader(path);h=hashlib.sha256();events=annotations=0
 for event in reader:
  if filter_counters and event.get('ph')=='C':continue
  event,changed=normalized(event) if inverse else (event,False);annotations+=changed
  h.update((json.dumps(event,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode());events+=1
 h.update(json.dumps(reader.metadata,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())
 return {'canonical_sha256':h.hexdigest(),'events':events,'inverse_annotation_events':annotations,'metadata':reader.metadata}
