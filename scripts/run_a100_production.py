#!/usr/bin/env python3
"""Run the complete 300m-a100 FineWeb-Edu workflow on Colab."""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trashpanda.cli import _tokenizer_from_prepared
from trashpanda.config import get_recipe
from trashpanda.data import SequenceReader
from trashpanda.evaluate import evaluate, evaluation_record, sample_text
from trashpanda.export import export_hf
from trashpanda.train import train

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',type=Path,default=Path('/content/fineweb-run'))
 p.add_argument('--prepare',action='store_true')
 p.add_argument('--train-documents',type=int,default=30000)
 p.add_argument('--validation-documents',type=int,default=1000)
 p.add_argument('--max-steps',type=int)
 p.add_argument('--resume',type=Path, help='checkpoint to recover before performing a real resumed update')
 a=p.parse_args(); root=a.root.resolve(); prepared=root/'prepared'; config=get_recipe('300m-a100')
 if a.prepare:
  subprocess.run([sys.executable,str(ROOT/'scripts/prepare_fineweb_edu.py'),str(prepared),'--train-documents',str(a.train_documents),'--validation-documents',str(a.validation_documents),'--allow-download'],check=True)
 if not prepared.is_dir(): p.error(f'missing prepared data: {prepared}')
 tok=_tokenizer_from_prepared(prepared,config); train_data=SequenceReader(prepared,split='train',verify_hash=True); start=time.time()
 result=train(config,train_data,tokenizer=tok,checkpoint_dir=root/'checkpoints',resume=a.resume,max_steps=a.max_steps)
 validation_data=SequenceReader(prepared,split='validation',verify_hash=True)
 evaluation=evaluate(result.model,validation_data,batch_size=8,device='cuda')
 evaluation_provenance=evaluation_record(evaluation,split='validation',settings={'batch_size':8,'device':'cuda'},input_identity=validation_data.manifest.get('identity'))
 prompt='The future of education is'; samples=[{'seed':s,'prompt':prompt,'text':sample_text(result.model,tok,prompt,max_new_tokens=32,temperature=.8,top_k=50,seed=s,device='cuda')} for s in range(5)]
 export=root/'export'; export_hf(result.model.to('cpu'),export,config=config,tokenizer=tok,provenance={'records':[evaluation_provenance]})
 prepared_manifest=train_data.manifest
 source_records=prepared_manifest.get('provenance',{}).get('records',[])
 meta={'recipe':config.recipe_name,'dataset':source_records,'checkpoint':str(result.checkpoint),'checkpoint_step':result.state.update_step,'consumed_tokens':result.state.consumed_tokens,'resumed_from':str(a.resume) if a.resume else None,'parameter_count':config.parameter_count,'elapsed_seconds':time.time()-start,'evaluation':evaluation.__dict__,'evaluation_provenance':evaluation_provenance,'samples':samples,'export':str(export)}
 root.mkdir(parents=True,exist_ok=True); (root/'run-metadata.json').write_text(json.dumps(meta,sort_keys=True,indent=2)+'\n'); print(json.dumps(meta,sort_keys=True,indent=2))
if __name__=='__main__': main()
