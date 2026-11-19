from pathlib import Path
import json,hashlib,shutil
R=Path('/workspace/Cell-Free-DT-new');name='B_pc18_pc21_seeds45';E=R/'experiments'/f'{name}_eval';rows=[]
for pc in [1.8,2.1]:
 for ts in [4,5]:
  training=R/'results/weights'/name/f'seed{ts}/pc_{pc}'
  done=json.loads((training/'complete.json').read_text());assert done['steps']==2520000
  meta=json.loads((training/'experiment.json').read_text());assert meta['seed']==ts and meta['weights']==[80,.2,.2,pc] and meta['lr_schedule']['type']=='constant'
  files=Path(done['actor_path']).parent
  for cp in [180,200,220,230,240]:
   src=E/f'models/pc{pc}_trainseed{ts}/eps{cp}';src.mkdir(parents=True)
   for kind in ['actor','critic']:shutil.copy2(files/f'{kind}_eps{cp}.pt',src/f'{kind}.pt')
   shutil.copy2(R/f'results/weights/B1000_checkpoint_test/by_tag/eps200/pc_{pc}/architecture.json',src/'architecture.json')
   m={f'{k}.pt':hashlib.sha256((src/f'{k}.pt').read_bytes()).hexdigest() for k in ['actor','critic']}
   (src/'manifest.json').write_text(json.dumps(m,indent=2))
   (src/'provenance.json').write_text(json.dumps(dict(training_seed=ts,checkpoint=cp,training_dir=str(training),source_files=str(files),experiment=meta,complete=done),indent=2))
   rows.append(dict(pc=pc,training_seed=ts,checkpoint=cp,source=str(src),tag=f'pc{pc}_trainseed{ts}_cp{cp}'))
(E/'specs.json').write_text(json.dumps(rows,indent=2))
