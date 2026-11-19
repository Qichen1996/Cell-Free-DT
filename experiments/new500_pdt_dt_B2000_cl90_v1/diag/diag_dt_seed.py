# Diagnostic (user: 能不能找出原因): rerun of eval_b2000_cl90.py evaluate() for one seed with per-50-step timing + UE count.
# Output sim_stats/new500_pdt_dt_B2000_cl90_v1_diag. Not a result run.
import sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[1]/"eval_b2000_cl90.py"
src=P.read_text().split("if __name__==")[0]
a="STATS=ROOT/'analysis/sim_stats/new500_pdt_dt_B2000_cl90_v1'";assert src.count(a)==1;src=src.replace(a,a[:-1]+"_diag'")
a="  assert bool(done)==(step==10079)\n";assert src.count(a)==1
src=src.replace(a,a+"  if step%50==0:\n   _n=getattr(env,'net',None) or env.env.net;print('STEP',step,round(time.time()-_T0,1),'ues',len(_n.ues),'sleep_frac',round(float((act[:,1]!=0).float().mean()),3),'a0',act[:,0].tolist(),'a1',act[:,1].tolist(),flush=True)\n")
a=" for step in range(10080):\n";assert src.count(a)==1;src=src.replace(a," _T0=time.time()\n"+a)
src="import time\n"+src
g={"__name__":"diag","__file__":str(P)};exec(compile(src,str(P),"exec"),g)
g["evaluate"]("DT","mean","B2000",int(sys.argv[1]))
