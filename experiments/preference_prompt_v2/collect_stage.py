from workflow import *
if __name__=='__main__':
 validate_sources()
 space()
 save_new(BASE/'collection-plan.json',dict(teachers=[2.4,2.7,3.0],seeds=[1,2,3,4,5],steps=10080,source_hashes=EXPECTED))
 parallel_jobs([(['collect',str(pc)],f'collect_pc_{pc:.1f}') for pc in [2.4,2.7,3.0]],3)
 save_new(BASE/'collection-complete.json',dict(teachers=[2.4,2.7,3.0],episodes=15))
