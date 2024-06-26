import pandas as pd
import numpy as np

def reblock(eloc,warmup,nblocks):
    index = int(len(eloc)*warmup)
    elocblock=np.array_split(eloc[index:],nblocks)
    print(tau,len(elocblock))
    blockenergy=[np.mean(x) for x in elocblock]
    return np.mean(blockenergy),np.std(blockenergy)/np.sqrt(nblocks)

warmup=0.25 #percent part of array to discard
blocksize=1.0 # in Hartree-1

df=pd.read_csv("dmc.csv")
dfreblock=[]
for tau,grp in df.groupby("tau"):
    blocktau=blocksize/tau
    eloc=grp.sort_values('step')['elocal'].values
    nblocks=int((len(eloc)*(1-warmup))/blocktau)
    avg,err=reblock(eloc,warmup,nblocks)
    dfreblock.append({'tau':tau,
        'eavg':avg,
        'err':err})

pd.DataFrame(dfreblock).to_csv("dmc_reblocked.csv")


