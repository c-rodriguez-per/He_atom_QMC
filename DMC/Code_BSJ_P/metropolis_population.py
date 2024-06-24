import numpy as np
from hamiltonian import Hamiltonian
from wavefunction import JastrowWF,MultiplyWF




def drift_prob(poscur,posnew,gauss_move_cur,driftnew,tau):
    """ return the ratio of forward and backfward move probabilities for rejection algorith,
    Input:
      poscur: electron positions before move (nelec,ndim,nconf) 
      posnew: electron positions after  move (nelec,ndim,nconf)
      gauss_move_cur: randn() numbers for forward move
      driftnew: drift vector at posnew 
      tau: time step
    Return:
      ratio: [backward move prob.]/[forward move prob.]
      """
    # randn numbers needed for backward move
    gauss_move_new = (poscur-posnew-tau*driftnew)/np.sqrt(tau)
    # assume the following drift-diffusion move
    #assert np.allclose( poscur,posnew+np.sqrt(tau)*gauss_move_new+tau*driftnew ) 

    # calculate move probabilities
    gauss_cur_sq = np.sum( np.sum(gauss_move_cur**2.,axis=1) ,axis=0)
    gauss_new_sq = np.sum( np.sum(gauss_move_new**2.,axis=1), axis=0)
    forward_green  = np.exp(-gauss_cur_sq/2.)
    backward_green = np.exp(-gauss_new_sq/2.)

    ratio = backward_green/forward_green
    return ratio

def metropolis_sample(pos,wf,tau=0.01,nstep=1000):
  """
  Input variables:
    pos: a 3D numpy array with indices (electron,[x,y,z],configuration ) 
    wf: a Wavefunction object with value(), gradient(), and laplacian()
  Returns: 
    posnew: A 3D numpy array of configurations the same shape as pos, distributed according to psi^2
    acceptance ratio: 
  """
  df={}

  for i in ['tau','step','elocal', 'elocalvar', 'eref', 'emax', 'emin','r12min']:
    df[i]=[]
  # initialize
  ham=Hamiltonian(Z=2) # Helium
  posnew = pos.copy() # proposed positions
  poscur = pos.copy() # current positions
  wfold  = wf.value(poscur)
  acceptance=0.0
  nconf=pos.shape[2]
  acceptance_step = 0 #number of accepted moves in each 

  eref = -2.9
  # This loop performs the metropolis move many times to attempt to decorrelate the samples.
  for i in range (1): # small vmc to thermalize
    gauss_move_cur = np.random.randn(np.shape(pos)[0], np.shape(pos)[1], np.shape(pos)[2])
    posnew = poscur + np.sqrt(tau)*gauss_move_cur
    posnew += tau*wf.gradient(poscur)

    # calculate Metropolis-Rosenbluth-Teller acceptance probability
    #  VMC uses rejection to sample psi_sq by maintaining detailed balance
    a = wf.value(posnew)**2/wf.value(poscur)**2
    a *= drift_prob(poscur,posnew,gauss_move_cur,wf.gradient(posnew),tau)

    # get indices of accepted moves
    u = np.random.random_sample(nconf) 

    # update stale stored values for accepted configurations
    poscur[:,:,u<a] = posnew[:,:,u<a]

  print('Thermalized')
  for istep in range(nstep):
    nconf=poscur.shape[2]
    # propose a move
    gauss_move_cur = np.random.randn(np.shape(poscur)[0], np.shape(poscur)[1], np.shape(poscur)[2])
    posnew = poscur + np.sqrt(tau)*gauss_move_cur
    posnew += tau*wf.gradient(poscur)#*wf.value(poscur) 

    # calculate Metropolis-Rosenbluth-Teller acceptance probability
    a = wf.value(posnew)**2/wf.value(poscur)**2
    a *= drift_prob(poscur,posnew,gauss_move_cur,wf.gradient(posnew),tau)

    elocold = -0.5*np.sum(wf.laplacian(poscur),axis=0) + ham.pot_en(poscur) + ham.pot_ee(poscur)
    # get indices of accepted moves
    u = np.random.random_sample(nconf)  #acceptance ratio in this step
    acceptance += np.sum(u<a)/np.shape(poscur)[2]

    # update stale stored values for accepted configurations
    poscur[:,:,u<a] = posnew[:,:,u<a]

    #BRANCHING
    # The local energy.
    eloc = -0.5*np.sum(wf.laplacian(poscur),axis=0) + ham.pot_en(poscur) + ham.pot_ee(poscur)
    #w *= np.exp(-tau*(eloc-eref)) #OLD WEIGHT
    P = np.exp(-0.5*tau*(eloc+elocold-2*eref))
    eta = np.random.random(nconf)
    M = np.minimum((P+eta).astype(int), 2)
    eref -= np.log(np.mean(P)) 

    #
    posnew= np.repeat(poscur, M, axis = 2)
    poscur = posnew.copy()

    eloc = -0.5*np.sum(wf.laplacian(poscur),axis=0) + ham.pot_en(poscur) + ham.pot_ee(poscur)
    
    #Let us write to the dataframe
    df['tau'].append(tau)
    df['step'].append(istep)
    df['elocal'].append(np.mean(eloc))
    df['elocalvar'].append(np.var(eloc))
    df['eref'].append(eref)
    df['emax'].append(np.max(eloc))
    df['emin'].append(np.min(eloc))
    df['r12min'].append(np.min(np.sqrt(np.sum((poscur[0,:,:]-poscur[1,:,:])**2,axis=0))))
    #if np.linalg.norm(poscur[0,:,:]- np.tile(np.array([0.5,0,0]),poscur.shape[2]).reshape(3,poscur.shape[2]) ,axis=0).any()<0.5:
     # print('YES')
    #print(istep)
  acceptance = acceptance/(nstep)
  return poscur,acceptance, df







##########################################   Test

def test_metropolis(
      nconfig0=100000,
      ndim=3,
      nelec=2,
      nstep0=4000,
      tau0=0.01 
    ):
  # This part of the code will test your implementation. 
  # You can modify the parameters to see how they affect the results.
  from slaterwf import ExponentSlaterWF
  from hamiltonian import Hamiltonian
  import pandas as pd
  
  alpha = 2
  beta = 0
  wep = 2.56
  wee = 1.35
  fep = 0.777
  fee = 0.41
  wbf = 0.1
  fbf = -0.5
  
  wf=MultiplyWF(ExponentSlaterWF(alpha),JastrowWF(beta,wep,wee,fep,fee,wbf,fbf))
  ham=Hamiltonian(Z=2)
  dfs = []
  for tau in [0.01, 0.005, 0.0025, 0.0015]:#, 0.001]:
    nstep = int(nstep0*int(tau0/tau))
    nconfig = int(nconfig0*(tau0/tau))
    possample     = np.random.randn(nelec,ndim,nconfig)
    possample,acc,df = metropolis_sample(possample,wf,tau=tau,nstep=nstep)
    print(acc)
    print(possample.shape[2])
    dfs.append(pd.DataFrame(df))
  
  df = pd.concat(dfs)
  pd.DataFrame(df).to_csv("dmc_try.csv",index=False)

#Let us time the code
import time
start_time = time.time()

if __name__=="__main__":
  test_metropolis()
  
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
