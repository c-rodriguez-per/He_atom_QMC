import numpy as np
from hamiltonian import Hamiltonian
from wavefunction import JastrowWF,MultiplyWF
from slaterwf import ExponentSlaterWF

#Let us time the code
import time
start_time = time.time()

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

def metropolis_sample(pos,wf,tau=0.01,nstep=1000, warmup = 0.3):
  """
  Input variables:
    pos: a 3D numpy array with indices (electron,[x,y,z],configuration ) 
    wf: a Wavefunction object with value(), gradient(), and laplacian()
  Returns: 
    posnew: A 3D numpy array of configurations the same shape as pos, distributed according to psi^2
    acceptance ratio: 
  """
  df={}

  for i in ['tau','step','elocal', 'weight','elocalvar','weightvar', 'eref']:
    df[i]=[]

  # initialize
  ham=Hamiltonian(Z=2)
  posnew = pos.copy() # proposed positions
  poscur = pos.copy() # current positions
  wfold  = wf.value(poscur)
  acceptance=0.0
  nconf=pos.shape[2]
  acceptance_step = 0 #number of accepted moves in each 

  #METROPOLIS TEST
  E = np.zeros(nstep)

  # This loop performs the metropolis move many times to attempt to decorrelate the samples.
  for istep in range(nstep):
    # propose a move
    gauss_move_old = np.random.randn(*poscur.shape)
    posnew=poscur+np.sqrt(tau)*gauss_move_old + tau*wf.gradient(poscur)

    wfnew=wf.value(posnew)

    # calculate Metropolis-Rosenbluth-Teller acceptance probability
    prob = wfnew**2/wfold**2 # for reversible moves
    prob *= drift_prob(poscur,posnew,gauss_move_old,wf.gradient(posnew),tau)

    # get indices of accepted moves
    acc_idx = (prob + np.random.random_sample(nconf) > 1.0)

    # update stale stored values for accepted configurations
    poscur[:,:,acc_idx] = posnew[:,:,acc_idx]
    wfold[acc_idx] = wfnew[acc_idx]
    acceptance += np.mean(acc_idx)/nstep
    #METROPOLIS TEST
    #LET US FILL THE DATAFRAME
    eloc = -0.5*np.sum(wf.laplacian(poscur),axis=0) + ham.pot_en(poscur) + ham.pot_ee(poscur)
    E[istep] = np.average(eloc)


  #print(pos_full[:,0,0,0])
  #print(pos_full.shape)
  E_fin = np.mean(E[int(warmup*len(E)):])
  E_fin_errorbar = np.sqrt(np.var(E[int(warmup*len(E)):])/len(E[int(warmup*len(E)):]))
  return poscur,acceptance, E_fin, E_fin_errorbar


##########################################   Test

def test_metropolis(
      nconfig=1000,
      ndim=3,
      nelec=2,
      nstep=1000,
      tau=0.25, #at 0.1 we achieve acceptance=0.55
      warmup = 0.4
    ):
  # This part of the code will test your implementation. 
  # You can modify the parameters to see how they affect the results.
  N_points = 5
  alpha_trial = np.linspace(1.7,2.1,N_points)
  beta_trial = np.linspace(0.1,0.7,N_points)
  wep_trial = np.linspace(4.5,5.5,N_points)
  wee_trial = np.linspace(0.5,1.5,N_points)
  fep_trial = np.linspace(1,2.5,N_points)
  fee_trial = np.linspace(0.1,1.2,N_points)

  wbf_trial = np.linspace(0.1,0.6,N_points)
  fbf_trial = np.linspace(0.05,1,N_points)
  ham=Hamiltonian(Z=2)
  #DATAFRAME TO STORE RESULTS
  df={}
  for i in ['alpha','beta','w_ep','w_ee','f_ep','f_ee','acceptance', 'E', 'E_errbar', 'w_bf','f_bf']:
    df[i]=[]
  alpha = 2#FIXED CUSP
  beta = 0
  wep = 5.25 #OPTIMISED
  wee = 1.35
  fep = 2.125 #OPTIMISED
  fee = 0.41
  wbf = 0.1
  fbf = -0.25 #FIXED CUSP
  #To go back just set fee=fep=0.5
  #for alpha in alpha_trial:  #for beta in beta_trial:  #for wep in wep_trial:  #for wee in wee_trial:  #for fep in fep_trial:  #for fee in fee_trial:
  for wee in wee_trial:
    for fee in fee_trial:
      wf=MultiplyWF(ExponentSlaterWF(alpha),JastrowWF(beta,wep,wee,fep,fee,wbf,fbf))
      possample     = np.random.randn(nelec,ndim,nconfig)
      possample,acc,Efin,Efin_errbar = metropolis_sample(possample,wf,tau=tau,nstep=nstep,warmup=warmup)
      df['alpha'].append(alpha)
      df['beta'].append(beta)
      df['w_ep'].append(wep)
      df['w_ee'].append(wee)
      df['f_ep'].append(fep)
      df['f_ee'].append(fee)
      df['acceptance'].append(acc)
      df['E'].append(Efin)
      df['E_errbar'].append(Efin_errbar)
      df['w_bf'].append(wbf)
      df['f_bf'].append(fbf)

  import pandas as pd
  pd.DataFrame(df).to_csv("helium.csv",index=False)

if __name__=="__main__":
  test_metropolis()
  
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
