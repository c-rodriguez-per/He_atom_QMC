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








def metropolis_sample(pos,wf,tau=0.01,nstep=1000, M = 11, block_length = 100):
  """
  Input variables:
    pos: a 3D numpy array with indices (electron,[x,y,z],configuration ) 
    wf: a Wavefunction object with value(), gradient(), and laplacian()
  Returns: 
    posnew: A 3D numpy array of configurations the same shape as pos, distributed according to psi^2
    acceptance ratio: 
  """
  df={}
  for i in ['E', 'E_var','r_av']:
    df[i]=[]
  #DENSITY MATRIX(r,r'=(r_av,0,0))
  Nblocks = int(nstep/block_length)
  rho_acc = np.ones((Nblocks,M,M,M,M,M,M), dtype=float)
  L = 2
  fs = 1/(2*L)**3
  x = np.linspace(-L,L,M)
  y = np.linspace(-L,L,M)
  z = np.linspace(-L,L,M)



  # initialize
  ham=Hamiltonian(Z=2)
  posnew = pos.copy() # proposed positions
  poscur = pos.copy() # current positions
  wfold  = wf.value(poscur)
  acceptance=0.0
  nconf=pos.shape[2]

  #METROPOLIS TEST
  #LET US THERMALIZE
  for istep in range(500):
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

  # This loop performs the metropolis move many times to attempt to decorrelate the samples.
  for j in range(Nblocks):
    for istep in range(block_length):
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
      df['E'].append(np.average(eloc))
      df['E_var'].append(np.var(eloc))
      df['r_av'].append(np.mean(np.sqrt(np.sum(poscur**2, axis = 1))))
      #DENSITY MATRIX
      #Let us first calculate the relevat n parameters for s
      s = (-2*L)*np.random.random_sample([3,nconf])+L
      xs_index = np.argmin(np.abs(s[0,:] - x[:, np.newaxis]),axis = 0)
      ys_index = np.argmin(np.abs(s[1,:] - y[:, np.newaxis]),axis = 0)
      zs_index = np.argmin(np.abs(s[2,:] - z[:, np.newaxis]),axis = 0)
      

      #for r1
      r1 = poscur[0,:,:]
      r2 = poscur[1,:,:]
      x1_index = np.argmin(np.abs(r1[0,:] - x[:, np.newaxis]),axis = 0)
      y1_index = np.argmin(np.abs(r1[1,:] - y[:, np.newaxis]),axis = 0)
      z1_index = np.argmin(np.abs(r1[2,:] - z[:, np.newaxis]),axis = 0)
      for i in range(len(x1_index)):
        rho_acc[j,x1_index[i],y1_index[i],z1_index[i],xs_index[i],ys_index[i],zs_index[i]] += wf.value(np.stack([s[:,i],r2[:,i]],axis = 0))/(wf.value(np.stack([r1[:,i], r2[:,i]],axis = 0))*fs)
        
      #rho_acc[x1_index,y1_index,z1_index] += wf.value(np.stack([s_new, r2],axis = 0))/wf.value(poscur)

      #for r2
      x2_index = np.argmin(np.abs(r2[0,:] - x[:, np.newaxis]),axis = 0)
      y2_index = np.argmin(np.abs(r2[1,:] - y[:, np.newaxis]),axis = 0)
      z2_index = np.argmin(np.abs(r2[2,:] - z[:, np.newaxis]),axis = 0)
      for i in range(len(x2_index)):
        rho_acc[j,x2_index[i],y2_index[i],z2_index[i],xs_index[i],ys_index[i],zs_index[i]] += wf.value(np.stack([r1[:,i], s[:,i]],axis = 0))/(wf.value(np.stack([r1[:,i], r2[:,i]],axis = 0))*fs)

  #print(pos_full[:,0,0,0])
  #print(pos_full.shape)
  rho = rho_acc/(block_length*nconf)
  return poscur,acceptance, df, rho


##########################################   Test

def test_metropolis(
      nconfig=50000, #10k for normal 50k big
      ndim=3,
      nelec=2,
      nstep=1000, #2k for normal, 5kfor big
      tau=0.35, #at 0.1 we achieve acceptance=0.55
      M = 11
    ):
  alpha = 2
  beta = 0
  wep = 2.56
  wee = 1.35
  fep = 0.777
  fee = 0.41
  wbf = 0.1
  fbf = -0.5
  #DATAFRAME TO STORE RESULTS
  

  wf=MultiplyWF(ExponentSlaterWF(alpha),JastrowWF(beta,wep,wee,fep,fee,wbf,fbf))
  possample     = np.random.randn(nelec,ndim,nconfig)
  possample,acc,df,rho = metropolis_sample(possample,wf,tau=tau,nstep=nstep,M=M)

  print(acc)
  np.save('rho', rho)
  import pandas as pd
  pd.DataFrame(df).to_csv("helium_one.csv",index=False)

if __name__=="__main__":
  test_metropolis()
  
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
