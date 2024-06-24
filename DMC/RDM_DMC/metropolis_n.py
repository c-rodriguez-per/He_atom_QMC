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

def metropolis_sample(pos,wf,tau=0.01,nstep=1000,M = 10,block_length = 100):
  """
  Input variables:
    pos: a 3D numpy array with indices (electron,[x,y,z],configuration ) 
    wf: a Wavefunction object with value(), gradient(), and laplacian()
  Returns: 
    posnew: A 3D numpy array of configurations the same shape as pos, distributed according to psi^2
    acceptance ratio: 
  """
  df={}

  for i in ['tau','step','elocal', 'elocalvar', 'eref']:
    df[i]=[]
  
  #DENSITY MATRIX
  Nblocks = int(nstep/block_length)
  rho_acc = np.zeros((Nblocks,M,M,M))
  x = np.linspace(-3,3,M)
  y = np.linspace(-3,3,M)
  z = np.linspace(-3,3,M)
  r_av = 0 #0.8936
  r_closest_index = np.argmin(np.abs(x - r_av))
  dV = (6/(M-1))**3

  # initialize
  ham=Hamiltonian(Z=2) # Helium
  posnew = pos.copy() # proposed positions
  poscur = pos.copy() # current positions
  acceptance=0.0
  nconf=pos.shape[2]

  eref = -2.9
  # This loop performs the metropolis move many times to attempt to decorrelate the samples.
  for i in range (50): # small vmc to thermalize
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

  for j in range(Nblocks):
    for istep in range(block_length):
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

      #Density matrix calculations
      #for r1
      r1 = poscur[0,:,:]
      r2 = poscur[1,:,:]
      x1_index = np.argmin(np.abs(r1[0,:] - x[:, np.newaxis]),axis = 0)
      y1_index = np.argmin(np.abs(r1[1,:] - y[:, np.newaxis]),axis = 0)
      z1_index = np.argmin(np.abs(r1[2,:] - z[:, np.newaxis]),axis = 0)
      indices = np.column_stack((x1_index, y1_index, z1_index))
      unique_indices, counts = np.unique(indices, axis=0, return_counts=True)
      rho_acc[j,unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]] += counts/poscur.shape[2]

      #for r2
      x2_index = np.argmin(np.abs(r2[0,:] - x[:, np.newaxis]),axis = 0)
      y2_index = np.argmin(np.abs(r2[1,:] - y[:, np.newaxis]),axis = 0)
      z2_index = np.argmin(np.abs(r2[2,:] - z[:, np.newaxis]),axis = 0)
      indices = np.column_stack((x2_index, y2_index, z2_index))
      unique_indices, counts = np.unique(indices, axis=0, return_counts=True)
      rho_acc[j,unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]] += counts/poscur.shape[2]

  acceptance = acceptance/(nstep)
  rho = rho_acc/(block_length*dV) #multiply by volume
  #print('This should be #e',np.sum(np.average(rho,axis=0)))
  return poscur,acceptance, df, rho







##########################################   Test

def test_metropolis(
      nconfig0=10000, #10k for normal 50k bigold #Normal unos 8minutos
      ndim=3,
      nelec=2,
      nstep0=5000, #5k for normal
      tau0=0.01,
      M=51
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
  for tau in [0.01]:#, 0.005, 0.0025, 0.0015]:#, 0.001]:
    nstep = int(nstep0*int(tau0/tau))
    nconfig = int(nconfig0*(tau0/tau))
    possample     = np.random.randn(nelec,ndim,nconfig)
    possample,acc,df,density = metropolis_sample(possample,wf,tau=tau,nstep=nstep,M=M,block_length=100)
    print('Acceptance: ', acc)
    print('Final #walkers ',possample.shape[2])
    print('#Blocks: ', density.shape[0])
    dfs.append(pd.DataFrame(df))
  
  np.save('n_d', density) #to open s = np.load(open('n(r).npy','rb'))
  df = pd.concat(dfs)
  pd.DataFrame(df).to_csv("dmc.csv",index=False)

#Let us time the code
import time
start_time = time.time()

if __name__=="__main__":
  test_metropolis()
  
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
