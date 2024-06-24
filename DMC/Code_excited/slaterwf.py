import numpy as np


class ExponentSlaterWF:
  """ 
  Slater determinant specialized to one up and one down electron, each with
  exponential orbitals.
  Member variables:
    alpha: decay parameter.

  Note:
  pos is an array such that
    pos[i][j][k]
  will return the j-th component of the i-th electron for the 
  k-th sample (or "walker").
  """
  def __init__(self,alpha=1):
    self.alpha=alpha
#-------------------------
  def value(self,pos):
    #print(pos[1,:,1])    all positions of electron 1 in walker 1
    dist = np.sqrt(np.sum(pos**2, axis = 1))
    r1 = dist[0,:]
    r2 = dist[1,:]
    phi12 = (2-self.alpha*r2)*np.exp(-self.alpha*(r1+r2/2))
    phi21 = (2-self.alpha*r1)*np.exp(-self.alpha*(r2+r1/2))
    #Returns WF for each walker.
    return phi12-phi21
#-------------------------
  def gradient(self,pos):
    #Calculates normalized gradient
    dist=np.sqrt(np.sum(pos**2,axis=1))
    unit_vector = pos/dist[:,np.newaxis,:]
    r1 = dist[0,:]
    r2 = dist[1,:]
    phi12 = (2-self.alpha*r2)*np.exp(-self.alpha*(r1+r2/2))
    phi21 = (2-self.alpha*r1)*np.exp(-self.alpha*(r2+r1/2))
    grad1 = self.alpha*(-phi12+phi21/2+np.exp(-self.alpha*(r2+r1/2)))*unit_vector[0,:,:]
    grad2 = self.alpha*(-phi21+phi12/2+np.exp(-self.alpha*(r1+r2/2)))*unit_vector[1,:,:]

    #print(np.array([grad1,grad2]).shape)
    return np.array([grad1,-grad2])/self.value(pos) #-alpha times unitary vector

#-------------------------
  def laplacian(self,pos):
    dist=np.sqrt(np.sum(pos**2,axis=1))
    r1 = dist[0,:]
    r2 = dist[1,:]
    phi_12 = (2-self.alpha*r2)*np.exp(-self.alpha*r1)*np.exp(-self.alpha*r2/2)
    phi_21 = (2-self.alpha*r1)*np.exp(-self.alpha*r2)*np.exp(-self.alpha*r1/2)
    #lap_1 = self.alpha*2*(-phi_12 + np.exp(-self.alpha*(r2+r1/2)) + phi_21/2)/r1 + (self.alpha**2)*(phi_12-np.exp(-self.alpha*(r2+r1/2))-phi_21/4)
    #lap_2 = self.alpha*2*(-phi_21 + np.exp(-self.alpha*(r1+r2/2)) + phi_12/2)/r2 + (self.alpha**2)*(phi_21-np.exp(-self.alpha*(r1+r2/2))-phi_12/4)
    lap_1 = (self.alpha/(4*r1))*(self.alpha*r1-2)*np.exp(-self.alpha*(r2+r1))*(np.exp(self.alpha*r2/2)*(8-4*r2*self.alpha)+np.exp(self.alpha*r1/2)*(self.alpha*r1-8))
    lap_2 = (self.alpha/(4*r2))*(self.alpha*r2-2)*np.exp(-self.alpha*(r1+r2))*(np.exp(self.alpha*r1/2)*(8-4*r1*self.alpha)+np.exp(self.alpha*r2/2)*(self.alpha*r2-8))
    #print(np.average(lap_1))
    #print(np.average(self.value(pos)))
    return np.array([lap_1, -lap_2])/self.value(pos)
#-------------------------


if __name__=="__main__":
  # This part of the code will test your implementation. 
  # Don't modify it!
  import wavefunction
  # 2 electrons, 3 dimensions, 5 configurations.
  testpos=np.random.randn(2,3,5)
  print("Exponent wavefunction")
  ewf=ExponentSlaterWF(0.5)
  wavefunction.test_wavefunction(ExponentSlaterWF(alpha=0.5))
  
