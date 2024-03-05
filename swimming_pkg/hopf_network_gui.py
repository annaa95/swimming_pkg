import numpy as np
from scipy.spatial.transform import Rotation as Rot
# for RL 
MU_LOW = 1
MU_UPP = 2

class HopfNetwork():
  """ CPG network based on hopf polar equations mapped to foot positions in Cartesian space.  

  Foot Order is FR, FL, RR, RL
  (Front Right, Front Left, Rear Right, Rear Left)

  Some sample values:
  Pace, Trot, Bound:
  omega_swing=5*2*np.pi, 
  omega_stance=2*2*np.pi, 

  WALK: 
  omega_swing=8*2*np.pi, 
  omega_stance=2*2*np.pi, 
  """
  def __init__(self,
                mu=1**2,                 # intrinsic amplitude, converges to sqrt(mu)
                omega_swing=5*2*np.pi,   # frequency in swing phase 
                omega_stance=2*2*np.pi,  # frequency in stance phase 
                gait=np.array([0, np.pi, np.pi, 0]),             # Gait, can be TROT, WALK, PACE, BOUND, etc.
                alpha=25,                # amplitude convergence factor default 50
                coupling_strength=1,     # coefficient to multiply coupling matrix
                couple=True,             # whether oscillators should be coupled
                time_step=0.001,         # time step at which to integrate dynamics equations
                robot_height=0.3,       # in nominal case (standing)
                des_step_len=0.05,       # desired step length
                max_step_len_rl=0.1,     # max step length, for RL scaling 
                use_RL=False,             # whether to learn paramters with RL 
                belly_length = 0.009054,
                step_height = 2*0.017351, 
                vertical_offset = -0.15,
                horizontal_offset = 0,
                compression_ratio = 0.553652,
                belly_curvature =  0.884409,
                inclination = 0
                ):
    
    ###############
    #print(gait) #order in gui FR -HR -FL -HL # order in cpg code FR, FL, RR, RL
    gait = np.array([gait[0], gait[2], gait[1], gait[3]])
    # initialize CPG data structures: amplitude and phase
    self.X = np.zeros((2,4))
    self.X_dot = np.zeros((2,4))

    self._mu = mu
    self._omega_swing = omega_swing
    self._omega_stance = omega_stance  
    self._couple = couple
    self._coupling_strength = coupling_strength
    self._alpha = alpha

    self._set_gait(gait)

    # set initial conditions  
    self.X[0,:] = np.random.rand(4) * .1
    self.X[1,:] = self.PHI[0,:] 

    # time step 
    self._dt = time_step

    # for body and swing heights 
    """
    L           0.038866
    l           0.009054
    h           0.017351
    phi              0.0
    b           0.884409
    c           0.553652
    theta_to         0.0
    theta_td         0.0
    """
    self._h = step_height
    self._b = belly_curvature
    self._c = compression_ratio
    self._l = belly_length
    self._robot_height = robot_height 
    self._des_step_len = des_step_len#0.038866 #L
    self._Xoff = horizontal_offset,
    self._Yoff = vertical_offset,
    self._phi = inclination
    
    rmin = Rot.from_rotvec([0, self._phi, 0]).as_matrix()
    rmin = np.array([rmin[0,0], rmin[0,2],rmin[2,0],rmin[2,2]]).reshape(2,2)
    self._weights = [1,3,3,1]
    # keypoints for trajectories  simplified
    pts = {'p_to': np.dot(rmin, np.array([-self._des_step_len/2, 0])),
        'p_td': np.dot(rmin,np.array([self._des_step_len/2,0])),
        'p_sw': np.dot(rmin, np.array([0, -self._h/2*(-1+self._b)])),
        'p_st': np.dot(rmin, np.array([0, -self._h/2*(1+self._b)])),
        'p_tosw':np.dot(rmin, np.array([(-self._des_step_len/2),self._h/2*(1+self._c)])),
        'p_swto':np.dot(rmin, np.array([-self._l , -self._h/2*(-1+self._b)])),
        'p_swtd':np.dot(rmin, np.array([self._l, -self._h/2*(-1+self._b)])),
        'p_tdsw':np.dot(rmin, np.array([(self._des_step_len/2), self._h/2*(1+self._c)])),
        'p_tdst':np.dot(rmin, np.array([(self._des_step_len/2),-self._h/2*(1-self._c)])),
        'p_sttd':np.dot(rmin, np.array([self._l, -self._h/2*(1+self._b)])),
        'p_stto':np.dot(rmin, np.array([-self._l,-self._h/2*(1+self._b)])),
        'p_tost':np.dot(rmin, np.array([(-self._des_step_len/2),-self._h/2*(1-self._c)]))
        }
    self.keys0 = np.array([pts['p_to'], pts['p_tosw'], pts['p_swto'], pts['p_sw']])
    self.keys1 = np.array([pts['p_sw'], pts['p_swtd'], pts['p_tdsw'], pts['p_td']])
    self.keys2 = np.array([pts['p_td'], pts['p_tdst'], pts['p_sttd'], pts['p_st']])
    self.keys3 = np.array([pts['p_st'], pts['p_stto'], pts['p_tost'], pts['p_to']])  

    # for RL
    self.use_RL = use_RL
    self._omega_rl = np.zeros(4)
    self._mu_rl = np.zeros(4) 
    self._max_step_len_rl = max_step_len_rl
    if use_RL:
      self.X[0,:] = MU_LOW # mapping MU_LOW=1 to MU_UPP=2

  def _set_gait(self,gait):
    """ For coupling oscillators in phase space. """
    self.PHI = np.array([gait, gait-gait[1], gait-gait[2], gait-gait[3]])
    print(self.PHI)  

  def update(self):
    """ Update oscillator states. """

    # update parameters, integrate
    if not self.use_RL:
      self._integrate_hopf_equations()
    else:
      self._integrate_hopf_equations_rl()
    
    # map CPG variables to Cartesian foot xz positions
    xz = np.zeros((4,2))

    # loop through phases, check if in swing/stance        
    
    wgt_theta =lambda theta,j: self._weights[j]*((1-theta/(np.pi/2))**(3-j))*((theta/(np.pi/2))**j)

    for i,theta in enumerate(self.X[1,:]):
        if theta >= 0 and theta< np.pi/2:
            keys0 = self.keys0.copy()
            keys0[0:2,0]*= self.X[0,i]
            xz[i] = np.dot(np.array([wgt_theta(theta, 0), wgt_theta(theta, 1), wgt_theta(theta,2), wgt_theta(theta,3)]),keys0)
        elif theta >= np.pi/2 and theta< np.pi:
            keys1 = self.keys1.copy()
            keys1[2:4,0]*= self.X[0,i]
            xz[i] = np.dot(np.array([wgt_theta(theta-np.pi/2, 0), wgt_theta(theta-np.pi/2, 1), wgt_theta(theta-np.pi/2,2), wgt_theta(theta-np.pi/2,3)]),keys1)
        elif theta >= np.pi and theta< 3*np.pi/2:
            keys2 = self.keys2.copy()
            keys2[0:2,0]*= self.X[0,i]
            xz[i] = np.dot(np.array([wgt_theta(theta-np.pi, 0), wgt_theta(theta-np.pi, 1), wgt_theta(theta-np.pi,2), wgt_theta(theta-np.pi,3)]),keys2)
        else:
            keys3 = self.keys3.copy()
            keys3[2:4,0]*= self.X[0,i]
            xz[i] = np.dot(np.array([wgt_theta(theta-3*np.pi/2, 0), wgt_theta(theta-3*np.pi/2, 1), wgt_theta(theta-3*np.pi/2,2), wgt_theta(theta-3*np.pi/2,3)]),keys3)
                 
    if not self.use_RL:
      return xz.transpose()[0]+self._Xoff, xz.transpose()[1]+self._Yoff
    else: 
      # with RL, due to our mu ranges, scale to fraction of max step len
      r = np.clip(self.X[0,:],MU_LOW,MU_UPP) 
      r = self._max_step_len_rl * (r - MU_LOW)
      return -r * np.cos(self.X[1,:])
              
  def _integrate_hopf_equations(self):
    """ Hopf polar equations and integration """
    # bookkeeping - save copies of current CPG states 
    X = self.X.copy() # r0, theta0, r1, theta1, r2, theta2, r3, theta3
    X_dot_prev = self.X_dot.copy() # optional
    X_dot = np.zeros((2,4))

    # loop through each leg's oscillator
    for i in range(4):
      # get amplitude and phase
      r, theta = X[:,i]
      # amplitude 
      r_dot = self._alpha *(self._mu - r**2)*r
      # phase
      theta = theta % (2*np.pi)
      if theta > np.pi: 
        omega = self._omega_stance
      else:
        omega = self._omega_swing

      if self._couple:
        omega += np.sum( X[0,:] * self._coupling_strength * np.sin(X[1,:] - theta - self.PHI[i,:] ) )

      X_dot[:,i] = [r_dot, omega]

    # integrate 
    self.X = X + (X_dot_prev + X_dot) * self._dt / 2
    # self.X = X + X_dot * self._dt 
    self.X_dot = X_dot
    self.X[1,:] = self.X[1,:] % (2*np.pi)

  ###################### Helper functions for accessing CPG States
  def get_r(self):
    return self.X[0,:]

  def get_theta(self):
    return self.X[1,:]

  def get_dr(self):
    return self.X_dot[0,:]

  def get_dtheta(self):
    return self.X_dot[1,:]

  ###################### Functions for setting parameters for RL
  def set_omega_rl(self, omegas):
    self._omega_rl = omegas 

  def set_mu_rl(self, mus):
    self._mu_rl = mus

  def _integrate_hopf_equations_rl(self):
    """ Hopf polar equations and integration, using quantities set by RL to learn to coordinate gaits. """
    # bookkeeping - save copies of current CPG states 
    X = self.X.copy()
    X_dot_prev = self.X_dot.copy() 
    X_dot = np.zeros((2,4))

    # loop through each leg's oscillator
    for i in range(4):
      # get amplitude and phase
      r, theta = X[:,i]
      # amplitude 
      r_dot = self._alpha *(self._mu_rl[i] - r**2)*r
      # phase
      theta_dot = self._omega_rl[i]

      X_dot[:,i] = [r_dot, theta_dot]

    # integrate 
    self.X = X + (X_dot_prev + X_dot) * self._dt / 2
    self.X_dot = X_dot
    self.X[1,:] = self.X[1,:] % (2*np.pi)



