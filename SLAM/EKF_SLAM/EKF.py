import sys
sys.path.append('../utils')
from models import Robot, Sensor, Model, Environment

import numpy as np

class Robot2D(Robot):
    def __init__(self, control_dimension=2):
        super().__init__()
        v_ind, omega_ind = self.getControlIndices()
        self.u[v_ind] = 1
        self.u[omega_ind] = np.pi/16
        
    def getControl(self):
        u = super().getControl()
        v_ind, omega_ind = self.getControlIndices()
        return u[v_ind], u[omega_ind]
    
class EKF(Model):
    def __init__(self, state_covariance_init=1e-6, process_noise_covariance_init=0.01):
        super().__init__()
        self.file_prefix = 'ekf'
        
        self.path = np.zeros((3,1))

        # state vector and covariance matrix
        self.x = np.zeros(3)
        self.P = np.eye(3)*state_covariance_init
        self.path_covariance = self.P[:, :, np.newaxis]

        # jacobians and process noise covariance
        self.F_x = np.ones((3,3))
        self.F_u = np.ones((3,2))
        self.Q = np.eye(2)*process_noise_covariance_init
        
        # measurement
        self.z = None
        
        # measurement jacobian and measurement noise covariance
        self.R = None
        self.H = None
        
        # Kalman gain
        self.K = None
    
    def getPose(self):
        x_ind, y_ind, psi_ind = self.getPoseIndices()
        return self.x[x_ind], self.x[y_ind], self.x[psi_ind]
    
    # equation of motion x_robot = f(x_robot, u, n) where u is the control input and n is the perturbation
    def predict(self, robot, timestamp):
        v, omega = robot.getControl()
        x, y, psi = self.getPose()
        dt = self.current_timestamp - self.previous_timestamp
        self.previous_timestamp = self.current_timestamp
        self.current_timestamp = timestamp
        
        # motion model
        self.x[self.getPoseIndices()] += dt*np.array([v*np.cos(psi), 
                                                      v*np.sin(psi),
                                                      omega])
        
        # jacobians
        self.F_x = np.array([[1, 0, -dt*v*np.sin(psi)],
                             [0, 1,  dt*v*np.cos(psi)],
                             [0, 0,                 1]])
        self.F_u = dt*np.array([[np.cos(psi), 0],
                                [np.sin(psi), 0],
                                [0          , 1]])
   
        # update covariance
        self.P = self.F_x@self.P@self.F_x.T + self.F_u@self.Q@self.F_u.T
        
        # store history
        self.path = np.hstack((self.path, self.x[self.getPoseIndices(), np.newaxis]))
        self.path_covariance = np.append(self.path_covariance, self.P[:, :, np.newaxis], axis=-1)
                                        
    # measurement equation
    def h():
        pass
    
test_robot = Robot2D()
test_ekf = EKF()
timestep = 0.1
time = 0 + timestep

for i in range(1,10):
    test_ekf.predict(test_robot, time)
    time += timestep
    
test_ekf.plotPath()