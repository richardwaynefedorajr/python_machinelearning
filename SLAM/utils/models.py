import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

class Robot():
    def __init__(self, control_dimension=2):
        self.control_dimension = control_dimension
        self.u = np.zeros(self.control_dimension)
        
    def getControl(self): return self.u
    
    def getControlIndices(self):
        return range(0, self.control_dimension)
    
class Sensor():
    def __init__():
        self.v = None

class Environment():
    def __init__():
        pass
    
class Model():
    def __init__(self):     
        self.time_step = 0
        self.previous_timestamp = 0
        self.current_timestamp = 0
    
    def squeezeAngle(self, angle):
        if angle > np.pi:
            angle -= 2*np.pi
        if angle < -np.pi:
            angle += 2*np.pi
        return angle
    
    def getPoseIndices(self):
        return [0, 1, 2]
    
    def predict(self):
        pass
        
    def update(self):
        pass
    
    def linePlot(self, data, title, linewidth=0.75):
        sns.set(rc={"figure.dpi":600, 'savefig.dpi':600})
        vals = np.linspace(0,1,256)
        np.random.shuffle(vals)
        cmap = plt.cm.colors.ListedColormap(plt.cm.gist_rainbow(vals))
        sns.set_palette(cmap(np.linspace(0,1,cmap.N)))
        sns.set_style('darkgrid', {'axes.facecolor':'0.85'})
        plot = sns.lineplot(data, x='Pose [x]', y='Pose [y]', dashes=False, linewidth=linewidth, markers=True, sort=False)
        plot.figure.suptitle(title)
        return plot
    
    def plotPath(self):
        df = pd.DataFrame(data=np.c_[self.path[0,:], self.path[1,:]], 
                          columns=['Pose [x]','Pose [y]'])
        plot = self.linePlot(df, self.file_prefix+' Path')
        
        for i in range(self.path_covariance.shape[2]):
            x, y, psi = self.path[self.getPoseIndices(), i]
            self.confidence_ellipse(x, y, psi, i, plot.axes)
        
        plt.savefig(self.file_prefix+'_path.svg', bbox_inches='tight')
        # plt.show()
        
    def confidence_ellipse(self, x, y, psi, cov_ind, ax, n_std=3):
        cov = self.path_covariance[:, :, cov_ind]
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        radius_x = np.sqrt(1 + pearson)
        radius_y = np.sqrt(1 - pearson)
        print(radius_x, radius_y)
        ellipse = Ellipse((x,y), width=radius_x * 2, height=radius_y * 2, fill=False)
    
        # scale to 95% confidence intervals
        scale_x = np.sqrt(cov[0, 0]) * n_std        
        scale_y = np.sqrt(cov[1, 1]) * n_std
        
        # transf = transforms.Affine2D().rotate_deg(45) \
        #                               .scale(scale_x, scale_y) \
        #                               .translate(x, y)
        transf = transforms.Affine2D().scale(scale_x, scale_y)
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)