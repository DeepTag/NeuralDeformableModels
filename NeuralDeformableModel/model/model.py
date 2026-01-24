import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import trimesh
import math
from .geomorph import PointTransformerTriEncoder

class ODEFunc(nn.Module):
    '''
    This refers to the dynamics function f(x,t) in a IVP defined as dh(x,t)/dt = f(x,t). 
    For a given location (t) on point (x) trajectory, it returns the direction of 'flow'.
    Refer to Section 3 (Dynamics Equation) in the paper for details. 
    '''
    def __init__(self, num_hidden = 512, latent_len = 512):
        '''
        Initialization. 
        num_hidden: number of nodes in a hidden layer
        latent_len: size of the latent code being used
        '''
        
        super(ODEFunc, self).__init__()
        
        self.l1 = nn.Linear(3, num_hidden)
        self.l2 = nn.Linear(num_hidden, num_hidden)   
        self.l3 = nn.Linear(num_hidden, num_hidden)
        self.l4 = nn.Linear(num_hidden, 3)
        
        self.cond = nn.Linear(latent_len, num_hidden) 

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        self.nfe = 0
        self.zeros = torch.zeros((1, 1, latent_len))
        self.latent_dyn_zeros=None
#         self.tw = nn.Linear(1, num_hidden)
        
    def forward(self, t, xz):
        '''
        t: Torch tensor of shape (1,) 
        xz: Torch tensor of shape (N, #pts, 3+zdim). Along dimension 2, the point and shape embeddings are concatenated. 
        
        **NOTE**
        For the uniqueness property to hold, a single dynamics function (operating in 3D) must be used to compute 
        trajectories pertaining to points of a single shape. 
        
        Here, the shape encoding (same for all points of a shape) is used to choose a function which is applied over all the shape points.
        Hence, even though the input xz appears to be a 3+zdim dimensional state, the ODE is still restricted to a 3D state-space. 
        The concatenation is purely to make programming simpler without affecting the underlying theory. 
        
        '''
        point_features = self.relu(self.l1((xz[...,:3]))) # Extract point features Nx#ptsx3 -> Nx#ptsx512
        shape_features = self.tanh(self.cond(xz[...,3:]))  # Extract shape features Nx#ptsxzdim -> Nx#ptsx512
        
        point_shape_features = point_features*shape_features  # Compute point-shape features by elementwise multiplication
        # [Insight :]  Conditioning is critical to allow for several shapes getting learned by same NeuralODE. 
        #              Note that under current formulation, all points belonging to a shape share a common dynamics function.
        
        # Two residual blocks
        point_shape_features = self.relu(self.l2(point_shape_features)) + point_shape_features
        point_shape_features = self.relu(self.l3(point_shape_features)) + point_shape_features
        # [Insight :] Using less residual blocks leads to drop in performance
        #             while more residual blocks make model heavy and training slow due to more complex trajectories being learned.
        
        dyns_x_t = self.tanh(self.l4(point_shape_features)) #Computed dynamics of point x at time t
        # [Insight :] We specifically choose a tanh activation to get maximum expressivity as observed by He, et.al and Massaroli, et.al
        
        self.nfe+=1  #To check #ode evaluations
        
        # To prevent updating of latent codes during ODESolver calls, we simply make their dynamics all zeros. 
        if self.latent_dyn_zeros is None or self.latent_dyn_zeros.shape[0] != dyns_x_t.shape[0] or self.latent_dyn_zeros.shape[1] != dyns_x_t.shape[1]:
            self.latent_dyn_zeros = self.zeros.repeat(dyns_x_t.shape[0], dyns_x_t.shape[1], 1).type_as(dyns_x_t)
        
        return torch.cat([dyns_x_t, self.latent_dyn_zeros], dim=2) # output is therefore like [dyn_x, dyn_y, dyn_z, 0,0..,0] for a point
#         return torch.cat([dyns_x_t, xz[...,3:]],dim=2)
    
class NODEBlock(nn.Module):
    '''
    Function to solve an IVP defined as dh(x,t)/dt = f(x,t). 
    We use the differentiable ODE Solver by Chen et.al used in their NeuralODE paper.
    '''
    def __init__(self, odefunc, tol):
        '''
        Initialization. 
        odefunc: The dynamics function to be used for solving IVP
        tol: tolerance of the ODESolver
        '''
        super(NODEBlock, self).__init__()
        self.odefunc = odefunc
        self.cost = 0
        self.rtol = tol
        self.atol = tol
        
    def forward(self, x, time):
        '''
        Solves the ODE in the forward time. 
        '''
        self.odefunc.nfe = 0  #To check #ode evaluations
        self.forward_time = torch.tensor([0, time]).float().type_as(x)  # Time of integration (must be monotinically increasing!)
        # Solve the ODE with initial condition x and interval time.
        out = odeint(self.odefunc.to(x), x, self.forward_time, rtol = self.rtol, atol = self.atol)
        self.cost = self.odefunc.nfe  # Number of evaluations it took to solve it
        return out[1] 
    
    def invert(self, x, time):
        '''
        Solves the ODE in the reverse time. 
        '''
        self.inverse_time = torch.tensor([time, 0]).float().type_as(x)
        out = odeint(self.odefunc, x, self.inverse_time, rtol = self.rtol, atol = self.atol)
        self.cost = self.odefunc.nfe
        return out[1]
    
class DeformBlock(nn.Module):
    '''
    A single DeformBlock is made up of two NODE Blocks. Refer secion 3 (Overall Architecture)
    '''
    def __init__(self, time=0.2, num_hidden = 512, latent_len = 512, tol = 1e-5):
        super(DeformBlock, self).__init__()
        '''
        Initialization.
        time: some number 0-1
        num_hidden: Number of hidden nodes in the MLP of dynamics
        latent_len: Length of shape embeddings
        tol: tolerance of the ODE Solver
        '''
        
        # Two NODE Blocks
        self.l1 = NODEBlock(ODEFunc(num_hidden, latent_len), tol = tol)
        self.l2 = NODEBlock(ODEFunc(num_hidden, latent_len), tol = tol)
        
        self.time = time
        
    def forward_(self, x, code, time=None):
        '''
        Forward flow method
        
        x: BxNx3 input tensor
        code: Bxzdim tensor embedding
        time: some number 0-1
        
        y: BxNx3 output tensor
        '''
        
        if time is None:
            time=self.time
            
        xz = torch.cat([x, code.repeat(1, x.shape[1], 1)], dim=2)
        # Note: To enable condioned flows, we concatenate points with their corresponding shape embeddings. 
        #       Refer to code comments in ODEFunc.forward() for more details about this choice.

        x = self.l1(xz, time)
        x = self.l2(x, time)
        
        y = x[...,:3]  # output the corresponding 'flown' points.
        return y
    
    def backward_(self, x, code, time=None):
        '''
        Backward flow method
        
        x: BxNx3 input tensor
        code: Bxzdim tensor embedding
        time: some number 0-1
        
        y: BxNx3 output tensor
        
        **NOTE** We do not use this method in the main NMF pipeline, but may come handy for things like inverting the NMF!
        
        '''
        if time is None:
            time=self.time
            
        xz = torch.cat([x, code.repeat(1,x.shape[1],1)],dim=2)
        # Note: To enable condioned flows, we concatenate points with their corresponding shape embeddings. 
        #       Refer to code comments in ODEFunc.forward() for more details about this choice.

        x = self.l2.invert(xz, time)
        x = self.l1.invert(x, time)
        
        y = x[...,:3]
        
        return y
    
    def forward(self, code, x, y=None, time=None):
        '''
        code: Bxzdim tensor embedding
        x: BxNx3 input tensor
        y: BxNx3 output tensor
        time: some number 0-1
        '''
        
        # Calculate forward flow
        pred_y = self.forward_(x,code,time)
        
        
        if y is not None:
            # Calculate backward flow if required
            pred_x = self.backward_(y,code,time)
            return pred_y, pred_x
        
        return pred_y, None

def create_ellipsoid_spheroid_points1(pho, t=6, zs=1.0, n_theta=50, n_phi=70):
    theta = torch.linspace(-math.pi / 2, math.pi / t, n_theta) #50
    phi = torch.linspace(-math.pi, math.pi*(1 - 2/n_phi), n_phi) #70

    theta, phi = torch.meshgrid(theta, phi)
    x = pho * torch.cos(theta) * torch.cos(phi)
    y = pho * torch.cos(theta) * torch.sin(phi)
    z = zs*pho * torch.sin(theta)
    x = x.view(-1, 1)
    y = y.view(-1, 1)
    z = z.view(-1, 1)
    points = torch.stack([x, y, z])
    points = points.view(3, -1)
    points = torch.transpose(points, 0, 1)

    return points

def add_lids_to_ellipsoid_spheroid_points(m_pc1, n_theta=50, n_phi=70, n=5, a=0.95):
    m_pc1 = m_pc1.view(n_theta, n_phi, 3)
    s = m_pc1.shape
    m_pc11 = torch.zeros([s[0] + n, s[1], s[2]])
    m_pc11[:s[0], ::] = m_pc1
    for i in range(n):
        m_pc11[s[0] + i, :, 0] = a * m_pc11[s[0] + i - 1, :, 0]
        m_pc11[s[0] + i, :, 1] = a * m_pc11[s[0] + i - 1, :, 1]
        m_pc11[s[0] + i, :, 2] = m_pc11[s[0] + i - 1, :, 2]

    return m_pc11.view(-1, 3)

def create_ellipsoid_spheroid_points3(pho, t=6, zs=1.0, n=48, inner=0):
    theta = torch.linspace(-math.pi / 2, math.pi / t, 50) #50
    if inner==0:
        phi = torch.linspace((-math.pi / 2) * (1 - 1 / n), (math.pi / 2) * (1 - 1 / n), n) #112
    else:
        phi = torch.linspace((math.pi / 2) * (1 - 1 / n), (-math.pi / 2) * (1 - 1 / n), n)  # 112

    theta, phi = torch.meshgrid(theta, phi)
    x = - pho * torch.cos(theta) * torch.cos(phi)
    y = 1 * torch.cos(theta) * torch.sin(phi)
    z = zs*1 * torch.sin(theta)
    x = x.view(-1, 1)
    y = y.view(-1, 1)
    z = z.view(-1, 1)
    points = torch.stack([x, y, z])
    points = points.view(3, -1)
    points = torch.transpose(points, 0, 1)

    return points


class NeuralDeformableModel(nn.Module):
    '''
    Implementation of the Neural Mesh Flow pipeline. Refer Section 3 in the paper for more details.
    '''

    def __init__(self, zdim=32, time=0.2, tol=1e-5):
        super(NeuralDeformableModel, self).__init__()
        '''
        Initialization
        encoder_type: 'image' or 'point' for SVR and shape completion tasks repectively
        PATH_svr: model file for trained PointsSVR
        zdim: length of latent embedding
        time: some number 0-1
        tol: tolerance of ODESolver.
        '''

        '''
        **** NOTE on design choices  ******

        zdim : We did not observe much benefit of increased latent embedding size (i.e. >1000) and it simply increases memory requirement
        time : time of integration (set to 0.2) is chosen since it is long enough for effective integration but not too large to cause complex dynamics
        tol  : A lower tol is always better but comes at a cost of inference time. Refer to ablation in Supplementary for this.

        '''

        print("Neural Mesh Flow with {} length embedding initialized".format(zdim))

        # Three deform blocks to cause successive refinements. Refer Section 3 (Overal architecture)

        ### primitive 1
        self.scale = nn.Sequential(
            nn.Linear(zdim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.trans = nn.Sequential(
            nn.Linear(zdim, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.Tanh()
        )
        self.quaternion = nn.Sequential(
            nn.Linear(zdim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Tanh()
        )

        self.a1 = nn.Sequential(
            nn.Linear(zdim, 16),
            nn.ReLU(),
            nn.Linear(16, 50),
            nn.Sigmoid()
        )

        self.a2 = nn.Sequential(
            nn.Linear(zdim, 16),
            nn.ReLU(),
            nn.Linear(16, 50),
            nn.Sigmoid()
        )

        self.a3 = nn.Sequential(
            nn.Linear(zdim, 16),
            nn.ReLU(),
            nn.Linear(16, 50),
            nn.Sigmoid()
        )

        self.e1 = nn.Sequential(
            nn.Linear(zdim, 16),
            nn.ReLU(),
            nn.Linear(16, 50),
            nn.Tanh()
        )

        self.e2 = nn.Sequential(
            nn.Linear(zdim, 16),
            nn.ReLU(),
            nn.Linear(16, 50),
            nn.Tanh()
        )

        ### primitive 2
        self.scale2 = nn.Sequential(
            nn.Linear(zdim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.trans2 = nn.Sequential(
            nn.Linear(zdim, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.Tanh()
        )
        self.quaternion2 = nn.Sequential(
            nn.Linear(zdim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Tanh()
        )

        self.a12 = nn.Sequential(
            nn.Linear(zdim, 16),
            nn.ReLU(),
            nn.Linear(16, 50),
            nn.Sigmoid()
        )

        self.a22 = nn.Sequential(
            nn.Linear(zdim, 16),
            nn.ReLU(),
            nn.Linear(16, 50),
            nn.Sigmoid()
        )

        self.a32 = nn.Sequential(
            nn.Linear(zdim, 16),
            nn.ReLU(),
            nn.Linear(16, 50),
            nn.Sigmoid()
        )

        self.e12 = nn.Sequential(
            nn.Linear(zdim, 16),
            nn.ReLU(),
            nn.Linear(16, 50),
            nn.Tanh()
        )

        self.e22 = nn.Sequential(
            nn.Linear(zdim, 16),
            nn.ReLU(),
            nn.Linear(16, 50),
            nn.Tanh()
        )

        ### primitive 3
        self.scale3 = nn.Sequential(
            nn.Linear(zdim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.trans3 = nn.Sequential(
            nn.Linear(zdim, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.Tanh()
        )
        self.quaternion3 = nn.Sequential(
            nn.Linear(zdim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Tanh()
        )

        self.a13 = nn.Sequential(
            nn.Linear(zdim, 16),
            nn.ReLU(),
            nn.Linear(16, 50),
            nn.Sigmoid()
        )

        self.a23 = nn.Sequential(
            nn.Linear(zdim, 16),
            nn.ReLU(),
            nn.Linear(16, 50),
            nn.Sigmoid()
        )

        self.a33 = nn.Sequential(
            nn.Linear(zdim, 16),
            nn.ReLU(),
            nn.Linear(16, 50),
            nn.Sigmoid()
        )

        self.e13 = nn.Sequential(
            nn.Linear(zdim, 16),
            nn.ReLU(),
            nn.Linear(16, 50),
            nn.Tanh()
        )

        self.e23 = nn.Sequential(
            nn.Linear(zdim, 16),
            nn.ReLU(),
            nn.Linear(16, 50),
            nn.Tanh()
        )

        ### primitive 4
        self.a14 = nn.Sequential(
            nn.Linear(zdim, 16),
            nn.ReLU(),
            nn.Linear(16, 50),
            nn.Sigmoid()
        )

        # Template spheres for training/testing.

        # [Insight :] While trainig with smaller (#vertices) sphere is faster, inference with larger sphere is more accurate.
        # Coosing spheres with even lower vertices can cause drop in performance due to unstable optimization.
        # Using #vertices >2520 doesn't yeild much benefit and takes more training time.

        # These spheres are generated using Pymesh library : pymesh.generate_icosphere(radius=1, center=(0,0,0), refinement_order=3 or 4)

        epi_pc = create_ellipsoid_spheroid_points1(1, n_theta=50, n_phi=100)  # 50*100
        endo_pc = create_ellipsoid_spheroid_points1(0.65, zs=1.4, n_theta=50, n_phi=100)  # 50*100
        rv1 = create_ellipsoid_spheroid_points3(1, t=2, n=60, inner=0).view(50, 60,  3)  # 50*60
        rv2 = create_ellipsoid_spheroid_points3(0.2, t=2, n=40, inner=1).view(50, 40, 3)  # 50*50
        rv = torch.cat([rv1, rv2], dim=1) # 50*100
        # print('rv.shape')
        # print(rv.shape)
        epi_pc = add_lids_to_ellipsoid_spheroid_points(epi_pc, n_theta=50, n_phi=100, n=5, a=0.95) # 55*100
        # endo_pc = add_lids_to_ellipsoid_spheroid_points(endo_pc, n_theta=50, n_phi=100, n=5, a=0.95) # 55*100
        self.v = torch.cat([epi_pc, endo_pc, rv.view(-1, 3)], dim=0)

        self.encoder = PointTransformerTriEncoder(nblocks=4, nneighbor=16, d_points=3, transformer_dim=512).float()  # Initialize PointNet encoder

        # Three deform blocks to cause successive refinements. Refer Section 3 (Overal architecture)
        self.db1 = DeformBlock(time, num_hidden=512, latent_len=zdim, tol=tol)
        self.db2 = DeformBlock(time, num_hidden=512, latent_len=zdim, tol=tol)
        self.db3 = DeformBlock(time, num_hidden=512, latent_len=zdim, tol=tol)

        # Template spheres for training/testing.

        # [Insight :] While trainig with smaller (#vertices) sphere is faster, inference with larger sphere is more accurate.
        # Coosing spheres with even lower vertices can cause drop in performance due to unstable optimization.
        # Using #vertices >2520 doesn't yeild much benefit and takes more training time.

        # These spheres are generated using Pymesh library : pymesh.generate_icosphere(radius=1, center=(0,0,0), refinement_order=3 or 4)

        self.time = time

        # Initialize InstanceNorm layers
        # self.norm0 = InstanceNorm(zdim)
        # self.norm1 = InstanceNorm(zdim)

    def get_code_(self, x):
        '''
        Fetch the shape embeddings for point clouds in x
        x: BxNx3 input tensor
        code: Bx1xzdim tensor embedding
        '''
        # code = self.encoder(x)
        code1, code2, code3 = self.encoder(torch.cat([x, x], -1))
        code1 = code1.unsqueeze(1)
        code2 = code2.unsqueeze(1)
        code3 = code3.unsqueeze(1)

        return code1, code2, code3

    def forward(self, input):
        '''
        input: BxNx3 tensor
        pred_y1: BxNx3 tensor; vertices after first deformation block
        pred_y2: BxNx3 tensor; vertices after second deformation block
        pred_y3: BxNx3 tensor; vertices after third deformation block
        face: BxKx3; faces to be used for constructing differentiable meshes
        '''
        batch_size = input.shape[0]

        # Input is a (centered) point cloud. Directly compute its embedding
        points_gap1, points_gap2, points_gap3 = self.get_code_(input)  # Get latent code using point cloud at native resolution
        points_gap1 = points_gap1.type_as(input)
        points_gap2 = points_gap2.type_as(input)
        points_gap3 = points_gap3.type_as(input)

        # print('points_gap.shape')
        # print(points_gap.shape)

        sph = self.v.unsqueeze(0).repeat(batch_size, 1, 1).type_as(input)

        scale = self.scale(points_gap1)
        trans = self.trans(points_gap1)
        quaternion = self.quaternion(points_gap1)

        scale2 = self.scale2(points_gap2)
        trans2 = self.trans2(points_gap2)
        quaternion2 = self.quaternion2(points_gap2)

        scale3 = self.scale3(points_gap3)
        trans3 = self.trans3(points_gap3)
        quaternion3 = self.quaternion3(points_gap3)

        a1 = 5 * self.a1(points_gap1)
        a2 = 5 * self.a2(points_gap1)
        a3 = 5 * self.a3(points_gap1)
        e1 = self.e1(points_gap1)
        e2 = self.e2(points_gap1)

        a12 = 5 * self.a12(points_gap2)
        a22 = 5 * self.a22(points_gap2)
        a32 = 5 * self.a32(points_gap2)
        e12 = self.e12(points_gap2)
        e22 = self.e22(points_gap2)

        a13 = 5 * self.a13(points_gap3)
        a23 = 5 * self.a23(points_gap3)
        a33 = 5 * self.a33(points_gap3)
        e13 = self.e13(points_gap3)
        e23 = self.e23(points_gap3)

        a14 = 5 * self.a14(points_gap3)

        return points_gap1, points_gap2, points_gap3, trans.view(-1, 1, 3), quaternion.view(-1, 4), scale.view(-1, 1, 1), a1.view(-1, 50, 1), a2.view(
            -1, 50, 1), a3.view(-1, 50, 1), e1.view(-1, 50, 1), e2.view(-1, 50, 1), \
               trans2.view(-1, 1, 3), quaternion2.view(-1, 4), scale2.view(-1, 1, 1), a12.view(-1, 50, 1), a22.view(
            -1, 50, 1), a32.view(-1, 50, 1), e12.view(-1, 50, 1), e22.view(-1, 50, 1), \
               trans3.view(-1, 1, 3), quaternion3.view(-1, 4), scale3.view(-1, 1, 1), a13.view(-1, 50, 1), a23.view(
            -1, 50, 1), a33.view(-1, 50, 1), e13.view(-1, 50, 1), e23.view(-1, 50, 1), a14.view(-1, 50, 1), sph

    def neural_mesh_forward1(self, code, sph, tgt, time=None):
        '''
        input: BxNx3 tensor
        pred_y1: BxNx3 tensor; vertices after first deformation block
        pred_y2: BxNx3 tensor; vertices after second deformation block
        pred_y3: BxNx3 tensor; vertices after third deformation block
        face: BxKx3; faces to be used for constructing differentiable meshes
        '''
        # First Deform Block computation and its instance norm
        pred_y1, pred_y0 = self.db1(code, sph, tgt, time)

        return pred_y1, pred_y0

    def neural_mesh_forward2(self, code, sph, tgt, time=None):
        '''
        input: BxNx3 tensor
        pred_y1: BxNx3 tensor; vertices after first deformation block
        pred_y2: BxNx3 tensor; vertices after second deformation block
        pred_y3: BxNx3 tensor; vertices after third deformation block
        face: BxKx3; faces to be used for constructing differentiable meshes
        '''
        # First Deform Block computation and its instance norm
        pred_y1, pred_y0 = self.db2(code, sph, tgt, time)

        return pred_y1, pred_y0

    def neural_mesh_forward3(self, code, sph, tgt, time=None):
        '''
        input: BxNx3 tensor
        pred_y1: BxNx3 tensor; vertices after first deformation block
        pred_y2: BxNx3 tensor; vertices after second deformation block
        pred_y3: BxNx3 tensor; vertices after third deformation block
        face: BxKx3; faces to be used for constructing differentiable meshes
        '''
        # First Deform Block computation and its instance norm
        pred_y1, pred_y0 = self.db3(code, sph, tgt, time)

        return pred_y1, pred_y0