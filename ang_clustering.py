# Import stuff
import numpy as np 
import scipy
from scipy.special import erf
from colossus.cosmology import cosmology
from scipy import interpolate 
from scipy.interpolate import UnivariateSpline
import camb
from camb import model, initialpower
#====================================================


# Now create a class that can create CAMB cosmologies for different matter densities and sigma_8

class Cosmology:
    
    def __init__(self,omega_m,sigma_8,h,z):
        self.omega_m = omega_m
        self.sigma_8 = sigma_8
        self.h = h
        self.z = z
        self.k_max = 10.0
        self.c = 2.99792e+5
        #=========================
        
        cosmo = camb.CAMBparams()
        cosmo.set_cosmology(H0=100*self.h, ombh2=0.049*(self.h**2.0), omch2=(self.omega_m - 0.049)*(self.h**2.0), mnu=0.06, omk=0, tau=0.06)
        cosmo.InitPower.set_params(As=2.0e-9, ns=0.96)
        results = camb.get_results(cosmo)
        cosmo.set_matter_power(redshifts=[0.0], kmax=10.0)
        cambres= camb.get_transfer_functions(cosmo)
        cosmo.NonLinear = model.NonLinear_both
        kh, z, pk = cambres.get_matter_power_spectrum(minkh=1e-3, maxkh=0.1, npoints = 10)
        sigma_8_temp = cambres.get_sigma8()
        As_new  = ((self.sigma_8/sigma_8_temp)**2.0)*(2.0e-9)
        cosmo.InitPower.set_params(As=As_new, ns=0.96)
        cambres = camb.get_results(cosmo)
        backres = camb.get_background(cosmo)

        self.chi = backres.comoving_radial_distance(self.z)/self.h
           
        self.PK = camb.get_matter_power_interpolator(cosmo, nonlinear=True, 
                hubble_units=True, k_hunit=True, kmax=self.k_max, zmin = 0.0, zmax=self.z[-1]) 
        
        self.H_z = self.h*(backres.hubble_parameter(self.z))/self.c #Hubble parameter in 1/Mpc h^{-1}



#===================================================================================================================
#===================================================================================================================

# Selecting cosmologies

# Instantize cosmologies 

omega_m = 0.31
sigma_8 = 0.81
h = 0.675
alpha_om  = omega_m/10.0
alpha_sig = sigma_8/10.0

#==========================
nz = 1000 #number of steps to use for the radial/redshift integration

zarray = np.linspace(0,4.0,nz)
z = zarray[1:-1]

cosmo_fid = Cosmology(omega_m, sigma_8, h, z)
cosmo_1 = Cosmology(omega_m + alpha_om, sigma_8, h, z)
cosmo_2 = Cosmology(omega_m - alpha_om, sigma_8, h, z)
cosmo_3 = Cosmology(omega_m, sigma_8 + alpha_sig, h, z)
cosmo_4 = Cosmology(omega_m, sigma_8 - alpha_sig, h, z)



#=====================================================================================================
#=====================================================================================================
def cosmoselector(omega, sigma):
    #function that selects cosmology
    
    omfid = 0.31
    sigfid = 0.81
    
    cosmo_dict = {'cosmo_fid': cosmo_fid,
                  'cosmo_1' : cosmo_1,
                  'cosmo_2' : cosmo_2,
                  'cosmo_3' : cosmo_3,
                  'cosmo_4' : cosmo_4}
    
    
    if (omega==omfid):
        if (sigma == sigfid):
            cosm_sel = cosmo_dict['cosmo_fid']
        elif (sigma > sigfid):
            cosm_sel = cosmo_dict['cosmo_3']
        else:
            cosm_sel = cosmo_dict['cosmo_4']
    elif (omega > omfid): 
        cosm_sel = cosmo_dict['cosmo_1']
    else:
        cosm_sel = cosmo_dict['cosmo_2']
        
    
    return cosm_sel

#==============================================================================================
#==============================================================================================


# Function that calculates and returns window function W(z) for clustering 

def W_z_clust(z, dz, z_i, z_f, sig_z, b_0, z_0 ):
    """
    Function that calculates the window function for 2D galaxy clustering
    -----------------
    Inputs:
    z: array of redshifts where we are going to calculate the window function
    dz: array of dz's - useful for the integral
    z_i : lower redshift limit of the bin
    z_f : upper redshift limit of the bin
    sig_z : photometric error
    b_0 : present day bias
    z_0 : parameter related to the mean redshift of the sample
    ---------------
    Returns:
    The window function and its integral over all redshifts for a given bin with given limits
    
    """
    # Overall redshift distribution - DES like
    dNdz = (2.27*(z**1.25)/(z_0**2.25))*np.exp(-(z/z_0)**2.29)
    
    # Photometric window function
    x_min = (z - z_i)/((1.0+z)*sig_z*np.sqrt(2.0))
    x_max = (z - z_f)/((1.0+z)*sig_z*np.sqrt(2.0))
    F_z = 0.5*(erf(x_min) - erf(x_max))
    
    # Normalization
    norm_const = np.dot(dz, dNdz*F_z)
    
    #bias model
    bias = b_0*np.sqrt(1.0 + z)
    
    # Window function 
    
    W_z_bin = bias*dNdz*F_z/norm_const

    
    
    return W_z_bin, norm_const



# Function that calculates C_l,ij

def C_l_ij(z_i_1,z_f_1,z_i_2,z_f_2, Omega_m_var , sig_8_var , b_0 , sig_z , z_0):
    """
    Function that calculates the C_l between two bins 
    -----------------
    Inputs:
    z_i_1 : Lower limit of the first bin
    z_f_1 : Upper limit of the first bin
    z_i_2 : Lower limit of the second bin
    z_f_2 : Upper limit of the second bin
    Omega_m_var: Omega matter - can change
    sig_8_var : Sigma_8 parameter - can change
    sig_z : photo-z error
    b_0: present day bias
    z_0 : paramerer related to the mean redshift of the sample
    ---------------
    Returns:
    ls and C_l betwenn two bins, i and j. It is the auto spectrum if i=j
    """
    # Constant
    h = 0.675
    c = 2.99792e+5
    
    #======================================
    #====================================================================================
    #====================================================================================
    # Selecting cosmology
    
    cosmo = cosmoselector(Omega_m_var, sig_8_var)
    
    #====================================================================================
    #====================================================================================
    #Redshift range for calculations and integration
    
    nz = 1000 #number of steps to use for the radial/redshift integration
    kmax=10.0  #kmax to use

    zarray = np.linspace(0,4.0,nz)
    dzarray = (zarray[2:]-zarray[:-2])/2.0
    zarray = zarray[1:-1]
    
    
    #Take two window functions
    
    W_1 = W_z_clust(zarray, dzarray, z_i_1, z_f_1, sig_z, b_0, z_0 )[0]
    W_2 = W_z_clust(zarray, dzarray, z_i_2, z_f_2, sig_z, b_0, z_0 )[0]
    
    #====================================================================================
    #====================================================================================
    #Calculate Hubble parameter and comoving distance
    
    Hubble = cosmo.H_z
    
    # Get comoving distance - in Mpc/h
    chis = cosmo.chi
    
    #========================================================
    # Get the full prefactor of the integral
    prefact = W_1*W_2*Hubble/(chis**2.0)
    #====================================================================================
    
    #===================================================================================
    #===================================================================================
    #Do integral over z
    ls_lin = np.linspace(1.0, 3.0, 50, dtype = np.float64)
    ls = 10.0**ls_lin
    c_ell=np.zeros(ls.shape)
    w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
    for i, l in enumerate(ls):
        k=(l+0.5)/chis
        w[:]=1
        w[k<1e-4]=0
        w[k>=kmax]=0
        c_ell[i] = np.dot(dzarray, w*cosmo.PK.P(zarray, k, grid=False)*prefact)
    
    #===================================================================================
    # Retrurn the array of C_ell
    
    return ls, c_ell


# Now the Derivatives
#============================================================================================

def matter_der_C_l_ij(z_i_1,z_f_1,z_i_2,z_f_2, Omega_m , sig_8, b_0 , sig_z , z_0):
    """
    Function that calculates the derivative of C_l with respect to matter between two bins 
    -----------------
    Inputs:
    z_i_1 : Lower limit of the first bin
    z_f_1 : Upper limit of the first bin
    z_i_2 : Lower limit of the second bin
    z_f_2 : Upper limit of the second bin
    Omega_m: Omega matter
    sig_8: Sigma_8 parameter
    sig_z : photo-z error
    b_0: present day bias
    z_0 : paramerer related to the mean redshift of the sample
    ---------------
    Returns:
    derivative w/r to matter of C_l betwenn two bins, i and j
    """
    alpha = Omega_m/10.0
    
    C_mat_1 = C_l_ij(z_i_1,z_f_1,z_i_2,z_f_2, Omega_m+alpha , sig_8 , b_0 , sig_z , z_0)[1]
    C_mat_2 = C_l_ij(z_i_1,z_f_1,z_i_2,z_f_2, Omega_m-alpha , sig_8 , b_0 , sig_z , z_0)[1]
    
    mat_der = (C_mat_1 - C_mat_2)/(2.0*alpha)
    return mat_der
    
    
    
#===================================================================================
    
def sigma_der_C_l_ij(z_i_1,z_f_1,z_i_2,z_f_2, Omega_m , sig_8, b_0 , sig_z , z_0):
    """
    Function that calculates the derivative of C_l with respect to sigma_8 between two bins 
    -----------------
    Inputs:
    z_i_1 : Lower limit of the first bin
    z_f_1 : Upper limit of the first bin
    z_i_2 : Lower limit of the second bin
    z_f_2 : Upper limit of the second bin
    Omega_m: Omega matter
    sig_8: Sigma_8 parameter
    sig_z : photo-z error
    b_0: present day bias
    z_0 : paramerer related to the mean redshift of the sample
    ---------------
    Returns:
    derivative w/r to matter of C_l betwenn two bins, i and j
    """
    
    alpha = sig_8/10.0
    
    C_sig_1 = C_l_ij(z_i_1,z_f_1,z_i_2,z_f_2, Omega_m, sig_8+alpha , b_0 , sig_z , z_0)[1]
    C_sig_2 = C_l_ij(z_i_1,z_f_1,z_i_2,z_f_2, Omega_m , sig_8-alpha , b_0 , sig_z , z_0)[1]
    
    sig_der = (C_sig_1 - C_sig_2)/(2.0*alpha)
    return sig_der

#======================================================================================
#======================================================================================


def Fisher_calculator(z_init, z_end, N_bins, f_sky, b_0, sig_z, n_tot, z_0):
    """
    Calculates the Fisher Matrix for a number of bins:
    --------------------------------------------------
    Inputs:
    z_i = lower redshift limit of the distribution
    z_f = upper redshift limit of the distribution
    N_bins = number of bins
    f_sky = fraction of the sky of the survey
    sig_z = redshift error
    n_tot = number density of galaxies in gal/arcmin^2
    z_0 = mean redshift of the distribution
    
    -------------------------------------------------
    Returns:
    The full fisher matrix including cross-correlations
    """
    #==============================================================================
    #Cosmological parameters 
    Omega_m = 0.31
    sigma_8 = 0.81
    h = 0.675
    
    # Set cosmology - for later calculations
    # Set cosmology - for later calculations
    cosmo = camb.CAMBparams()
    cosmo.set_cosmology(H0=67.5, ombh2=0.049*(h**2.0), omch2=(Omega_m - 0.049)*(h**2.0), mnu=0.06, omk=0, tau=0.06)
    backres = camb.get_background(cosmo)
    #==============================================================================
    #arcmins^2 to steradians converter
    #Converts the total number density of galaxies in gal/arcmin^2 to gal/steradian

    arcmin_to_ster = 8.4616e-8
    n_tot_ster = n_tot/arcmin_to_ster #Total number density in galaxies/steradian
    #==================================================================
    #=============================================================
    #Redshift range for calculations and integration
    nz = 1000 #number of steps to use for the radial/redshift integration

    zarray = np.linspace(0,4.0,nz)
    dzarray = (zarray[2:]-zarray[:-2])/2.0
    zarray = zarray[1:-1]
    #==================================================================
    #In this module, find the redshift limits of the bins
    
    delta_z = (z_end -z_init)/N_bins # Width of each bin
    
    z_lims = np.zeros(N_bins+1) #An empty array that will contain the upper and lower limits of the bins
                                #The number of limits is equals the number of bins plus one
    
    # Now populate the array of the redshift limits
    for i in range (0,N_bins+1):
        z_lims[i] = z_init + i*delta_z
    
    #===================================================================
    #===================================================================
    # Now we will create 3D matrices where we are going to store C_l^{ij}, C_l,omega_m^{ij}, C_l,sig_8^{ij}
    
    #Initialize matrices 
    l_size = 50   #Size in \ell - this can change
    
    
    C_l_matr1 = np.zeros([N_bins,N_bins,l_size])  #Contains C_ell's
    C_l_omeg1 = np.zeros([N_bins,N_bins,l_size])  #Contains the derivatives of C_ell's with respect to matter density
    C_l_sig1  = np.zeros([N_bins,N_bins,l_size])  #Contains the derivatives of C_ell's with respect to sig_8 
    
    
    #Now let's populate them - run in all combinations of i, j (the different possible bins)
    #But remember that the matrices are symmetric 
    
    for i in range(0,N_bins):
        # Now we have to calculate the limits of the two bins
        #Limits of the first -i- bin
        z_init_i = z_lims[i]
        z_final_i = z_lims[i+1]
        for j in range(i,N_bins):
            #Limits of the second -j-bin
            z_init_j = z_lims[j]
            z_final_j = z_lims[j+1]
            #======================================================
            if (i==j):
                #Calculate the poisson noise - number of galaxies/steradian in the diagonal 
                n_bin = n_tot_ster*W_z_clust(zarray, dzarray, z_init_i, z_final_i, sig_z, b_0, z_0 )[1]
                C_l_matr1[i,i,:] = C_l_ij(z_init_i,z_final_i,z_init_i,z_final_i, Omega_m , sigma_8 , b_0 , sig_z , z_0)[1]+ 1.0/n_bin
                C_l_omeg1[i,i,:] = matter_der_C_l_ij(z_init_i,z_final_i,z_init_i,z_final_i, Omega_m , sigma_8, b_0 , sig_z , z_0)
                C_l_sig1[i,i,:] = sigma_der_C_l_ij(z_init_i,z_final_i,z_init_i,z_final_i, Omega_m , sigma_8, b_0 , sig_z , z_0)
            else:
                C_l_matr1[i,j,:] = C_l_matr1[j,i,:] = C_l_ij(z_init_i,z_final_i,z_init_j,z_final_j, Omega_m , sigma_8, b_0, sig_z, z_0)[1]
                C_l_omeg1[i,j,:] = C_l_omeg1[j,i,:] = matter_der_C_l_ij(z_init_i,z_final_i,z_init_j,z_final_j, Omega_m , sigma_8, b_0 , sig_z , z_0)
                C_l_sig1[i,j,:] = C_l_sig1[j,i,:] = sigma_der_C_l_ij(z_init_i,z_final_i,z_init_j,z_final_j, Omega_m , sigma_8, b_0 , sig_z , z_0)
                #C_l_sig1[i,j,:] = C_l_sig1[j,i,:] = 0.0
                #C_l_omeg1[i,j,:] = C_l_omeg1[j,i,:] = 0.0
                #C_l_matr1[i,j,:] = C_l_matr1[j,i,:] = 0.0
                
            
     
    #==============================================================================================
    #==============================================================================================
    
    ell_lin = np.linspace(1.0, 3.0, 50, dtype = np.float64)
    
    
    ls = np.arange(10,1000, dtype=np.float64)
    
    C_l_matr = np.zeros([N_bins,N_bins,np.size(ls)])  #Contains C_ell's
    C_l_omeg = np.zeros([N_bins,N_bins,np.size(ls)])  #Contains the derivatives of C_ell's with respect to matter density
    C_l_sig  = np.zeros([N_bins,N_bins,np.size(ls)])  #Contains the derivatives of C_ell's with respect to sig_8 
    
    
    
    for i in range(0,N_bins):
        for j in range(i,N_bins):
            for s in range(0,np.size(C_l_omeg1)):
                if (np.sign(C_l_omeg1[i,j,s])>=0.0):
                    l_break = 0.5*(ell_lin[s]+ell_lin[s-1])
                    break
            
            C_l_matr_interp  = UnivariateSpline(ell_lin, np.log10(C_l_matr1[i,j,:]+ 1.0e-20))
            C_omeg_interp = UnivariateSpline(ell_lin, np.log10(abs(C_l_omeg1[i,j,:]+1.0e-20)))
            C_sig_interp = UnivariateSpline(ell_lin, np.log10(C_l_sig1[i,j,:]+1.0e-20))
            for k, l in enumerate(ls):
                ell = np.log10(float(l))
                C_l_matr[i,j,k] = C_l_matr[j,i,k] = 10.0**(C_l_matr_interp(ell))
                C_l_sig[i,j,k] = C_l_sig[j,i,k] = 10.0**(C_sig_interp(ell))
                if (ell < l_break):
                    C_l_omeg[i,j,k] = C_l_omeg[j,i,k] = -(10.0**C_omeg_interp(ell))
                else: 
                    C_l_omeg[i,j,k] = C_l_omeg[j,i,k] = (10.0**C_omeg_interp(ell))
              
                
    #========================================================================================
    #Create array of the l_max for each bin
    
    k_cutoff = 0.2     #Cutoff scale in h Mpc^{-1}
    # Empty array that will contain the l_max of each bin 
    l_max = np.zeros(N_bins+1)   
    #Populate now
    for i in range(0,N_bins+1):
        if (i>0):
            z_low = z_lims[i-1]      #Lower redshift limit of the bin
            z_up = z_lims[i]         #Upper redshift limit of the bin
            z_m = 0.5*(z_low + z_up)  #The mean redshift limit of the bin
            chi_m = backres.comoving_radial_distance(z_m)/h  # comoving distance corresponding to the mean redshift of the bin 
            l_max[i] = int(round(k_cutoff*chi_m))
        else:
            l_max[i] = 9
    #==============================================================================================
    #==============================================================================================
    
    #Now we are ready to calculate the elements of the Fisher matrix
    Fish = np.zeros([2,2]) #initialize
    
    # 0 = matter, 1 = sigma_8
    
    for i in range(0, N_bins):
        low_ell = int(l_max[i]) - 9          #lower \ell_limit
        upper_ell = int(l_max[i+1]) - 9       #upper \ell_limit
        #------------------------------------------------------
        #Define the parts of the three matrices in for the bins into consideration
        ls_loc = ls[low_ell:upper_ell]
        C_l_loc = C_l_matr[i:,i:,low_ell:upper_ell]
        C_l_om_loc = C_l_omeg[i:,i:,low_ell:upper_ell]
        C_l_sig_loc = C_l_sig[i:,i:,low_ell:upper_ell]
        #------------------------------------------------------
        #Calculate the inverse of C_l_loc
    
        C_l_inv_loc = np.zeros_like(C_l_loc)
        for k in range(0, np.size(ls_loc)):
            C_l_inv_loc[:,:,k] = np.linalg.inv(C_l_loc[:,:,k])
    
        #-------------------------------------------------------
        #------------------------------------------------------
        #Now we have everything - we can calculate Fisher elements
        
        Fish[0,0] += sum((2.0*ls_loc+1.0)*np.trace(C_l_inv_loc*C_l_om_loc*C_l_inv_loc*C_l_om_loc))
        Fish[1,1] += sum((2.0*ls_loc+1.0)*np.trace(C_l_inv_loc*C_l_sig_loc*C_l_inv_loc*C_l_sig_loc))
        Fish[0,1] += sum((2.0*ls_loc+1.0)*np.trace(C_l_inv_loc*C_l_om_loc*C_l_inv_loc*C_l_sig_loc))
        
        
    Fish[1,0] = Fish[0,1]
    Fish = (0.5*f_sky)*Fish
    
    return Fish


