#!/usr/bin/python3
# Functions to calculate stiffness matrix and critical speed of railroad subgrades
# Based on Moving_Bell_Load.m by Eduardo Kausel, MIT, from https://doi.org/10.1016/j.soildyn.2018.09.003 ref 14

# % Material properties (size of arrays = number of layers)
# %********************************************************
# Cs   = [200 141 200];   % Shear wave velocity [m/s]
# Rho  = [1 1 1]*2000;    % Mass density of layers (kg/m**3)
# Pois = [1 1 1]*0.25;    % Poisson's ratio
# H    = [2  3 inf];      % depth of layers [m]
# % If the depth of the last layer is infinite, then it is a half-space,
# % otherwise it is a stratum (layer over fixed base)
# %********************************************************
#
# % Related material properties
# Cp = Cs.*sqrt((2-2*Pois)/(1-2*Pois)); % P-wave velocity
# % The following is a close approximation to the speed of Rayleigh waves
# Cr = Cs.*(0.874+0.197*Pois-0.056*Pois.**2-0.0276*Pois.*3); % Rayleigh speed
# nl = len(Cs);             % Number of layers (materials)
# nl2 = 2*nl;                  % double the number of layers

import numpy
import scipy.optimize

def GlobalStiffMat(layers, k, w):
# Assembles the stiffness matrix for the layered soil
# outputs KGsh, KGsvp

    # Check for valid data
    nCp = len(layers['Cp'])
    for el in layers.values():
        if len(el) != nCp:
            print('Error: properties dimension mismatch')
            return

    # Initialize stiffness matrices for s-wave horizontal and s-wave vertical/p-wave stiffness matrices
    KGsh = numpy.zeros((k.shape[0],nCp, nCp), dtype=numpy.complex128)
    KGsvp = numpy.zeros((k.shape[0],2*nCp, 2*nCp), dtype=numpy.complex128)

    # Assemble the top layers of the matrices
    for iLayer in range(0,nCp-1):
        Ksh, Ksvp = StiffElement(layers['Cp'][iLayer], layers['Cs'][iLayer], layers['Dp'][iLayer], layers['Ds'][iLayer], layers['rho'][iLayer], layers['h'][iLayer], k, w)
        KGsh[...,iLayer:iLayer+2, iLayer:iLayer+2] = KGsh[...,iLayer:iLayer+2, iLayer:iLayer+2] + Ksh
        KGsvp[...,iLayer*2:(iLayer+2)*2, iLayer*2:(iLayer+2)*2] = KGsvp[...,iLayer*2:(iLayer+2)*2, iLayer*2:(iLayer+2)*2] + Ksvp

    # Assemble the bottom layer of the matrix.  Assumes the bottom is mounted on a perfectly rigid and static base.
    Ksh, Ksvp = StiffElement(layers['Cp'][nCp-1], layers['Cs'][nCp-1], layers['Dp'][nCp-1], layers['Ds'][nCp-1], layers['rho'][nCp-1], layers['h'][nCp-1], k, w)
    KGsh[...,-1, -1] = KGsh[...,-1, -1] + Ksh[...,0,0]
    KGsvp[...,2*nCp-2:, 2*nCp-2:] = KGsvp[...,2*nCp-2:, 2*nCp-2:] + Ksvp[...,0:2, 0:2]
    return KGsh, KGsvp


def StiffElement(Cp,Cs,Dp,Ds,rho,h,kr,w):
    # Computes the s-wave horizontal and s-wave vertical/p-wave stiffness matrices for a homogeneous layer
    # Cp - primary seismic-wave velocity (m/s)
    # Cs - secondary seismic-wave velocity (m/s)
    # Dp - longitudinal damping constant
    # Ds - shear damping constant
    # rho - density (kg/m**3)
    # h - layer height (m)
    # kr - wavenumber magnitude (per meter)
    # w - angular frequency (per second)
    # % Same sign convention as in Kausel, Elastodynamics
    k = numpy.absolute(kr)
    wr = numpy.real(w)
    wi = numpy.imag(w)
    w = numpy.absolute(wr) + 1.j * wi   # Force into the positive-positive quarter-plane and cast as complex
    Cp = Cp * numpy.sqrt(1+2.j*Dp*numpy.sign(wr))
    Cs = Cs * numpy.sqrt(1+2.j*Ds*numpy.sign(wr))
    a2 = (Cs/Cp)**2
    G = Cs**2*rho
    kp = numpy.sqrt(k**2-(w/Cp)**2) # Eigenvectors of vertical wavenumber.  Eq (7)
    kp = numpy.absolute(numpy.real(kp)) + 1.j * numpy.imag(kp)
    ks = numpy.sqrt(k**2-(w/Cs)**2) # Eq (7)
    ks = numpy.absolute(numpy.real(ks)) + 1.j * numpy.imag(ks)
    p = numpy.sqrt(1-(w /(k*Cp))**2) # Alternatively, can be written as p = kp / k
    s = numpy.sqrt(1-(w /(k*Cs))**2)
    p = numpy.absolute(numpy.real(p)) + 1.j * numpy.imag(p)
    s = numpy.absolute(numpy.real(s)) + 1.j * numpy.imag(s)

    if numpy.isinf(h):
        Ksh = ks*G
        Ksvp = 2*k*G*((1-s**2)/(2*(1-p*s))*numpy.block([[p, -1*numpy.ones(p.shape)],[-1*numpy.ones(s.shape), s]]) + numpy.array([[0, 1], [1, 0]]))

    else:
        ksh = ks*h
        kph = kp*h
        Shs = numpy.sinh(ksh)
        Chs = numpy.cosh(ksh)
        Shp = numpy.sinh(kph)
        Chp = numpy.cosh(kph)
        Cths = Chs/Shs
        kh = k*h
        ps = p*s
        Denom = 2*(1-Chp*Chs)+(1/ps+ps)*Shp*Shs
        Ksh = ks*G*numpy.block([[Cths, -1/Shs],[-1/Shs, Cths]]) # Table 1.2 section 1
        S1 = k*G*(1-s**2)
        S2 = k*G*(1+s**2)
        k11 = S1*(Chp * Shs - ps*Shp*Chs)/s
        k12 = S1*(1-Chp*Chs+ps*Shp*Shs)+ S2*Denom
        k22 = S1*(Shp*Chs - ps*Chp*Shs)/p

        K11 = numpy.block([[k11, k12],[ k12, k22]])/Denom # Eq (10.19) of Stiffness Matrix Method for Layered Media
        K22 = numpy.block([[k11,-k12],[-k12, k22]])/Denom
        K12 = S1*numpy.block([[(ps*Shp-Shs)/s, Chp-Chs],[ -(Chp-Chs),  (ps*Shs-Shp)/p]])/Denom
        K21 = numpy.moveaxis(K12,-1,-2) # Apply symmetry of Ksvp to find K21

        Ksvp = numpy.block([[K11, K12],[K21, K22]]) # Eq (18)

    return Ksh, Ksvp

# def StiffPole(w, Cp, Cs, Dp, Ds, rho, h, k, index):
## Function to calculate the poles of the stiffness matrix.  This is a rapid, but not particularly accurate, way of finding the vehicle velocities at which ground shaking becomes excessive.
#     wc = numpy.array((w[0]+1.j*w[1],))
#     dk1 = 0.001
#     dk2 = 0.001 * 1.j
#     KG = GlobalStiffMat(Cp, Cs, Dp, Ds, rho, h, numpy.array((k,k+dk1,k+dk2))[:,numpy.newaxis,numpy.newaxis], wc)[index]
#     G = numpy.linalg.det(KG)
#     dG = (G[1]-G[0])/dk1 + (G[2]-G[0])/dk2
#     return (numpy.real(dG), numpy.imag(dG))

def generate_dispersion(layers,wr,outfile):
    # Finds the real zeros of the stiffness matrix to generate dispersion functions (Raleigh-wave speeds of sound) within the frequency range wr given.  The range of sound speeds to search are hard coded in the definition of k.
    # Recommended, wr = numpy.arange(0.5,100.0*2.0*3.14159,0.5)
    k_zero = [] # Save dispersion as a list, as more than one zero may be found for a given frequency w.
    for w in wr:
        k = numpy.arange( w/200.0,w/35.0, w*1.0e-6) # Search between 35 and 200 m/s
        KG = GlobalStiffMat(layers, k[:,numpy.newaxis,numpy.newaxis], w)

        def find_k_zero(eigenvals):
            # Find the imaginary eigenvalues in E1(k;w) by differencing then running a linear interpolation on the real part.
            angles = numpy.real(eigenvals)
            zero_index = numpy.nonzero( (numpy.sign(angles[1:]) != numpy.sign(angles[:-1])))#  Note, we now assume the first index is over k.
            if not zero_index: return numpy.empty((0,3))

            triplet = numpy.stack((numpy.ones(k.shape[0])*w,k,eigenvals/k/k), axis=1)
            # return triplet[zero_index]  # Simplified output without linear interpolation
            return ((triplet[numpy.add(zero_index,1)[0]]*angles[zero_index, numpy.newaxis] - triplet[zero_index] * angles[numpy.add(zero_index,1)[0],numpy.newaxis])/(angles[zero_index,numpy.newaxis]-angles[numpy.add(zero_index,1)[0],numpy.newaxis]))[0]

        G = numpy.linalg.det(KG[1])
        k_zero.append(find_k_zero(G))

        # Find dispersion
    outfile.write('#Surface Raleigh waves\n')
    numpy.savetxt(outfile, numpy.real(numpy.concatenate(k_zero)))
    return

def generate_x_surface_displacement(layers,source,k,v,outfile):
# Calculate the surface displacement and strain at layer boundaries from a spacially distributed vertical impulse moving at speed v (m/s).
    w = k*v
    i = layers['i'] # Index of layer in which to calculate the strain
    calc_index = numpy.nonzero(source[:,1])[0] # Avoid calculating stiffness matrices over wavenumbers with no source term, saving memory and calculation time.

    Svp = GlobalStiffMat(layers,k[calc_index,numpy.newaxis,numpy.newaxis],w[calc_index,numpy.newaxis,numpy.newaxis])[1]
    Fvp = numpy.linalg.inv(Svp)
    disp_k1 = numpy.matmul(Fvp, source[calc_index,:,numpy.newaxis])

    kshape = numpy.array(disp_k1.shape)
    kshape[0] = k.shape[0]
    disp_k = numpy.zeros(kshape, dtype=numpy.complex128)
    disp_k[calc_index] = disp_k1
    p = numpy.sqrt(1+0.j-(v/layers['Cp'][i]))**2 # Alternatively, kp / k
    s = numpy.sqrt(1+0.j-(v/layers['Cs'][i]))**2
    s12 = 1+s**2
    QRinv = numpy.block([[[4*p**2-s12**2,2*(p-s)*s12],[-2*(p-s)*s12,-(1-s**2)**2]]])*k[calc_index,numpy.newaxis,numpy.newaxis]/(1-s*p)/2 # Matrix to obtain xz and zz strain from displacement.
    strain_k1 = numpy.matmul(QRinv,disp_k1[:,2*i:2*i+2])
    disp_x = numpy.fft.ifft(disp_k, axis=0)*0.25
    # Note, the 0.25 normalization is implemented following cross-checks with calculations from https://doi.org/10.1016/j.conbuildmat.2022.127485  I have not yet found where this discrepancy comes from.
    kshape[1]=2

    strain_k = numpy.zeros(kshape, dtype=numpy.complex128)
    strain_k[calc_index] = strain_k1 # Recast to fill in the zero values of the strain matrix

    strain_x = numpy.fft.ifft(strain_k, axis=0)*0.25
    x = numpy.arange(k.shape[0])/2/numpy.pi/k[1]
    numpy.savetxt(outfile,numpy.real(numpy.concatenate([x[:,numpy.newaxis],disp_x[:,:,0],strain_x[:,0,0,numpy.newaxis]], axis=1)))
    return


if __name__ == '__main__':

    # Shear wave speed in clay as function of depth from https://crrrc.ca/_documents/Locat_report_160608_final.pdf
    # Assume primary wave speed of 1500 m/s (that of water) in all sediment below 2m (water table).
    # Assume 600 m/s shear wave in till
    # https://pburnley.faculty.unlv.edu/GEOL452_652/seismology/notes/SeismicNotes10RVel.html
    # Assume 160 m/s shear wave in sand https://www.ngi.no/globalassets/bilder/prosjekter/ngts/ngts-rapporter/holten-johannes-gaspar.-project-thesis-2016.pdf
    # Assume top layers of ballast and embankment as     Cp = [346,244,] Cs = [200,141,]     Dp = [0.005,0.005,] Ds = [0.005,0.005,] rho = [2000,2700,] h = [0.6,2,]

    # Semi-infinite thickness of Leda clay
    thick_Leda = {'Cp': [346.0,244.0,1500.0,1500.0],
        'Cs' : [200.0,141.0,70.0,100.0],
        'Dp' : [0.005,0.005,0.002,0.002],
        'Ds' : [0.005,0.005,0.002,0.002],
        'rho' : [2000.0,2000.0,2000.0,2000.0],
        'h' : [0.4,0.3,8.0,numpy.inf],
        'i' : 2}

    # Semi-infinite thickness of Leda clay with a degraded strength at depths more shallow than 3.7m.
    degraded_Leda = {'Cp': [346.0,244.0,1500.0,1500.0,1500.0],
        'Cs' : [200.0,141.0,50.0,70.0,100.0],
        'Dp' : [0.005,0.005,0.002,0.002,0.002],
        'Ds' : [0.005,0.005,0.002,0.002,0.002],
        'rho' : [2000.0,2000.0,2000.0,2000.0,2000.0],
        'h' : [0.4,0.3,3.0,5.0,numpy.inf],
        'i' : 2}

    # Semi-infinite depth of clay as measured at the south-gloucester site - doi:10.3934/geosci.2019.3.390
    # Assumed 1m clay crust on surface
    south_gloucester = {'Cp': [200.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0],
        'Cs' : [120.0,75.0,60.0,79.0,79.0,95.0,120.0,135.0,170.0],
        'Dp' : [0.005,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'Ds' : [0.005,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'rho' : [1400.0,1670.0,1570.0,1700.0,1550.0,1550.0,1580.0,1710.0,2000.0],
        'h' : [1.0,2.0,2.0,3.0,2.0,2.0,3.0,3.0,numpy.inf],
        'i' : 2}

    # Semi-infinite depth of clay as measured at the Beauharnois site - http://dx.doi.org/10.1139/cgj-2020-0254
    # Assumed 1m clay crust on surface
    beauharnois = {'Cp': [163.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0],
        'Cs' : [100.0,75.0,79.0,83.0,89.0,95.0,100.0,104.0,110.0],
        'Dp' : [0.005,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'Ds' : [0.005,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'rho' : [1850.0,1620.0,1640.0,1690.0,1750.0,1800.0,1850.0,1900.0,1940.0],
        'h' : [1.0,5.0,2.0,1.0,1.0,2.0,2.0,4.0,numpy.inf],
        'i' : 1}

    beauharnois_UIC60 = {'Cp': [4720.0,1348.0,300.0,163.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0],
        'Cs' : [2550.0,46.2,197.0,100.0,75.0,79.0,83.0,89.0,95.0,100.0,104.0,110.0],
        'Dp' : [0.0001,0.1,0.05,0.005,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'Ds' : [0.0001,0.1,0.05,0.005,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'rho' : [7700.0,1100.0,2000.0,1850.0,1620.0,1640.0,1690.0,1750.0,1800.0,1850.0,1900.0,1940.0],
        'h' : [0.0595,0.121,0.6,0.6,5.0,2.0,1.0,1.0,2.0,2.0,4.0,numpy.inf],
        'i' : 4} # Equivalent plate thickness to UIC rail assuming load is spread across width of 2.59m long rail tie.  Convert by taking I*12*(1+nu^2)/l where I is the moment of inertia of both rails (2 * 3038 cm^4), nu is Poisson's ratio (0.5), and l is the length of tie
    # Rubber pads in second layer, 50 MN/m every .67 metres stiffness under each rail, shore hardness 75, 7.05 MPa, assume 2 GPa bulk modulus, giving sound speeds.

    # Same as beauharnois_UIC60 with clay of reduced strength due to shear strain at depths of up 2.2 m.  Stiffness reduced per dynamic loading curves from http://dx.doi.org/10.1139/cgj-2020-0254.
    beauharnois_UIC60_degraded = {'Cp': [4720.0,1348.0,300.0,163.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0],
        'Cs' : [2550.0,46.2,197.0,100.0,62.7,75.0,79.0,83.0,89.0,95.0,100.0,104.0,110.0],
        'Dp' : [0.0001,0.1,0.05,0.005,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'Ds' : [0.0001,0.1,0.05,0.005,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'rho' : [7700.0,1100.0,2000.0,1850.0,1620.0,1620.0,1640.0,1690.0,1750.0,1800.0,1850.0,1900.0,1940.0],
        'h' : [0.0595,0.121,0.6,0.6,1.0,4.0,2.0,1.0,1.0,2.0,2.0,4.0,numpy.inf],
        'i' : 4}

    # Same as beauharnois_UIC60_degraded with clay laying on stiff bedrock at a depth of 10m.
    beauharnois_UIC60_degraded10m = {'Cp': [4720.0,1348.0,300.0,163.0,1500.0,1500.0,1500.0,1500.0,1500.0],
        'Cs' : [2550.0,46.2,197.0,100.0,62.7,75.0,79.0,83.0,89.0],
        'Dp' : [0.0001,0.1,0.05,0.005,0.002,0.002,0.002,0.002,0.002],
        'Ds' : [0.0001,0.1,0.05,0.005,0.002,0.002,0.002,0.002,0.002],
        'rho' : [7700.0,1100.0,2000.0,1850.0,1620.0,1620.0,1640.0,1690.0,1750.0],
        'h' : [0.0595,0.121,0.6,0.6,1.0,4.0,2.0,1.0,1.0],
        'i' : 4}

    # Same as beauharnois_UIC60_degraded with clay laying on stiff bedrock at a depth of 8m.
    beauharnois_UIC60_degraded8m = {'Cp': [4720.0,1348.0,300.0,163.0,1500.0,1500.0,1500.0],
        'Cs' : [2550.0,46.2,197.0,100.0,62.7,75.0,79.0],
        'Dp' : [0.0001,0.1,0.05,0.005,0.002,0.002,0.002],
        'Ds' : [0.0001,0.1,0.05,0.005,0.002,0.002,0.002],
        'rho' : [7700.0,1100.0,2000.0,1850.0,1620.0,1620.0,1640.0],
        'h' : [0.0595,0.121,0.6,0.6,1.0,4.0,2.0],
        'i' : 4}

    # Same as beauharnois_UIC60_degraded with clay laying on stiff bedrock at a depth of 6m.
    beauharnois_UIC60_degraded6m = {'Cp': [4720.0,1348.0,300.0,163.0,1500.0,1500.0],
        'Cs' : [2550.0,46.2,197.0,100.0,62.7,75.0],
        'Dp' : [0.0001,0.1,0.05,0.005,0.002,0.002],
        'Ds' : [0.0001,0.1,0.05,0.005,0.002,0.002],
        'rho' : [7700.0,1100.0,2000.0,1850.0,1620.0,1620.0],
        'h' : [0.0595,0.121,0.6,0.6,1.0,4.0],
        'i' : 4}

    # Same as beauharnois_UIC60_degraded with clay laying on stiff bedrock at a depth of 5m.
    beauharnois_UIC60_degraded5m = {'Cp': [4720.0,1348.0,300.0,163.0,1500.0,1500.0],
        'Cs' : [2550.0,46.2,197.0,100.0,62.7,75.0],
        'Dp' : [0.0001,0.1,0.05,0.005,0.002,0.002],
        'Ds' : [0.0001,0.1,0.05,0.005,0.002,0.002],
        'rho' : [7700.0,1100.0,2000.0,1850.0,1620.0,1620.0],
        'h' : [0.0595,0.121,0.6,0.6,1.0,3.0],
        'i' : 4}

    # Same as beauharnois_UIC60_degraded with clay laying on stiff bedrock at a depth of 4m.
    beauharnois_UIC60_degraded4m = {'Cp': [4720.0,1348.0,300.0,163.0,1500.0,1500.0],
        'Cs' : [2550.0,46.2,197.0,100.0,62.7,75.0],
        'Dp' : [0.0001,0.1,0.05,0.005,0.002,0.002],
        'Ds' : [0.0001,0.1,0.05,0.005,0.002,0.002],
        'rho' : [7700.0,1100.0,2000.0,1850.0,1620.0,1620.0],
        'h' : [0.0595,0.121,0.6,0.6,1.0,2.0],
        'i' : 4}

    # Same as beauharnois_UIC60_degraded with clay laying on stiff bedrock at a depth of 3m.
    beauharnois_UIC60_degraded3m = {'Cp': [4720.0,1348.0,300.0,163.0,1500.0,1500.0],
        'Cs' : [2550.0,46.2,197.0,100.0,62.7,75.0],
        'Dp' : [0.0001,0.1,0.05,0.005,0.002,0.002],
        'Ds' : [0.0001,0.1,0.05,0.005,0.002,0.002],
        'rho' : [7700.0,1100.0,2000.0,1850.0,1620.0,1620.0],
        'h' : [0.0595,0.121,0.6,0.6,1.0,1.0],
        'i' : 4}

    # Same as beauharnois_UIC60_degraded with clay laying on stiff bedrock at a depth of 2m.
    beauharnois_UIC60_degraded2m = {'Cp': [4720.0,1348.0,300.0,163.0,1500.0],
        'Cs' : [2550.0,46.2,197.0,100.0,62.7],
        'Dp' : [0.0001,0.1,0.05,0.005,0.002],
        'Ds' : [0.0001,0.1,0.05,0.005,0.002],
        'rho' : [7700.0,1100.0,2000.0,1850.0,1620.0],
        'h' : [0.0595,0.121,0.6,0.6,1.0],
        'i' : 4}

    # Same as beauharnois_UIC60_degraded with increased degradation of the shallow clay layer.
    beauharnois_UIC60_degraded50 = {'Cp': [4720.0,1348.0,300.0,163.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0],
        'Cs' : [2550.0,46.2,197.0,100.0,50.0,75.0,79.0,83.0,89.0,95.0,100.0,104.0,110.0],
        'Dp' : [0.0001,0.1,0.05,0.005,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'Ds' : [0.0001,0.1,0.05,0.005,0.07,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'rho' : [7700.0,1100.0,2000.0,1850.0,1620.0,1620.0,1640.0,1690.0,1750.0,1800.0,1850.0,1900.0,1940.0],
        'h' : [0.0595,0.121,0.6,0.6,1.0,4.0,2.0,1.0,1.0,2.0,2.0,4.0,numpy.inf],
        'i' : 4}

    # Same as beauharnois_UIC60_degraded with increased degradation of the shallow clay layer.
    beauharnois_UIC60_degraded40 = {'Cp': [4720.0,1348.0,300.0,163.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0],
        'Cs' : [2550.0,46.2,197.0,100.0,40.0,75.0,79.0,83.0,89.0,95.0,100.0,104.0,110.0],
        'Dp' : [0.0001,0.1,0.05,0.005,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'Ds' : [0.0001,0.1,0.05,0.005,0.12,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'rho' : [7700.0,1100.0,2000.0,1850.0,1620.0,1620.0,1640.0,1690.0,1750.0,1800.0,1850.0,1900.0,1940.0],
        'h' : [0.0595,0.121,0.6,0.6,1.0,4.0,2.0,1.0,1.0,2.0,2.0,4.0,numpy.inf],
        'i' : 4}

    # Same as beauharnois_UIC60_degraded with significantly increased degradation of the shallow clay layer.
    beauharnois_UIC60_degraded30 = {'Cp': [4720.0,1348.0,300.0,163.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0],
        'Cs' : [2550.0,46.2,197.0,100.0,30.0,75.0,79.0,83.0,89.0,95.0,100.0,104.0,110.0],
        'Dp' : [0.0001,0.1,0.05,0.005,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'Ds' : [0.0001,0.1,0.05,0.005,0.15,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'rho' : [7700.0,1100.0,2000.0,1850.0,1620.0,1620.0,1640.0,1690.0,1750.0,1800.0,1850.0,1900.0,1940.0],
        'h' : [0.0595,0.121,0.6,0.6,1.0,4.0,2.0,1.0,1.0,2.0,2.0,4.0,numpy.inf],
        'i' : 4}

    # Same as beauharnois_UIC60_degraded with significantly increased degradation of the shallow clay layer.
    beauharnois_UIC60_degraded20 = {'Cp': [4720.0,1348.0,300.0,163.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0],
        'Cs' : [2550.0,46.2,197.0,100.0,20.0,75.0,79.0,83.0,89.0,95.0,100.0,104.0,110.0],
        'Dp' : [0.0001,0.1,0.05,0.005,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'Ds' : [0.0001,0.1,0.05,0.005,0.20,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'rho' : [7700.0,1100.0,2000.0,1850.0,1620.0,1620.0,1640.0,1690.0,1750.0,1800.0,1850.0,1900.0,1940.0],
        'h' : [0.0595,0.121,0.6,0.6,1.0,4.0,2.0,1.0,1.0,2.0,2.0,4.0,numpy.inf],
        'i' : 4}

    # Same as beauharnois_UIC60 with increased effective thickness of the top surface represention north american 136lb/yd rail.
    beauharnois_136lb = {'Cp': [4720.0,1348.0,300.0,163.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0],
        'Cs' : [2550.0,46.2,197.0,100.0,75.0,79.0,83.0,89.0,95.0,100.0,104.0,110.0],
        'Dp' : [0.0001,0.1,0.05,0.005,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'Ds' : [0.0001,0.1,0.05,0.005,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'rho' : [7700.0,1100.0,2000.0,1850.0,1620.0,1640.0,1690.0,1750.0,1800.0,1850.0,1900.0,1940.0],
        'h' : [0.0650,0.121,0.6,0.6,5.0,2.0,1.0,1.0,2.0,2.0,4.0,numpy.inf],
        'i' : 4}

    # Same as beauharnois_UIC60 with ballast layers replaced by a model of slab track per https://doi.org/10.1016/j.conbuildmat.2022.127485.
    beauharnois_UIC60_slab = {'Cp': [4720.0,1348.0,5000.0,3000.0,163.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0,1500.0],
        'Cs' : [2550.0,46.2,3230.0,881.0,100.0,75.0,79.0,83.0,89.0,95.0,100.0,104.0,110.0],
        'Dp' : [0.0001,0.1,0.002,0.002,0.005,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'Ds' : [0.0001,0.1,0.002,0.002,0.005,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002],
        'rho' : [7700.0,1100.0,2300.0,2300.0,1850.0,1620.0,1640.0,1690.0,1750.0,1800.0,1850.0,1900.0,1940.0],
        'h' : [0.0595,0.121,0.2,0.3,0.5,5.0,2.0,1.0,1.0,2.0,2.0,4.0,numpy.inf],
        'i' : 5}

    layers = beauharnois_UIC60_degraded10m
    outfile = open('beauharnois_UIC60_degraded10m.dat','w')
    # generate_dispersion(layers,numpy.arange(0.5,100.0*2.0*3.14159,0.5),outfile)
    k = numpy.concatenate([numpy.arange(0,5.1,0.333),numpy.arange(-0.333,-5.1,-0.333)])
    source = numpy.zeros((k.shape[0],2*len(layers['Cp'])), dtype=numpy.complex128)
    # source[1:k.shape[0]//2,1] = 34000*9.8/2/3.14159/2*numpy.exp(-k[1:k.shape[0]//2]**2/2/0.5**2) # 34 ton bogie approximated as a 2m width gaussian every 6 pi = 18.8 meters.  Gaussian assumed to model effects of rail rigidity combined with 3m distance between axles.  Do not use with soil models including an upper layer of steel.
    source[1:k.shape[0]//2,1] = 17000*9.8/2.59*(numpy.exp(3.0j*k[1:k.shape[0]//2])+1.0) # For use with layers including a steel rail in the top layer.   Two 17 ton bogies spaced 3 m apart every 18.8 meters with 2.59 m wide stiff ties.

    outfile.write('\n\n')
    for v in numpy.arange(20.0,101.0,2.0):
        outfile.write('# At speed %f m/s \n' % v)
        generate_x_surface_displacement(layers,source,k,v,outfile)
        outfile.write('\n\n')
    outfile.close()

