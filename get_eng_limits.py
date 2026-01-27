#!/usr/bin/python3
# Script to find the maximum strain and surface displacement caused by the passage of an HSR train.

import numpy
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: get_eng_limits.py infile outfile\n")
        sys.exit(1)

    index = 0
    infile = open(sys.argv[1],'r')
    lines = infile.readlines()
    infile.close()

    velocities = []
    for line in lines:
        if line and line[0] == '#':
            velocities.append(line[11:20]) # Hardcode position of velocity information to retrieve in section header.

    # Retrieve all non-empty non-commented lines as a numpy array
    displacement_arr = numpy.loadtxt([line for line in lines if line and not line[0] == '#'])
    disp_shape = displacement_arr.shape
    displacement_arr = displacement_arr.reshape([len(velocities),disp_shape[0]//len(velocities),disp_shape[1]])

    # Tabulate three effects of a trains passage
    speed_effects = numpy.zeros((len(velocities),4))
    speed_effects[:,0] = velocities

    # Find the maximum ground deplacement as a function of velocity in x and z
    speed_effects[:,1] = numpy.sqrt(numpy.max(displacement_arr[:,:,1]**2 + displacement_arr[:,:,2]**2,axis=1))

    # Find the maximum ground acceleration as a function of velocity
    extended_arr = numpy.concatenate((displacement_arr[:,-1,numpy.newaxis,:3],displacement_arr[:,:,:3],displacement_arr[:,1,numpy.newaxis,:3]), axis=1)
    acc_arr = numpy.absolute((2*extended_arr[:,1:-1,1:]-extended_arr[:,0:-2,1:]-extended_arr[:,2:,1:])/(displacement_arr[:,1,0,numpy.newaxis,numpy.newaxis]-displacement_arr[:,0,0,numpy.newaxis,numpy.newaxis])**2)
    speed_effects[:,2] = numpy.max(numpy.sqrt(numpy.absolute(acc_arr[:,:,0]**2+acc_arr[:,:,1]**2)),axis=1) * speed_effects[:,0]**2 # Not implemented: Consider a penalty factor for x acceleration vs. y acceleration

    # Find maximum strain as a function of velocity
    speed_effects[:,3] = numpy.max(numpy.absolute(displacement_arr[:,:,-1]),axis=1)

    # Search through to find the lowest speed at which the peak ground acceleration exceeds 3.5 m/s^2.  If no limit is found, report the maximum speed given in the data.
    max_acc = speed_effects[-1,0]
    for i,a in enumerate(speed_effects[:,2]):
        if a > 3.5:
            if i:
                max_acc = speed_effects[i-1,0]
            else:
                max_acc = 0
            break
    print("Maximum velocity to maintain surface acceleration under 3.5 m/s2 is %f m/s \n" % max_acc)

    # Search through to find the lowest speed at which the peak ground strain exceeds 1.5e-4
    max_strain = speed_effects[-1,0]
    for i,a in enumerate(speed_effects[:,3]):
        if a > 1.5e-4:
            if i:
                max_strain = speed_effects[i-1,0]
            else:
                max_strain = 0
            break
    print("Maximum velocity to maintain shear under 1.5e-4 is %f m/s \n" % max_strain)

    numpy.savetxt(sys.argv[2],speed_effects)

    sys.exit(0)
