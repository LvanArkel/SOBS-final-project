# -*- coding: utf-8 -*-
"""
This file contains the segdyn algorithm described in Casius et al. 2004, and 
some useful helper functions to quickly assess the output. No need to change 
this file, unless you absolutely know what you're doing! 

This work is largely based on matlab implementation of similar algorithm 
developed by Knoek van Soest. 

Author: Koen Lemaire (k.k.lemaire2@vu.nl)
"""
#%% load modules required in segdyn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%% segdyn; the main algorithm for solving equations of motion

def segdyn(segdynstate,segparms,V,Acons=None,Bcons=None):
    """
    segdynstated, Vnew = segdyn(segdynstate,segparms,V,Acons=None,Bcons=None)
    
    Implementation of 2D Newton-Euler dynamics for nseg segments connected in hinge 
    joints. Based on Casius, Bobbert and van Soest (2004) "Forward Dynamics of 
    Two-Dimensional Skeletal Models. A Newton-Euler Approach". Mostly translated from 
    a matlab implementation written by Knoek van Soest.
    
    Note: this function and its helper functions defined elsewhere deal only
    with numpy arrays. The functions might produce (correct) output for other
    types of input, but this is generally not guaranteed. 
    
    
    Simplified version: - requires exactly 3*nseg unknowns (one more for each constraint
                          equation), i.e., least squares solutions are not supported
                        - the complete system of equations is set up each time this
                          function is invoked, i.e., very inefficient
                          
    INPUT 

    state:    vector containing 2*nseg+4 state variables: 
                    [phi_1 .. phi_n, phid_1 .. phid_n, xb, yb, xbd, ybd], where 
              phi_i is ith segment angle, phid_i the ith segment angular velocity, 
              (xb, yb) the base position, and (xbd, ybd) the base velocity

    segparms: dictionary with fields: 
              L=[l_1 .. l_n], d=[d_1 .. d_n], m=[m_1 .. m_n], and J=[J_1 .. J_n], where
              L_i is the length of the ith segment (m), d_i is the distance from the
              proximal end of the ith segment to its center of gravity (m), m_i is the
              mass of the ith segment (kg), and J_i the moment of inertia of the ith
              segment with respect to its center of gravity (Nm/kg) 
              
    V:        vector containg 7*nseg+5 values for the follwing variables:
                  Fxr   = V[0        : nseg+1]    # nseg+1 horizontal joint reaction forces
                  Fyr   = V[nseg+1   : 2*nseg+2]  # nseg+1 vertical joint reaction forces
                  M     = V[2*nseg+2 : 3*nseg+3]  # nseg+1 net joint moments
                  Fxext = V[3*nseg+3 : 4*nseg+3]  # nseg horizontal external forces
                  Fyext = V[4*nseg+3 : 5*nseg+3]  # nseg vertical externalforces
                  Mext  = V[5*nseg+3 : 6*nseg+3]  # nseg external moments
                  phidd = V[6*nseg+3 : 7*nseg+3]  # nseg segment angular accelerations
                  xbdd  = V[7*nseg+3 : 7*nseg+4]  # horizontal base acceleration
                  ybdd  = V[7*nseg+4 : 7*nseg+5]  # vertical base acceleration
              exactly 4*nseg+5 of these variables should contain an 
              appropriate, known value. The other 3*nseg variables must contain 
              np.nan (unknown) value.

    A_cons:   left-hand side of constraint equations (n_cons x 7*nseg+5)
    b_cons:   right-hand side of constraint equations (n_cons,)
    If constraints are added the number of additional unknowns in V must equal 
    the number of constraint equations (rows) added to Acons and Bcons 
   
    OUTPUT 

    stated:   first order derivative of state
    Vnew:     copy of V, with all unknown variables replaced with the value resulting
              from solving the equations of motion
                 
    Author: Koen Lemaire (k.k.lemaire2@vu.nl)  
    """
   
    # parse segment parameters
    nseg=segparms['nseg']
    m=segparms['m']
    L=segparms['L']
    J=segparms['J']
    d=segparms['d']
    
    # assert length of state vector
    assert len(segdynstate)==2*nseg+4, 'segdynstate must be length 2*nseg+4'
    
    # unravel state vector
    phi=segdynstate[0:nseg]
    phid=segdynstate[nseg:2*nseg]
    xb=segdynstate[2*nseg:2*nseg+1]
    yb=segdynstate[2*nseg+1:2*nseg+2]
    xbd=segdynstate[2*nseg+2:2*nseg+3]
    ybd=segdynstate[2*nseg+3:2*nseg+4]
    
    # generate A_star 
    # initialize upper blocks (2*nseg translational EOM):
    B1=np.zeros((2*nseg, 2*nseg+2))
    B2=np.zeros((2*nseg, nseg+1)) # B2 done!
    B3=np.diag(np.ones(2*nseg)) # B3 done!
    B4=np.zeros((2*nseg, nseg)) # B4 done!
    B5=np.zeros((2*nseg, nseg))
    B6=np.zeros((2*nseg, 2))
    # initialize lower blocks (nseg rotational EOM):
    B7=np.zeros((nseg, 2*nseg+2)) 
    B8=np.zeros((nseg, nseg+1)) 
    B9=np.zeros((nseg,2*nseg)) # B9 done!
    B10=np.diag(np.ones(nseg)) # B10 done! 
    B11=np.diag((-J)) # B11 done!
    B12=np.zeros((nseg, 2)) # B12 done!
    
    # compile remaining static blocks:
    # B1
    for i in range(2*nseg):
        if i < nseg:
            B1[i,i:i+2]=[1, -1]
        else:
            B1[i,i+1:i+3]=[1, -1]
    
    # B6
    B6[0:nseg,0]=-m
    B6[nseg:2*nseg,1]=-m
    
    # B8
    for i in range(nseg):
        B8[i,i:i+2]=[1, -1]
    
    # If speed of the algorithm is prioritized, everything up to this point 
    # could be moved out of this function into the calling script, i.e. the 
    # complete A_star matrix could be passed to this function and only the 
    # state dependent blocks would be constructed here. We will now compile 
    # these state dependent blocks 5 and 7, which contain the Moments of the 
    # reaction forces (cross-products) and centre of mass accelerations, 
    # respectively.
    
    # helper variables:
    p=L-d
    phidsqr=phid**2
    sinphi=np.sin(phi)
    cosphi=np.cos(phi)
    dsin=d*sinphi
    dcos=d*cosphi
    psin=p*sinphi
    pcos=p*cosphi
    lsin=L*sinphi
    lcos=L*cosphi    
    
    # B5    
    for i in range(nseg):
        for j in range(i):
            B5[i,j]      =  m[i]*lsin[j]
            B5[i+nseg,j] = -m[i]*lcos[j]
        B5[i,i]       =  m[i]*dsin[i]
        B5[i+nseg,i]  = -m[i]*dcos[i]
            
    # B7
    for i in range(nseg):
        B7[i,i:i+2]              = [ dsin[i],  psin[i]]
        B7[i,i+nseg+1:i+nseg+3]  = [-dcos[i], -pcos[i]]
    
    # compile matrix A_star; putting all blocks together
    B1_6=np.hstack((B1, B2, B3, B4, B5, B6))
    B7_12=np.hstack((B7, B8, B9, B10, B11, B12))
    A_star=np.vstack((B1_6,B7_12))
    
    # Compile b_star. Initially, b_star contains only the angular velocity 
    # terms from the segcom accelerations:
    b_star=np.zeros(3*nseg) # also possible to initiate b here
    for i in range(nseg):
        for j in range(i):
            b_star[i]      = b_star[i]      - m[i]*lcos[j]*phidsqr[j]
            b_star[i+nseg] = b_star[i+nseg] - m[i]*lsin[j]*phidsqr[j]
        b_star[i]      = b_star[i]      - m[i]*dcos[i]*phidsqr[i]
        b_star[i+nseg] = b_star[i+nseg] - m[i]*dsin[i]*phidsqr[i]
    
    # add constraint equations if given (simply stack them underneath) 
    if (Acons is not None):
        A_star = np.vstack((A_star,Acons))
        b_star = np.concatenate((b_star,Bcons))
    
    # compile Ax=b; 
    idx_nan=np.isnan(V) # bolean array of truth values; unknowns are True here
    
    # assert size of A_star, b_star and nr of unknowns in V
    nrow,ncolumn=A_star.shape
    assert ncolumn == 7*nseg+5, 'nr of columns in A_star must be 7nseg+5'
    assert sum(idx_nan) == nrow, 'nr of rows in A_star must match nr of unknowns in V'
    assert sum(idx_nan) == len(b_star), 'nr of elements in b_star must match nr of unknowns in V'
    
    # make A matrix; equal to columns of A_star corresponding to unknowns in V
    A=A_star[:,idx_nan]

    # Make b vector: substract columns of A_star corresponding to known entries
    # in V times the known values in V from b_star
    b=b_star-A_star[:,~idx_nan]@V[~idx_nan] 

    # solve the now square system:
    x=np.linalg.solve(A, b)
    
    # Vnew: equal to V but with all unknowns replaced with solution of Ax=b
    Vnew=V
    Vnew[idx_nan]=x 
    
    # compile state derivative
    segdynstated=np.zeros(len(segdynstate)) # initialize    
    segdynstated[0:nseg]=phid # from previous
    segdynstated[nseg:2*nseg]=Vnew[6*nseg+3 : 7*nseg+3] # nseg segment angular accelerations
    segdynstated[2*nseg:2*nseg+1]=xbd # from previous
    segdynstated[2*nseg+1:2*nseg+2]=ybd # from previous
    segdynstated[2*nseg+2:2*nseg+3]=Vnew[7*nseg+3 : 7*nseg+4] # horizontal base acceleration
    segdynstated[2*nseg+3:2*nseg+4]=Vnew[7*nseg+4 : 7*nseg+5] # vertical base acceleration
    
    # return output
    return segdynstated, Vnew

#%% total joint coordinates
def jointcoord(segdynstate,segparms,segdynstated=None):
    """
    joint, jointd, jointdd = jointcoord(segdynstate,segparms,segdynstated=None)
    
    Calculates the x and y coordinates and their derivatives of all joints as a 
    function of segdynstate and segdynstated (see segdyn for definition)
      
    INPUT 
    segdynstate:    vector containing segdynstate (see segdyn for definition) 
    segdynstated:    vector containing segdynstated (see segdyn for definition) 
          
    segparms: dictionary with fields: 
              L=[l_1 .. l_n], d=[d_1 .. d_n], m=[m_1 .. m_n], and J=[J_1 .. J_n], where
              L_i is the length of the ith segment (m), d_i is the distance from the
              proximal end of the ith segment to its center of gravity (m), m_i is the
              mass of the ith segment (kg), and J_i the moment of inertia of the ith
              segment with respect to its center of gravity (kgm**2) 
              
    OUTPUT 

    joint, jointd, jointdd:   
        x and y coordinates and their derivatives (...d, ...dd) for all joints, 
        such that: joint=(jointx,jointy), and that:
        jointx.shape = jointy.shape = (nseg+1,n_samples)
                 
    Author: Koen Lemaire (k.k.lemaire2@vu.nl)  
    """
    # parse segment parameters
    nseg=segparms['nseg']
    L=segparms['L']
    
    # unravel state vector
    phi=segdynstate[0:nseg]
    phid=segdynstate[nseg:2*nseg]
    xb=segdynstate[2*nseg:2*nseg+1]
    yb=segdynstate[2*nseg+1:2*nseg+2]
    xbd=segdynstate[2*nseg+2:2*nseg+3]
    ybd=segdynstate[2*nseg+3:2*nseg+4]

    if segdynstated is not None:
        do_second_deriv=True
        phidd=segdynstated[nseg:2*nseg]
        jointxdd=segdynstated[2*nseg+2:2*nseg+3] # initialize with basedd
        jointydd=segdynstated[2*nseg+3:2*nseg+4]        
    else:
        do_second_deriv=False
        jointxdd=None
        jointydd=None
    
    # initialize
    jointx=xb
    jointy=yb
    jointxd=xbd
    jointyd=ybd
    
    # calculate coordinates
    for i in range(nseg):
        # print(f"Coords {len(jointx)}, {len(L)}, {len(phi)}")
        jointx = np.vstack((jointx, jointx[i]+L[i]*np.cos(phi[i])))
        jointy = np.vstack((jointy, jointy[i]+L[i]*np.sin(phi[i])))
        jointxd = np.vstack((jointxd, jointxd[i]-L[i]*np.sin(phi[i])*phid[i]))
        jointyd = np.vstack((jointyd, jointyd[i]+L[i]*np.cos(phi[i])*phid[i]))
        if do_second_deriv:
            jointxdd = np.vstack((jointxdd, 
          jointxdd[i]-L[i]*np.sin(phi[i])*phidd[i]-L[i]*np.cos(phi[i])*phid[i]**2))            
            jointydd = np.vstack((jointydd,
          jointydd[i]+L[i]*np.cos(phi[i])*phidd[i]-L[i]*np.sin(phi[i])*phid[i]**2))
        
    joint=(jointx, jointy)
    jointd=(jointxd, jointyd)
    jointdd=(jointxdd, jointydd)
    
    return joint, jointd, jointdd

#%% segment center of mass kinematics
def comcoord(segdynstate,segparms,segdynstated=None):
    """
    segcom, segcomd, segcomdd, totcom, totcomd, totcomdd = comcoord(segdynstate,segparms,segdynstated=None)
    
    Calculates the x and y coordinates and their derivatives of all segment 
    center of mass (segcom) and the total center of mass (totcom) as a function 
    of segdynstate and segdynstated (see segdyn for definition)
      
    INPUT 
    segdynstate:    vector containing segdynstate (see segdyn for definition) 
    segdynstated:    vector containing segdynstated (see segdyn for definition) 
              
    segparms: dictionary with fields: 
              L=[l_1 .. l_n], d=[d_1 .. d_n], m=[m_1 .. m_n], and J=[J_1 .. J_n], where
              L_i is the length of the ith segment (m), d_i is the distance from the
              proximal end of the ith segment to its center of gravity (m), m_i is the
              mass of the ith segment (kg), and J_i the moment of inertia of the ith
              segment with respect to its center of gravity (kgm**2) 
              
    OUTPUT 

    segcom, segcomd, segcomdd: x and y positions and their derivatives
    all segment COM, such that segcom = (segcomx,segcomy), and 
    segcomx.shape=(nseg,n_samples)
    
    totcom, totcomd, totcomdd: x and y positions and their derivatives of 
    total body COM, such that totcom = (totcomx,totcomy), and 
    totcomx.shape=(nseg,n_samples)
    
    Author: Koen Lemaire (k.k.lemaire2@vu.nl)  
    """
    # parse segment parameters
    nseg=segparms['nseg']
    m=segparms['m']
    d=segparms['d']
    
    # unravel state vector
    phi=segdynstate[0:nseg]
    phid=segdynstate[nseg:2*nseg]
    
    # initialize
    n_states,n_samples=np.shape((segdynstate))
    segcomx=np.zeros((nseg,n_samples))
    segcomy=np.zeros((nseg,n_samples))
    segcomxd=np.zeros((nseg,n_samples))
    segcomyd=np.zeros((nseg,n_samples))
    segcomxdd=np.zeros((nseg,n_samples))
    segcomydd=np.zeros((nseg,n_samples))        

    # deal with optional acceleration
    if segdynstated is not None:
        do_second_deriv=True
        phidd=segdynstated[nseg:2*nseg]
    else:
        do_second_deriv=False
    
    # calculate joint coordinate positions and velocities 
    joint, jointd, jointdd = jointcoord(segdynstate,segparms,segdynstated)
    
    jointx,jointy=joint
    jointxd,jointyd=jointd
    jointxdd,jointydd=jointdd
    
    # calculate segment com positions and velocities
    for i in range(nseg):
        segcomx[i]  = jointx[i]  + d[i]*np.cos(phi[i])
        segcomy[i]  = jointy[i]  + d[i]*np.sin(phi[i])
        segcomxd[i] = jointxd[i] - d[i]*np.sin(phi[i])*phid[i]
        segcomyd[i] = jointyd[i] + d[i]*np.cos(phi[i])*phid[i]
        if do_second_deriv:
            segcomxdd[i] = jointxdd[i] - d[i]*np.sin(phi[i])*phidd[i] - d[i]*np.cos(phi[i])*phid[i]**2
            segcomydd[i] = jointydd[i] + d[i]*np.cos(phi[i])*phidd[i] - d[i]*np.sin(phi[i])*phid[i]**2
                
    # calculate total COM position and velocity as weighted sum of segcom 
    # position and velocity
    totcomx  = m@segcomx/np.sum(m)
    totcomy  = m@segcomy/np.sum(m)
    totcomxd = m@segcomxd/np.sum(m)
    totcomyd = m@segcomyd/np.sum(m)
    
    # condense output
    segcom=(segcomx,segcomy)
    segcomd=(segcomxd,segcomyd)
    
    totcom=(totcomx,totcomy)
    totcomd=(totcomxd,totcomyd)
    
    if do_second_deriv:
        totcomxdd = m@segcomxdd/np.sum(m)
        totcomydd = m@segcomydd/np.sum(m)
        segcomdd=(segcomxdd,segcomydd)
        totcomdd=(totcomxdd,totcomydd)
    else:
        segcomdd=(None,None)
        totcomdd=(None,None)
    
    # return output    
    return segcom, segcomd, segcomdd, totcom, totcomd, totcomdd

#%% angular momentum w.r.t. total center of mass
def angmom(segdynstate,segparms,segdynstated=None):
    """
    angmom, angmomd = angmom(segdynstate,segparms,segdynstated=None)
    
    Calculates the total angular momentum about the centre of mass of the 
    complete model as a function of segdynstate (see segdyn for definition)
    and segparms.
      
    INPUT 
    segdynstate: vector containing segdynstate (see segdyn for definition) 
              
    segparms: dictionary with fields: 
              L=[l_1 .. l_n], d=[d_1 .. d_n], m=[m_1 .. m_n], and J=[J_1 .. J_n], where
              L_i is the length of the ith segment (m), d_i is the distance from the
              proximal end of the ith segment to its center of gravity (m), m_i is the
              mass of the ith segment (kg), and J_i the moment of inertia of the ith
              segment with respect to its center of gravity (kgm**2)              
    OUTPUT 

    angmom_com: total angular momentum with respect to total body COM. 
                angmom_com.shape = (n_samples,)
                 
    Author: Koen Lemaire (k.k.lemaire2@vu.nl)  
    """
    # parse segment parameters
    nseg=segparms['nseg']
    m=segparms['m']
    J=segparms['J']
    
    # unravel state vector
    phid=segdynstate[nseg:2*nseg,:]
    
    # deal with second derivative case
    if segdynstated is not None:
        do_second_deriv=True
        phidd=segdynstated[nseg:2*nseg]
    else:
        do_second_deriv=False
        
    # get segcom and totcom positions and velocities
    segcom, segcomd, segcomdd, totcom, totcomd, totcomdd = comcoord(
        segdynstate,segparms,segdynstated)
        
    segcomx,segcomy=segcom
    segcomxd,segcomyd=segcomd
    segcomxdd,segcomydd=segcomdd
    
    totcomx,totcomy=totcom
    totcomxd,totcomyd=totcomd
    totcomxdd,totcomydd=totcomdd
    
    # initialize angular momentum:
    angmom = np.zeros((totcomx.shape)) # similar size as total comx
    angmomd = np.zeros((totcomx.shape)) # similar size as total comx
    # calculate total angular momentum
    for i in range(nseg):
        # spin angmom of each segment with respect to segcom:
        angmom_spin=J[i]*phid[i]        
        # linear momentum of each segment w.r.t. total com:
        px=m[i]*(segcomxd[i]-totcomxd)
        py=m[i]*(segcomyd[i]-totcomyd)        
        # 'arm' of linear momentum vector of each segment w.r.t. total com:
        rx=segcomx[i]-totcomx
        ry=segcomy[i]-totcomy        
        # orbital angular momentum of each segment; cross product of 'arm' 
        # vector with linear momentum of segcom w.r.t. totcom (!):
        angmom_orb=rx*py-ry*px        
        # total angular momentum; sum of spin and orbital angmom:
        angmom+=angmom_spin+angmom_orb        
        if do_second_deriv:
            # spin angmom dot of each segment with respect to segcom:
            angmom_spind=J[i]*phidd[i]            
            # linear momentum dot of each segment w.r.t. total com:
            pxd=m[i]*(segcomxdd[i]-totcomxdd)
            pyd=m[i]*(segcomydd[i]-totcomydd)            
            # (arm of linear momentum stays same!)            
            # orbital angular momentum dot of each segment; cross product of 
            # 'arm' vector with linear momentum dot of segcom w.r.t. totcom (!):
            angmom_orbd=rx*pyd-ry*pxd            
            # total angular momentum dot; sum of spin and orbital angmomd:
            angmomd+=angmom_spind+angmom_orbd                
    
    # return output    
    return angmom, angmomd

#%% kinetic and potential energy calculation
def energy(segdynstate,segparms):
    """
    Ekinx, Ekiny, Erot, Epot, Etot = energy(segdynstate,segparms)
    
    Calculates the total (summed over all segments) translational kinetic 
    energy (Ekinx and Ekiny), rotational kinetic energy (Erot) and 
    gravitational potential energy (Erot) based on segdynstate (see segdyn for 
    definition) and segparms.
      
    INPUT 
    segdynstate:    vector containing segdynstate (see segdyn for definition) 
              
    segparms: dictionary with fields: 
              L=[l_1 .. l_n], d=[d_1 .. d_n], m=[m_1 .. m_n], and J=[J_1 .. J_n], where
              L_i is the length of the ith segment (m), d_i is the distance from the
              proximal end of the ith segment to its center of gravity (m), m_i is the
              mass of the ith segment (kg), and J_i the moment of inertia of the ith
              segment with respect to its center of gravity (kgm**2) 
              
    OUTPUT 

    Ekinx, Ekiny, Erot, Epot, Epot: Translational and rotational kinetic energy, 
    gravitational potential energyn and total mechanical energy, summed accross 
    segments
                 
    Author: Koen Lemaire (k.k.lemaire2@vu.nl)  
    """
    # parse segment parameters
    nseg=segparms['nseg']
    m=segparms['m']
    J=segparms['J']
    g=segparms['g']
    
    # unravel state vector
    phid=segdynstate[nseg:2*nseg]
    
    # calculate segment COM positions and velocities 
    segcom, segcomd, *_ = comcoord(segdynstate,segparms)
    segcomx,segcomy=segcom
    segcomxd,segcomyd=segcomd
    
    # initialize
    n_states,n_samples=np.shape((segdynstate))
    Ekinx=np.zeros((nseg,n_samples))
    Ekiny=np.zeros((nseg,n_samples))
    Erot=np.zeros((nseg,n_samples))
    Epot=np.zeros((nseg,n_samples))
    
    # calculate segment energies:
    for i in range(nseg):
        Ekinx[i] = 0.5*m[i]*segcomxd[i]**2
        Ekiny[i] = 0.5*m[i]*segcomyd[i]**2
        Erot[i]  = 0.5*J[i]*phid[i]**2
        Epot[i]  = -g*m[i]*segcomy[i]
    
    # sum over segments:
    Ekinx=np.sum(Ekinx,axis=0)
    Ekiny=np.sum(Ekiny,axis=0)
    Erot=np.sum(Erot,axis=0)
    Epot=np.sum(Epot,axis=0)
    
    # total energy:
    Etot=Ekinx+Ekiny+Erot+Epot
        
    # return output
    return Ekinx, Ekiny, Erot, Epot, Etot

#%% animation function
def animate(t,segdynstate,segparms,axlim=2):
    """
    anim = animate(t,segdynstate,segparms)
    
    Makes simple animation of a stick figure based on segdynstate. Calls
    segdyn_jointcoord to calculate x and y joint coordinates. Animation will
    show in real time with a frame rate of 20 Hz.
    ! anim is a NECESSARY dummy variable to return to make the animation run !
    
    Note 1: you may have to tinker with 'inline' or 'automatic' figure setting
    (at least in spyder IDE 5.2.2 ...) in order for the animation to show.
    Note 2: (for developers) saving animation to file and automatic static 
    figure frame selection / user defined input frame are not yet implemented. 
    
    INPUT 
    t:              time (s)
    segdynstate:    vector containing segdynstate (see segdyn for definition) 
              
    segparms: dictionary with fields: 
              L=[l_1 .. l_n], d=[d_1 .. d_n], m=[m_1 .. m_n], and J=[J_1 .. J_n], where
              L_i is the length of the ith segment (m), d_i is the distance from the
              proximal end of the ith segment to its center of gravity (m), m_i is the
              mass of the ith segment (kg), and J_i the moment of inertia of the ith
              segment with respect to its center of gravity (kgm**2) 
    
    axlim: the range the axes are xlim = ylim = (-axlim, axlim)
              
    OUTPUT 
    anim:    NECESSARY dummy variable to make the animation run
             
    Author: Koen Lemaire (k.k.lemaire2@vu.nl)  
    """
    # nr of frames and time interval, such that total simulation time is accurate
    # interpolate data to match framerate of animator  
    time_interval=0.05 # 50 milliseconds for each frame
    nseg=segparms['nseg']
    nr_frames=np.ceil((t[-1]-t[0])/time_interval).astype(int) + 1
    t_new=np.linspace(t[0],t[-1],num=nr_frames)
    segdynstate_new=np.zeros((2*nseg+4,nr_frames))
    for i in range(2*nseg+4):
        segdynstate_new[i,:]=np.interp(t_new, t, segdynstate[i])
        
    # calculate joint coordinates
    joint, *_ = jointcoord(segdynstate_new,segparms)
    jointx,jointy=joint
    
    # determine range of image?? / user defined initial frame??
    
    # initiate figure
    fig = plt.figure()
    ax = plt.axes(xlim =(-axlim, axlim),
                    ylim =(-axlim, axlim))
    ax.set_aspect('equal', adjustable='box')
    line, = ax.plot([], [], lw=2)

    # initiate line
    def init_fig():
        line.set_data([], [])
        return line,
    
    def animate(i,jointx,jointy):
         # appending values to the previously
     	 # empty x and y data holders
        xdata=jointx[:,i]
        ydata=jointy[:,i]
        line.set_data(xdata, ydata)
        return line,

    # calling the animation function	
    anim = animation.FuncAnimation(fig, animate, init_func=init_fig,
                                    fargs=(jointx,jointy),
                                    frames = nr_frames,
     	       						interval = time_interval*1000,
     			     				blit = True)
    # make sure plot renders and shows
    plt.draw()
    plt.show()
    
    # save the animation somewhere. To be implemented later ??
    #anim.save('whateverName.mp4', writer = 'ffmpeg', fps = 1/time_interval/1000??)

    return anim # NECESARRY to return!! 