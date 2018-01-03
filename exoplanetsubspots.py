"""For calculating sub-observer and sub-stellar locations.

Defines the method :func:`sub_observerstellar`.

Thanks to Clara Sekowski for preliminary work on this script.

"""

import numpy as np
pi = np.pi

def sub_observerstellar(times,worb,wrot,inc,obl,sol,longzero=0):
    """Calculates an exoplanet's sub-observer and -stellar locations over time.

    Calculates time-dependent, trigonometric values of an exoplanet's sub-
    observer and sub-stellar locations when on a circular orbit. Planet
    coordinates are colatitude (theta) and longitude (phi). Orbital phase
    is zero when planet is opposite star from observer (superior conjunction)
    and increases CCW when system is viewed above star's North pole. See
    Appendix A of Schwartz et al. (2016).

    Args:
	times (1d array, int, or float): Discrete time values in any unit,
	    with total number n_time. At t=0 planet is at superior
	    conjunction.
	worb (int or float): Orbital angular frequency in radians per unit
	    time. Positive values are prograde orbits (CCW), negative are
	    retrograde (CW).
	wrot (int or float): Rotational angular frequency in radians per
	    unit time. For prograde orbits, positive values are prograde
	    rotation, negative are retrograde (vice versa for retrograde
	    orbits).
	inc (int or float): Inclination of orbital plane to the observer,
	    in radians. Zero is face-on, pi/2 is edge-on.
	obl (int or float): Obliquity relative to the worb vector, in radians.
	    This is the tilt of the planet's spin axis. Zero is North
	    pole up, pi/2 is maximal tilt, pi is North pole down.
	sol (int or float): The orbital phase of Northern Summer solstice,
	    in radians. If the wrot vector is projected into the orbital
	    plane, then this phase is where that projection points at the
	    star.
        longzero (int or float): Longitude of the sub-observer point when t=0,
            in radians. Default is zero.

    Returns:
        trigvals (ndarray): Array of trigonometric values with shape
            (8, n_time). First dimension is organized as:
            [ sin theta_obs, cos theta_obs, sin phi_obs, cos phi_obs,
              sin theta_st,  cos theta_st,  sin phi_st,  cos phi_st  ]
              
    """
    if isinstance(times,np.ndarray) and (times.size == times.shape[0]):
        timeA = times
        N_time = timeA.size  # Number of time steps from input array
    elif isinstance(times,(int,float)):
        timeA = np.array([times])
        N_time = 1
    else:
        print('sub_observerstellar aborted: input times should be ndarray (1D), int, or float.')
        return
    
    phaseA = worb*timeA  # Orbital phases
    phiGen = wrot*timeA - longzero  # General expression for PhiObs (without overall negative sign)
    
    cThObs = (np.cos(inc)*np.cos(obl)) + (np.sin(inc)*np.sin(obl)*np.cos(sol))
    cThObsfull = np.repeat(cThObs,N_time)
    sThObs = (1.0 - (cThObs**2.0))**0.5

    sThObsfull = np.repeat(sThObs,N_time)
    cThSt = np.sin(obl)*np.cos(phaseA - sol)
    sThSt = (1.0 - (cThSt**2.0))**0.5
    
    sol_md = (sol % (2.0*pi))
    inc_rd = round(inc,4)  # Rounded inclination, for better comparison
    p_obl_rd = round((pi - obl),4)  # Rounded 180 degrees - obliquity, for better comparison

    cond_face = (((inc == 0) or (inc == pi)) and ((obl == 0) or (obl == pi)))  # Pole-observer 1: face-on inclination
    cond_north = ((sol_md == 0) and ((inc == obl) or (inc_rd == -p_obl_rd))) # Ditto 2: North pole view
    cond_south = ((sol == pi) and ((inc_rd == p_obl_rd) or (inc == -obl))) # Ditto 3: South pole view
    
    if cond_face or cond_north or cond_south:
        if (obl == (pi/2.0)):
            aII = np.sin(phaseA)*np.cos(sol)  # Special "double-over-pole" time-dependent factor
            cPhiSt = np.ones(N_time)
            sPhiSt = np.zeros(N_time)
            g_i = (sThSt != 0)  # Excluding "star-over-pole" situations (g_i are "good indicies")
            cPhiSt[g_i] = (-np.sin(phiGen[g_i])*aII[g_i])/sThSt[g_i]
            sPhiSt[g_i] = (-np.cos(phiGen[g_i])*aII[g_i])/sThSt[g_i]
        else:
            aI = np.cos(phaseA)*np.cos(obl)  # Alternate "observer-over-pole" time-dependent factor
            bI = np.sin(phaseA)  # Ditto
            cPhiSt = ((np.cos(phiGen)*aI) + (np.sin(phiGen)*bI))/sThSt
            sPhiSt = ((-np.sin(phiGen)*aI) + (np.cos(phiGen)*bI))/sThSt
    else:
        a = (np.sin(inc)*np.cos(phaseA)) - (cThObs*cThSt)  # Normal time-dependent factor
        b = ((np.sin(inc)*np.sin(phaseA)*np.cos(obl) - np.cos(inc)*np.sin(obl)*np.sin(phaseA - sol)))  # Ditto
        if (obl == (pi/2.0)):
            cPhiSt = np.ones(N_time)
            sPhiSt = np.zeros(N_time)
            g_i = (sThSt != 0)  # Excluding "star-over-pole" situations (g_i are "good_indicies")
            cPhiSt[g_i] = ((np.cos(phiGen[g_i])*a[g_i]) + (np.sin(phiGen[g_i])*b[g_i]))/(sThObs*sThSt[g_i])
            sPhiSt[g_i] = ((-np.sin(phiGen[g_i])*a[g_i]) + (np.cos(phiGen[g_i])*b[g_i]))/(sThObs*sThSt[g_i])
        else:
            cPhiSt = ((np.cos(phiGen)*a) + (np.sin(phiGen)*b))/(sThObs*sThSt)
            sPhiSt = ((-np.sin(phiGen)*a) + (np.cos(phiGen)*b))/(sThObs*sThSt)
    
    trigvals = np.stack((sThObsfull,cThObsfull,np.sin(-phiGen),np.cos(-phiGen),sThSt,cThSt,sPhiSt,cPhiSt))
    return trigvals
