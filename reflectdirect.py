"""A suite about reflected light from directly imaged planets.

ReflectDirect is an MIT licensed Python suite for exploring
exoplanetary systems in JupyterLab. Customize a planet's
brightness map and geometry, generate and compare
light curves, analyze the kernel of reflection and more.

This suite defines the class :class:`DirectImaging_Planet` that
has most of the important features. There's also the method
:func:`Geometry_Reference` if you need a diagram that explains
some conventions.

Depends on:

    - numpy and scipy
    - matplotlib
    - Jupyter(Lab), ipywidgets and IPython

Uses the function :func:`sub_observerstellar()
<exoplanetsubspots.sub_observerstellar>` and the binary files
kernel_width_values_all5deg.npy and
kernel_domcolat_values_all5deg.npy for some backend
calculations.

See `Schwartz et al. (2016) <https://arxiv.org/abs/1511.05152>`_
for background info.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as pat
import ipywidgets as widgets

from pathlib import Path
from numpy.lib import stride_tricks
from scipy.special import sph_harm
from scipy.interpolate import RectBivariateSpline
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cbook import get_sample_data
from ipywidgets import Layout
from IPython.display import display as IPy_display

import exoplanetsubspots as exoss

pi = np.pi


def _rolling(vec,window):
    """Rolls a window over a vector and makes a new array."""
    new_dim = ((vec.size - window + 1),window)
    new_bytes = (vec.itemsize,vec.itemsize)
    return stride_tricks.as_strided(vec,shape=new_dim,strides=new_bytes)

## Use RD folder's absolute path to load reliably, especially when making Sphinx docs.
folder_path = str(Path(__file__).parent.absolute())
kernel_widths_ = np.load(folder_path + '/kernel_width_values_all5deg.npy')[:-1,:,:,:19]  # Pre-handling duplicates and unneeded
kernel_domcolats_ = np.load(folder_path + '/kernel_domcolat_values_all5deg.npy')[:-1,:,:,:19]


def _serial_shift(ary):
    """Rolls a window over an array and makes a new aray."""
    twice_ary = np.tile(ary,(2,1,1,1))
    new_dim = (ary.shape[0],)+ary.shape
    new_bytes = (ary.strides[0],)+ary.strides
    return stride_tricks.as_strided(twice_ary,shape=new_dim,strides=new_bytes)

shifted_domcolats_ = _serial_shift(np.copy(kernel_domcolats_))
kernel_delta_domcolats_ = np.absolute(kernel_domcolats_[np.newaxis,:,:,:,:] - shifted_domcolats_)

phase_4mesh_,inc_4mesh_,ss_,oo_ = np.meshgrid(np.linspace(0,2*pi,73),np.linspace(0,pi/2,19),
                                              np.linspace(0,2*pi,73),np.linspace(0,pi/2,19),indexing='ij')
del ss_,oo_
phase_4mesh_,inc_4mesh_ = phase_4mesh_[:-1,:,:,:],inc_4mesh_[:-1,:,:,:]
shifted_phase_4mesh_ = _serial_shift(np.copy(phase_4mesh_))

sol_2mesh_,obl_2mesh_ = np.meshgrid(np.linspace(0,2*pi,73),np.linspace(0,pi/2,19),indexing='ij')


colat_ticks_ = np.array([r'$0^{\circ}$',r'$45^{\circ}$',r'$90^{\circ}$',r'$135^{\circ}$',r'$180^{\circ}$'])
long_ticks_ = np.array([r'$-180^{\circ}$',r'$-90^{\circ}$',r'$0^{\circ}$',r'$90^{\circ}$',r'$180^{\circ}$'])
wlong_ticks_ = np.array([r'$0^{\circ}$',r'$25^{\circ}$',r'$50^{\circ}$',r'$75^{\circ}$',r'$100^{\circ}$'])
obl_ticks_ = np.array([r'$0^{\circ}$',r'$30^{\circ}$',r'$60^{\circ}$',''])
sol_ticks_ = np.array([r'$0^{\circ}$','',r'$90^{\circ}$','',r'$180^{\circ}$','',r'$270^{\circ}$',''])
relph_ticks_ = np.array([r'$-2^{\circ}$',r'$-1^{\circ}$',r'$0^{\circ}$',r'$1^{\circ}$',r'$2^{\circ}$'])


def _combine_2_colormaps(cm1,va1,vb1,n1,cm2,va2,vb2,n2,power,name):
    """Creates a new colormap by joining two others."""
    c1 = cm1(np.linspace(va1,vb1,n1))
    c2 = cm2(np.linspace(va2,vb2,n2))
    C = np.vstack((c1,c2))
    
    L = np.sum([0.2126,0.7152,0.0722]*C[:,0:3],axis=1)
    lwi,Li = L.argmin(),np.indices(L.shape)[0]
    d_Li = np.absolute(Li-lwi)
    
    C[:,0:3] *= (d_Li[:,np.newaxis]/d_Li.max())**power
    return LinearSegmentedColormap.from_list(name,C)

darkmid_BrBG_ = _combine_2_colormaps(cm1=cm.BrBG,va1=0.4,vb1=0,n1=128,
                                     cm2=cm.BrBG,va2=0.99,vb2=0.59,n2=128,
                                     power=0.5,name='darkmid_BrBG_')


def _rotate_ccw_angle(X,Y,ang):
    """Rotates arrays CCW by an angle."""
    S_ang,C_ang = np.sin(ang),np.cos(ang)
    X_new = C_ang*X - S_ang*Y
    Y_new = S_ang*X + C_ang*Y
    return X_new,Y_new


def Geometry_Reference(ref_save=False,**kwargs):
    """Makes a reference diagram about exoplanetary systems.

    .. image:: _static/geomref_example.png
        :width: 60%
        :align: center
    
    For example, this figure defines important angles. See
    Appendix A of `Schwartz et al. (2016) <https://arxiv.org/abs/1511.05152>`_.

    Args:
        ref_save (bool):
            Save the diagram as "geometry_reference.pdf" in the current
            working directory. Default is False.

    .. note::
        
        Keywords are only used by the class :class:`DirectImaging_Planet`
        for the interactive function :func:`Sandbox_Reflection()
        <reflectdirect.DirectImaging_Planet.Sandbox_Reflection>`.
    
    """
    ## Default keywords
    _active = kwargs.get('_active',False)
    incD = kwargs.get('incD',85)
    oblD = kwargs.get('oblD',0)
    solD = kwargs.get('solD',0)
    ratRO = kwargs.get('ratRO',10.0)
    phaseD = kwargs.get('phaseD',[0])
    ph_colors = kwargs.get('ph_colors',['k'])
    name = kwargs.get('name','NONE')
    reference = kwargs.get('reference',True)  # Gets set to False by Geometry_Diagram
    
    if _active:
        ax = kwargs.get('ax_I','N/A')  # Shouldn't need fig in this case, but may want to pass in.
        comp_tweak,comp_siz = 0.04,'medium'
    else:
        fig,ax = plt.subplots(figsize=(8,8))
        fig.set_facecolor('w')  # Avoids transparent stored fig_****; ditto throughout
        comp_tweak,comp_siz = 0.03,'x-large'
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.35,1.65])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0)
    if reference:
        here_incD,here_oblD,here_solD,here_ratRO,here_phaseD = 45,45,70,'N/A',225
    else:
        here_incD,here_oblD,here_solD,here_ratRO,here_phaseD = incD,oblD,solD,ratRO,phaseD

    star_color = (1,0.75,0)
    orbit = pat.Circle((0,0),radius=1,color='k',fill=False,zorder=0)
    ax.add_patch(orbit)
    star = pat.Circle((0,0),radius=0.33,color=star_color,fill=True,zorder=1)
    ax.add_patch(star)
    ax.plot([0,0],[-1.35,0],'k',lw=2,ls=':',zorder=-2)

    compass = 1.15
    ax.plot([0,0],[1.0,compass],'k',lw=1,ls='-',zorder=0)
    ax.text(comp_tweak,compass,r'$0^{\circ}$',size=comp_siz,ha='left',va='bottom')
    ax.plot([-1.0,-compass],[0,0],'k',lw=1,ls='-',zorder=0)
    ax.text(-compass,comp_tweak,r'$90^{\circ}$',size=comp_siz,ha='right',va='bottom')
    ax.text(-comp_tweak-0.01,-compass,r'$180^{\circ}$',size=comp_siz,ha='right',va='top')
    ax.plot([1.0,compass],[0,0],'k',lw=1,ls='-',zorder=0)
    ax.text(compass,-comp_tweak-0.01,r'$270^{\circ}$',size=comp_siz,ha='left',va='top')

    axis_color = (0,0.9,0)
    tri_x,tri_y = 0.2*np.array([0,0.5,-0.5]),0.2*np.array([1,-0.5,-0.5])
    tsol_x,tsol_y = _rotate_ccw_angle(tri_x,tri_y,np.radians(here_solD+180))
    sol_x,sol_y = _rotate_ccw_angle(0.67,0,np.radians(here_solD+90))
    triang = pat.Polygon(np.array([sol_x+tsol_x,sol_y+tsol_y]).T,color=axis_color,fill=True,zorder=1)
    ax.add_patch(triang)
    ax.plot([sol_x/0.67,sol_x],[sol_y/0.67,sol_y],color=axis_color,lw=2,ls='-',zorder=-1)
    ax.plot([0,0],[1.0,0.8],'k',lw=1,ls='--',zorder=-2)
    sang = pat.Arc((0,0),1.7,1.7,angle=90,theta1=0,theta2=here_solD,lw=1,color=(0.67,0.67,0),zorder=-3)
    ax.add_patch(sang)

    cupx,cupy,to_ell = 1.12,1.16,0.49
    ax.plot([-cupx,-cupx+to_ell],[cupy,cupy],'k',lw=2,ls=':',zorder=-1)
    starup = pat.Circle((-cupx,cupy),radius=0.1,color=star_color,fill=True,zorder=0)
    ax.add_patch(starup)
    ix,iy = np.array([0,0]),np.array([0,0.3])
    ax.plot(-cupx+ix,cupy+iy,'k',lw=1,ls='--',zorder=-2)
    if reference and not _active:
        ax.plot(-cupx-iy,cupy+ix,'k',lw=1,ls='--',zorder=-2)
        ax.text(-cupx+comp_tweak,cupy+iy[1],r'$0^{\circ}$',size=comp_siz,ha='left',va='top')
        ax.text(-cupx-iy[1],cupy-comp_tweak,r'$90^{\circ}$',size=comp_siz,ha='left',va='top')
        ax.text(-cupx+0.7*to_ell,cupy+0.02,'To\nobserver',size='medium',ha='center',va='bottom')
    iy += [-0.3,0]
    nix,niy = _rotate_ccw_angle(ix,iy,np.radians(here_incD))
    ax.plot(-cupx+nix,cupy+niy,'k',zorder=0)
    iang = pat.Arc((-cupx,cupy),0.4,0.4,angle=90,theta1=0,theta2=here_incD,lw=1,color=(0.67,0,0.67),zorder=-3)
    ax.add_patch(iang)

    planet_color = '0.5'
    ax.plot([cupx-to_ell,1.5],[cupy,cupy],'k',zorder=-1)
    planet = pat.Circle((cupx,cupy),radius=0.15,color=planet_color,fill=True,zorder=1)
    ax.add_patch(planet)
    ox,oy = np.array([0,0]),np.array([0,0.3])
    ax.plot(cupx+ox,cupy+oy,'k',lw=1,ls='--',zorder=-2)
    if reference and not _active:
        ax.plot(cupx+ox,cupy-oy,'k',lw=1,ls='--',zorder=-2)
        ax.text(cupx+comp_tweak,cupy+oy[1],r'$0^{\circ}$',size=comp_siz,ha='left',va='top')
        ax.text(cupx-oy[1]-0.02,cupy-comp_tweak,r'$90^{\circ}$',size=comp_siz,ha='left',va='top')
        ax.text(cupx+comp_tweak,cupy-oy[1],r'$180^{\circ}$',size=comp_siz,ha='left',va='bottom')
        ax.text(cupx-0.5*to_ell,cupy+0.7*oy[1],'North',size='medium',ha='right',va='center')
    oy += [-0.2,0]
    nox,noy = _rotate_ccw_angle(ox,oy,np.radians(here_oblD))
    ax.plot(cupx+nox,cupy+noy,c=axis_color,lw=2,zorder=0)
    oang = pat.Arc((cupx,cupy),0.45,0.45,angle=90,theta1=0,theta2=here_oblD,lw=1,color=(0,0.67,0.67),zorder=-3)
    ax.add_patch(oang)

    cex,cey = 1.5,1.65
    iarc = pat.Ellipse((-cex,cey),2.0,2.0,lw=2,ec='0.75',fc=(1,0,1,0.05),zorder=-4)
    ax.add_patch(iarc)
    oarc = pat.Ellipse((cex,cey),2.0,2.0,lw=2,ec='0.75',fc=(0,1,1,0.05),zorder=-4)
    ax.add_patch(oarc)

    if _active:
        n = 0
        for p in here_phaseD:
            if isinstance(p,(int,float)):
                phx,phy = _rotate_ccw_angle(0,1,np.radians(p))
                planet_loc = pat.Circle((phx,phy),radius=0.1,color=ph_colors[n],fill=True,zorder=1)
                ax.add_patch(planet_loc)
            n += 1
    else:
        phx,phy = _rotate_ccw_angle(0,1,np.radians(here_phaseD))
        planet_loc = pat.Circle((phx,phy),radius=0.1,color=planet_color,fill=True,zorder=1)
        ax.add_patch(planet_loc)

    tex_y = 1.6
    if _active:
        tex_x,lab_siz = 0.65,'medium'
        ax.text(-tex_x,tex_y,r'$%.1f^{\circ}$' % here_incD,color='k',size=lab_siz,ha='right',va='top')
        ax.text(tex_x,tex_y,r'$%.1f^{\circ}$' % here_oblD,color='k',size=lab_siz,ha='left',va='top')
        ax.text(0,0,'$%.1f^{\circ}$' % here_solD,color='k',size=lab_siz,ha='center',va='center')
        ax.text(-0.95,-1.25,'Spins per orbit:' '\n' '$%.2f$' % here_ratRO,
                color='k',size=lab_siz,ha='center',va='bottom')
        ax.text(0,tex_y,'Geometry',color='k',size=lab_siz,ha='center',va='top')
    elif reference:
        tex_x,lab_siz = 1.02,'x-large'
        ax.text(-0.5,0.32,'North',size='medium',ha='center',va='center')
        ax.text(-tex_x,tex_y,'Inclination angle',color='k',size=lab_siz,ha='center',va='top')
        ax.text(tex_x,tex_y,'Obliquity angle',color='k',size=lab_siz,ha='center',va='top')
        ax.text(-0.67,0.08,'Solstice\nangle',color='k',size=lab_siz,ha='center',va='top')
        ax.text(0,0,'Star',color='k',size=lab_siz,ha='center',va='center')
        ax.text(-0.45,-0.55,'Orbital\nphase',color='k',size=lab_siz,ha='center',va='bottom')
        ax.plot([-0.707,-0.45],[-0.707,-0.575],'k',lw=1,ls='--',zorder=-2)
        rat_tex = r'$\omega_{\mathrm{rot}} \ / \ \omega_{\mathrm{orb}}$'
        ax.text(-0.95,-1.25,'Spins per orbit:\n'+rat_tex+' is\n+ for prograde spins\n— for retrograde spins',
                color='k',size=lab_siz,ha='center',va='bottom')
        tim_tex = r'At $0^{\circ}$ phase:' '\n' '$t = nT_{\mathrm{orb}}$,' '\n' r'$n=0,\pm1,\pm2,...$'
        ax.text(0.21,0.775,tim_tex,color='k',size=lab_siz,ha='center',va='top')
        ax.text(0.6,-0.6,'Planet',color='k',size=lab_siz,ha='right',va='bottom')
        connect = pat.Arc((0.5,0),0.5,1.3,angle=0,theta1=-70,theta2=85,lw=1,ls='--',color='0.75',zorder=-3)
        ax.add_patch(connect)
        ax.text(1.3,-1.3,'Not to\nscale',color='k',size=lab_siz,ha='center',va='bottom',fontstyle='italic')
        ax.text(0,tex_y,'Geometry\nReference',color='k',size=lab_siz,ha='center',va='top',weight='bold')
    else:
        tex_x,lab_siz = 1.02,'x-large'
        ax.text(-tex_x,tex_y,r'Inclination: $%.1f^{\circ}$' % here_incD,color='k',size=lab_siz,ha='center',va='top')
        ax.text(tex_x,tex_y,r'Obliquity: $%.1f^{\circ}$' % here_oblD,color='k',size=lab_siz,ha='center',va='top')
        ax.text(0,0,'Solstice:' '\n' '$%.1f^{\circ}$' % here_solD,color='k',size=lab_siz,ha='center',va='center')
        rat_tex = r'$\omega_{\mathrm{rot}} \ / \ \omega_{\mathrm{orb}} \ = \ %.2f$' % here_ratRO
        ax.text(-0.95,-1.25,'Spins per orbit:\n'+rat_tex,color='k',size=lab_siz,ha='center',va='bottom')
        ax.text(1.3,-1.3,'Not to\nscale',color='k',size=lab_siz,ha='center',va='bottom',fontstyle='italic')
        ax.text(0,tex_y,'Geometry of\n{}'.format(name),color='k',size=lab_siz,ha='center',va='top')
    ax.text(0.25,-1.25,'To observer',color='k',size=lab_siz,ha='left',va='bottom')
    ax.plot([0,0.25],[-1.3,-1.25],'k',lw=1,ls='--',zorder=-2)
    
    if not _active:
        if reference:
            fig.tight_layout()
            if ref_save:
                fig.savefig('geometry_reference.pdf')
            plt.show()
        else:
            return fig  # Pass back to Geometry_Diagram


class DirectImaging_Planet:
    
    """An exoplanet that is directly imaged using reflected starlight.

    This class is based on the model, equations and discussion of
    `Schwartz et al. (2016) <https://arxiv.org/abs/1511.05152>`_,
    S16 in the methods below. It has two sets of planetary
    parameters, a primary and an alternate, that users control. These
    sets make calling many of the class methods simple and consistent.
    Several methods store figures that can be saved later.

    Planet coordinates are colatitude and longitude. Orbital phase
    is zero when planet is opposite star from observer and
    increases CCW when system is viewed above star's North pole.

    Methods:
        Update Param Sets:
        
            - :func:`Adjust_Geometry`
            - :func:`Adjust_MotionTimes`
            - :func:`Build_Amap`
            - :func:`InvertFlipBlend_Amap`
            - :func:`Setup_ProRet_Degeneracy`

        Use Param Sets:
        
            - :func:`Kernel_WidthDomColat`
            - :func:`Light_Curves`
            - :func:`SubOS_TimeDeg`
        
        Visualize Param Sets:
        
            - :func:`EquiRect_Amap`
            - :func:`Geometry_Diagram`
            - :func:`KChar_Evolve_Plot`
            - :func:`Kernels_Plot`
            - :func:`LightCurve_Plot`
            - :func:`Orthographic_Viewer`
            - :func:`Sandbox_Reflection` --- interactive
            - :func:`SpinAxis_Constraints`
        
        Other:
        
            - :func:`Info_Printout`
            - :func:`Kernel2D`
            - :func:`KernelClat`
            - :func:`KernelLong`
            
    Attributes:
        name (str):
            Your exoplanet's name.
        times (1d array):
            Time array based on the primary orbital period.
            
        n_clat (int):
            Number of colatitudes for the planetary grid.
        n_long (int):
            Number of longitudes for the planetary grid.
            
        clat_vec (1d array):
            Colatitude vector, zero to 180 degrees.
        long_vec (1d array):
            Longitude vector, zero to 360 degrees and zero in center.
        mono_long_vec (1d array):
            Monotonic longitude vector, -180 to 180 degrees.
            
        clats (2d array):
            Colatitude array, based on ``clat_vec``.
        longs (2d array):
            Longitude array, based on ``long_vec``.
        mono_longs (2d array):
            Monotonic longitude array, based on ``mono_long_vec``.
            
        delta_clat (float):
            Gap between colatitudes.
        delta_long (float):
            Gap between longitudes.
            
        cos_clats (2d array):
            Cosine of colatitudes.
        cos_longs (2d array):
            Cosine of longitudes.
        sin_clats (2d array):
            Sine of colatitudes.
        sin_longs (2d array):
            Sine of longitudes.
        
        Primary Params (append ``_b`` for Alternates):
            albedos (2d array):
                The planet's albedo values with shape (``n_clat``, ``n_long``).
                
            incD (int or float):
                Inclination of orbital plane to the observer, in degrees.
                Zero is face-on, 90 is edge-on.
                
            longzeroD (int or float):
                Longitude of the sub-observer point when t=0, in degrees.
                
            oblD (int or float):
                Obliquity relative to the orbital angular frequency vector,
                in degrees. This is the tilt of the planet's spin axis.
                Zero is North pole up, 90 is maximal tilt, 180 is
                North pole down.
                
            orbT (int or float):
                Orbital period of the planet, in any time unit.
                
            ratRO (int or float):
                Ratio of the planet's rotational and orbital angular
                frequencies. This is how many spins the planet makes
                per orbit. Can be fractional, and negative numbers are
                retrograde rotation.
                
            solD (int or float):
                The orbital phase of Northern Summer solstice, in degrees.
                If the rotational angular frequency vector is projected
                into the orbital plane, then this phase is where that
                projection points at the star.
        
        Stored Figures:
            
            - fig_equi --- :func:`EquiRect_Amap`
            - fig_geom --- :func:`Geometry_Diagram`
            - fig_kcha --- :func:`KChar_Evolve_Plot`
            - fig_kern --- :func:`Kernels_Plot`
            - fig_ligh --- :func:`LightCurve_Plot`
            - fig_orth --- :func:`Orthographic_Viewer`
            - fig_sand --- :func:`Sandbox_Reflection`
            - fig_spin --- :func:`SpinAxis_Constraints`
        
    """

    def _odd_check(self,n,number,quality):
        """Makes sure your input number is odd."""
        if (n % 2) == 1:
            return n
        else:
            print('Input {} is even, added 1 to {}.'.format(number,quality))
            return n + 1
    
    def _colat_long(self,n_clat,n_long):
        """Sets up colatitude and longitude attributes."""
        self.n_clat = self._odd_check(n_clat,'number of colatitudes','include the equator')
        self.clat_vec = np.linspace(0,pi,self.n_clat)
        self.n_long = self._odd_check(n_long,'number of longitudes','include the prime meridian')
        self.mono_long_vec = np.linspace(-pi,pi,self.n_long)
        self.long_vec = self.mono_long_vec % (2.0*pi)

        self.clats,self.longs = np.meshgrid(self.clat_vec,self.long_vec,indexing='ij')
        ignore,self.mono_longs = np.meshgrid(self.clat_vec,self.mono_long_vec,indexing='ij')
        del ignore
        self.delta_clat = pi/(self.n_clat - 1)
        self.delta_long = 2.0*pi/(self.n_long - 1)
        
        self.sin_clats = np.sin(self.clats)
        self.cos_clats = np.cos(self.clats)
        self.sin_longs = np.sin(self.longs)
        self.cos_longs = np.cos(self.longs)
        
        
    def _import_image(self,filename):
        """Imports a png image to make a brightness map."""
        raw_values = plt.imread(filename)
        if raw_values.ndim == 2:
            return raw_values
        else:
            ## Convert from sRGB to luminance (e.g. BT.709)
            ## imread should give values between 0 and 1 (divide by 255.0 otherwise)
            gamma_rgb = raw_values[:,:,:3]  # Don't use alpha channel if it's there
            linear_rgb = np.zeros(gamma_rgb.shape)
            
            ## Separate values by cutoff in formula
            low_mask = (gamma_rgb <= 0.04045)
            high_mask = np.logical_not(low_mask)
            
            ## Transform gamma encoded rgb to linear
            linear_rgb[low_mask] = gamma_rgb[low_mask]/12.92
            linear_rgb[high_mask] = ((gamma_rgb[high_mask]+0.055)/1.055)**2.4
            
            ## Multiply by the transform coefficients and sum
            return np.sum(linear_rgb*[0.2126,0.7152,0.0722],axis=2)
    
    def _pixel_bounds(self,i,skip):
        """Returns low and high limits for pixels."""
        low = max(int(round((i-0.5)*skip)),0)
        high = int(round((i+0.5)*skip))
        if high == low:
            high += 1
        return low,high
    
    def _convert_image_pixels(self,kind,img_values):
        """Converts an input image into a brightness map."""
        rows,cols = img_values.shape
        
        if kind in ['pngA','aryA']:
            row_skip,col_skip = (rows-1)/(self.n_clat-1),(cols-1)/(self.n_long-1)
            pre_albedos = np.zeros(self.clats.shape)
            for r in np.arange(self.n_clat):
                for c in np.arange(self.n_long):
                    r_low,r_high = self._pixel_bounds(r,row_skip)
                    c_low,c_high = self._pixel_bounds(c,col_skip)
                    pixel_sum = np.sum(img_values[r_low:r_high,c_low:c_high])
                    pre_albedos[r,c] = pixel_sum/((r_high-r_low)*(c_high-c_low))
                    
        elif kind in ['pngI','aryI']:
            r_v,c_v = np.linspace(0,pi,rows),np.linspace(-pi,pi,cols)
            img_interp = RectBivariateSpline(r_v,c_v,img_values)
            pre_albedos = img_interp(self.clat_vec,self.mono_long_vec)
            
        return pre_albedos
    
    def _rolling_amap(self,image,n):
        """Rolls the longitudes of a brightness map."""
        roll_image = np.copy(image)
        roll_image[:,:-1] = np.roll(roll_image[:,:-1],n,axis=1)
        roll_image[:,-1] = roll_image[:,0]
        return roll_image
    
    def _linear_convert(self,image,lims):
        """Converts values of an array into a given range."""
        lower_img,upper_img = image.min(),image.max()
        if lower_img != upper_img:
            new_low,new_high = lims
            linear_slope = (new_high - new_low)/(upper_img - lower_img)
            new_image = linear_slope*(image - lower_img) + new_low
        else:
            new_image = image
        return new_image
    
    def _amap_average(self,image):
        """Calculates the mean value of a brightness map."""
        return np.sum(image[:,:-1]*self.sin_clats[:,:-1])*self.delta_clat*self.delta_long/(4.0*pi)
    

    def InvertFlipBlend_Amap(self,image='pri',into='alt',invert=False,flip='none',blend='none'):
        """Inverts, flips and blends a given albedo map.

        Args:
            image (str or ndarray):
                The source map. If string, can be
                
                    - 'pri' to use primary map (default),
                    - 'alt' to use alternate map.
                
                Otherwise, an ndarry or values.
            
            into (str):
                Where the new map goes. Can be
            
                    - 'pri' for the primary map,
                    - 'alt' for the alternate map (default),
                    - 'none' to just return the map.
                
            .. note::
                        
                If you try to put an ``image`` ndarray ``into`` the primary
                or alternate map, it should have shape (``n_clat``, ``n_long``).
                
            invert (bool):
                Linearly change lower albedo values to higher
                values and vice versa. Default is False.
            
            flip (str):
            
                - 'EW' to flip map about the prime meridian,
                - 'NS' to flip map about the equator,
                - 'both' to flip map both ways,
                - 'none' to do nothing (default).
                
            blend (str):
            
                - 'EW' to blend map into Jupiter-like bands,
                - 'NS' to blend map into beach ball-like stripes,
                - 'both' to blend map into a uniform ball,
                - 'none' to do nothing (default).

        Effect:
            If ``into`` is 'pri' or 'alt', stores new albedo map as ``albedos``
            or ``albedos_b``, respectively.

        Returns:
            New albedo map with same shape as source map, if ``into``
            is 'none'.
            
        """
        if isinstance(image,str):
            if image == 'pri':
                old_image = self.albedos
                new_image = np.copy(self.albedos)
            elif image == 'alt':
                old_image = self.albedos_b
                new_image = np.copy(self.albedos_b)
        else:
            if into in ['pri','alt']:
                if image.shape != (self.n_clat,self.n_long):
                    print('InvertFlipBlend_Amap warning: you tried storing an image with shape {},'.format(image.shape))
                    print('    but right now {} is expecting albedo maps with shape ({}, {}).'.format(self.name,
                                                                                                      self.n_clat,
                                                                                                      self.n_long))
                    print('    I stopped this function so you do not get errors later on.')
                    return
            old_image = image
            new_image = np.copy(image)
        
        if invert:
            inv_lims = [old_image.max(),old_image.min()]
            new_image = self._linear_convert(new_image,inv_lims)
            
        if flip == 'both':
            new_image = np.fliplr(np.flipud(new_image))
        elif flip == 'EW':
            new_image = np.fliplr(new_image)
        elif flip == 'NS':
            new_image = np.flipud(new_image)
            
        if blend == 'both':
            new_image[:,:] = self._amap_average(new_image)
        elif blend == 'EW':
            ns_values = np.sum(new_image[:,:-1],axis=1)*self.delta_long/(2.0*pi)
            new_image = np.tile(ns_values,(self.n_long,1)).transpose()
        elif blend == 'NS':
            ew_values = np.sum(new_image*self.sin_clats,axis=0)*self.delta_clat/2.0
            new_image = np.tile(ew_values,(self.n_clat,1))
            
        new_image[:,-1] = new_image[:,0]

        if into == 'pri':
            self.albedos = new_image
        elif into == 'alt':
            self.albedos_b = new_image
        elif into == 'none':
            return new_image
    

    def Build_Amap(self,kind='ylm',map_data=[[1,-1,1.0],[2,0,-1.0]],primeD=0,limit=True,alb_lims=[0.0,1.0],
                   into='pri',invert=False,flip='none',blend='none'):
        """Creates an albedo map from input data.

        Args:
            kind (str):
                Style of planetary map. Can be
                
                    - 'pngA' to average values from a png image,
                    - 'pngI' to interpolate values from a png image,
                    - 'ylm' to use spherical harmonics (default),
                    - 'aryA' to average values from a 2D array,
                    - 'aryI' to interpolate values from a 2D array.
            
            map_data:
                Depends on ``kind``.
                
                    - For either 'png' this is the file path to your
                      image.
                    - For 'ylm' this is an n-by-3 list of spherical
                      harmonics with entries [degree ell, order m,
                      coefficient]. Default list is
                      [ [1, -1, 1.0], [2, 0, -1.0] ].
                    - For either 'ary' this is your 2D array itself.

            .. note::
                
                Png images are assumed to be equirectangular maps:
                
                    - poles on top and bottom edges,
                    - equator horizontal across middle,
                    - prime meridian vertical in center, and
                    - anti-prime meridian on left and right edges.
                
            primeD (int or float):
                Longitude of the prime meridian in degrees,
                relative to the input data. Rounded to the nearest grid
                longitude. Default is zero.
            
            limit (bool):
                Set the lowest and highest albedo values. Default is True.
            
            alb_lims (list):
                The albedo limits as [lower, upper]. Default is [0, 1.0].
            
            into (str):
                Where the new map goes. Can be
            
                    - 'pri' for the primary map,
                    - 'alt' for the alternate map (default),
                    - 'none' to just return the map.
            
            invert (bool):
                Linearly change lower albedo values to higher
                values and vice versa. Default is False.
            
            flip (str):
            
                - 'EW' to flip map about the prime meridian,
                - 'NS' to flip map about the equator,
                - 'both' to flip map both ways,
                - 'none' to do nothing (default).
            
            blend (str):
            
                - 'EW' to blend map into Jupiter-like bands,
                - 'NS' to blend map into beach ball-like stripes,
                - 'both' to blend map into a uniform ball,
                - 'none' to do nothing (default).

        Effect:
            If ``into`` is 'pri' or 'alt', stores new albedo map as ``albedos``
            or ``albedos_b``, respectively.

        Returns:
            New albedo map with shape (``n_clat``, ``n_long``), if ``into``
            is 'none'.
            
        """
        if (kind == 'pngA') or (kind == 'pngI'):
            img_values = self._import_image(map_data)
            pre_albedos = self._convert_image_pixels(kind,img_values)
            
        elif kind == 'ylm':
            pre_albedos = np.zeros(self.clats.shape)
            for y in np.arange(len(map_data)):
                ell,m,c = map_data[y]
                if abs(m) <= ell:
                    pre_albedos += c*np.real(sph_harm(m,ell,self.longs,self.clats))
                else:
                    print('Ylm warning for component {} in your list: degree {:.0f} is not >= |order {:.0f}|.'
                          .format(y,ell,m))
                
        elif (kind == 'aryA') or (kind == 'aryI'):
            pre_albedos = self._convert_image_pixels(kind,map_data)
            
        else:
            print('Build_Amap aborted because kind must be one of below.')
            print('    \'pngA\' or \'pngI\': values averaged or interpolated from a png image.')
            print('    \'ylm\': n-by-3 list of spherical harmonics with entries [degree ell, order m, coefficient].')
            print('    \'aryA\' or \'aryI\': values averaged or interpolated from an ndarray (2D).')
            return
        
        if (primeD % 360.0) != 0:
            simple_prime = ((primeD+180.0) % 360.0) - 180.0
            n_prime = round((simple_prime/360.0)*(self.n_long-1))
            primed_albedos = self._rolling_amap(pre_albedos,-n_prime)
            ang_amap_rotated = (-n_prime/(self.n_long-1))*360.0
            if ang_amap_rotated < 0:
                directions = ['East','West']
            else:
                directions = ['West','East']
            print('You asked to put the prime meridian {:.2f} degrees {} of normal on your input albedo map.'
                  .format(abs(simple_prime),directions[0]))
            print('    There are only {} unique longitudes, so I rotated your map {:.2f} degrees to the {}.'
                  .format(self.n_long-1,abs(ang_amap_rotated),directions[1]))
        else:
            primed_albedos = pre_albedos
        
        if limit:
            primed_albedos = self._linear_convert(primed_albedos,alb_lims)

        if into in ['pri','alt']:
            self.InvertFlipBlend_Amap(image=primed_albedos,into=into,
                                      invert=invert,flip=flip,blend=blend)
        elif into == 'none':
            return self.InvertFlipBlend_Amap(image=primed_albedos,into=into,
                                             invert=invert,flip=flip,blend=blend)
    
    
    def _setup_for_actmodule(self):
        """Initializes attributes for the interactive module."""
        self._xph_lig = 'no'
        self._xph_med = 'no'
        self._xph_drk = 'no'
        
        head_space = {'description_width': 'initial'}
        layfull = Layout(width='100%',align_self='center')
        laynearfull = Layout(width='95%',align_self='center')
        
        self._orb_act = widgets.FloatSlider(value=0,min=0,max=360,step=0.1,
                                            description='Orbital Phase:',
                                            layout=Layout(width='50%',align_self='center'),
                                            style=head_space,continuous_update=False)
        
        self._inc_act = widgets.FloatSlider(value=85,min=0,max=90,step=0.1,
                                            description='Inclination:',
                                            layout=layfull,continuous_update=False)
        self._obl_act = widgets.FloatSlider(value=0,min=0,max=180,step=0.1,
                                            description='Obliquity:',
                                            layout=layfull,continuous_update=False)
        self._sol_act = widgets.FloatSlider(value=0,min=0,max=360,step=0.1,
                                            description='Solstice:',
                                            layout=layfull,continuous_update=False)
        
        self._ratRO_act = widgets.FloatSlider(value=72,min=-400,max=400,step=0.1,
                                              description='Spins per Orbit:',
                                              layout=layfull,style=head_space,continuous_update=False)
        self._res_act = widgets.IntSlider(value=101,min=11,max=301,step=1,
                                          description='Time Steps per Spin:',
                                          layout=layfull,style=head_space,continuous_update=False)
        self._zlong_act = widgets.IntSlider(value=0,min=-180,max=180,step=1,
                                            description=r'Initial Longitude:',
                                            layout=layfull,style=head_space,continuous_update=False)
        
        self._ligcur_act = widgets.Dropdown(description='Light Curve:',
                                            options=[('Flux (map \u00D7 kernel)','flux'),
                                                     ('Apparent Brightness','appar')],
                                            value='flux',
                                            layout=layfull)
        self._spax_act = widgets.Dropdown(description='Axis Constraint:',
                                          options=[('Rotational (red)','wid'),
                                                   ('Orbital (blue)','dom'),
                                                   ('Combined','both')],
                                          value='wid',
                                          layout=layfull,style=head_space)
        
        self._pslot_act = widgets.Dropdown(description='Extra Phase Slot:',
                                           options=[('Light','light'),
                                                    ('Medium','medium'),
                                                    ('Dark','dark'),
                                                    ('All','all')],
                                           value='light',
                                           layout=laynearfull,style=head_space)
        first_pword = '<center><font color="blue">Ready to save/clear orbital phases</font></center>'
        self._pword_act = widgets.HTML(value=first_pword,layout=laynearfull)
        
        self._psav_act = widgets.Button(description='Save',button_style='success',
                                        layout=laynearfull)
        self._psav_act.on_click(lambda x: self._savebutton_click())
        self._pclr_act = widgets.Button(description='Clear',button_style='warning',
                                        layout=laynearfull)
        self._pclr_act.on_click(lambda x: self._clearbutton_click())
        
        self._title_act = widgets.HTML(value='<center><b>Interact with '+self.name+'</b><center>',
                                       layout=layfull)

    def _setup_figurevars(self):
        """Initializes figure attributes."""
        null_draw = 'Figure not made yet'
        self.fig_equi = null_draw
        self.fig_geom = null_draw
        self.fig_kcha = null_draw
        self.fig_kern = null_draw
        self.fig_ligh = null_draw
        self.fig_orth = null_draw
        self.fig_sand = null_draw
        self.fig_spin = null_draw
    
    
    def __init__(self,name='This Exoplanet',n_clat=37,n_long=73,
                 kind='ylm',map_data=[[1,-1,1.0],[2,0,-1.0]],primeD=0,
                 limit=True,alb_lims=[0.0,1.0],
                 invert=False,flip='none',blend='none',
                 orbT=(24.0*365.0),ratRO=10.0,
                 incD=85,oblD=0,solD=0,longzeroD=0):
        """*Constructor for the class DirectImaging_Planet*

        All arguments are for your **primary** map and params.

        For your alternate map, inverts the primary map. Other
        alternate params are set equal to the primary values.
        
        Args:
            name (str):
                Your exoplanet's name. Default is 'This Exoplanet'.
            
            n_clat (int):
                Number of colatitudes for the planetary grid.
                Method ensures this is odd so the equator is included.
                Default is 37.
            
            n_long (int):
                Number of longitudes for the planetary grid.
                Method ensures this is odd so the prime meridian is
                included. Default is 73.
            
            kind (str):
                Style of planetary map. Can be
            
                    - 'pngA' to average values from a png image,
                    - 'pngI' to interpolate values from a png image,
                    - 'ylm' to use spherical harmonics (default),
                    - 'aryA' to average values from a 2D array,
                    - 'aryI' to interpolate values from a 2D array.
            
            map_data:
                Depends on ``kind``.
            
                    - For either 'png' this is the file path to your
                      image.
                    - For 'ylm' this is an n-by-3 list of spherical
                      harmonics with entries [degree ell, order m,
                      coefficient]. Default list is
                      [ [1, -1, 1.0], [2, 0, -1.0] ].
                    - For either 'ary' this is your 2D array itself.

            .. note::
                        
                Png images are assumed to be equirectangular maps:
                
                    - poles on top and bottom edges,
                    - equator horizontal across middle,
                    - prime meridian vertical in center, and
                    - anti-prime meridian on left and right edges.
                        
            primeD (int or float):
                Longitude of the prime meridian in degrees, relative to
                the input data. Rounded to the nearest grid longitude.
                Default is zero.
            
            limit (bool):
                Set the lowest and highest albedo values. Default is True.
            
            alb_lims (list):
                The albedo limits as [lower, upper]. Default is [0, 1.0].
            
            invert (bool):
                Linearly change lower albedo values to higher values and
                vice versa. Default is False.
            
            flip (str):
            
                - 'EW' to flip map about the prime meridian,
                - 'NS' to flip map about the equator,
                - 'both' to flip map both ways,
                - 'none' to do nothing (default).
            
            blend (str):
            
                - 'EW' to blend map into Jupiter-like bands,
                - 'NS' to blend map into beach ball-like stripes,
                - 'both' to blend map into a uniform ball,
                - 'none' to do nothing (default).
            
            orbT (int or float):
                Orbital period of the planet, in any unit.
                Default is 8760.0 (approx. hours in one Earth year).
            
            ratRO (int or float):
                Ratio of the planet's rotational and orbital angular
                frequencies. Default is 10.0.
                
            incD (int or float):
                Inclination in degrees. Default is 85.
            
            oblD (int or float):
                Obliquity in degrees. Default is zero.
                
            solD (int or float):
                Solstice in degrees. Default is zero.
            
            longzeroD (int or float):
                Longitude of the sub-observer point when t=0, in degrees.
                Default is zero.
                
        """
        self.name = name
        
        self._colat_long(n_clat,n_long)
        self.Build_Amap(kind=kind,map_data=map_data,primeD=primeD,
                        limit=limit,alb_lims=alb_lims,
                        into='pri',invert=invert,flip=flip,blend=blend)
        
        self.orbT = orbT
        self.ratRO = ratRO
        self._rot_res = self.n_long - 1
        self._orb_min = -0.5
        self._orb_max = 0.5
        steps_per_orbit = self._rot_res*abs(ratRO)
        n_default = round(max(steps_per_orbit,360)*(self._orb_max-self._orb_min)) + 1
        self.times = np.linspace(self._orb_min,self._orb_max,n_default)*abs(orbT)
        
        self.incD = incD
        self.oblD = oblD
        self.solD = solD
        self.longzeroD = longzeroD
        
        self.InvertFlipBlend_Amap(image='pri',into='alt',invert=True)
        self.orbT_b = orbT
        self.ratRO_b = ratRO
        self.incD_b = incD
        self.oblD_b = oblD
        self.solD_b = solD
        self.longzeroD_b = longzeroD
        
        self._setup_for_actmodule()
        self._setup_figurevars()
    
    
    def _compare_param(self,new,old):
        """Checks if a new value matches the old value."""
        if isinstance(new,str):
            return old
        else:
            return new
    
    
    def Adjust_Geometry(self,which='both',incD='no',oblD='no',solD='no',longzeroD='no'):
        """Changes your planetary system's geometry.

        Args:
            which (str):
            
                - 'pri' to adjust primary params,
                - 'alt' to adjust alternate params,
                - 'both'.
            
            incD (int, float, or str):
                New inclination in degrees (0 to 90), or any string to
                keep the current value. Default is 'no'. Other args use
                same format.
                
            oblD:
                New obliquity (0 to 180).
                
            solD:
                New solstice (0 to 360).
            
            longzeroD:
                New sub-observer longitude at t=0.
            
        """
        if which in ['pri','both']:
            self.incD = self._compare_param(incD,self.incD)
            self.oblD = self._compare_param(oblD,self.oblD)
            self.solD = self._compare_param(solD,self.solD)
            self.longzeroD = self._compare_param(longzeroD,self.longzeroD)
        if which in ['alt','both']:
            self.incD_b = self._compare_param(incD,self.incD_b)
            self.oblD_b = self._compare_param(oblD,self.oblD_b)
            self.solD_b = self._compare_param(solD,self.solD_b)
            self.longzeroD_b = self._compare_param(longzeroD,self.longzeroD_b)
        if which not in ['pri','alt','both']:
            print('I adjusted nothing because which should be \'pri\', \'alt\' or \'both\'.')
    
    
    def Adjust_MotionTimes(self,which='both',orbT='no',ratRO='no',
                           orb_min='no',orb_max='no',rot_res='no'):
        """Changes the orbital and rotational params of your planet.

        Args:
            which (str):
            
                - 'pri' to adjust primary ``orbT`` and ``ratRO``,
                - 'alt' to adjust alternate values,
                - 'both'.
            
            orbT (int, float, or str):
                New orbital period in any unit, or any string to keep
                the current value. Default is 'no'. Other args
                use same format.
            
            ratRO:
                New rotational-to-orbital frequency ratio.

            **The args below are set relative to the primary params.**
            
            orb_min:
                New minimum time in orbits, can be negative.
                
            orb_max:
                New maximum time in orbits, can be negative.
                
            rot_res:
                New number of time steps per rotation.

        .. note::
                
            Whatever you choose for ``rot_res``, there will be at least
            360 time steps per full orbit.

        Effect:
            Also updates ``times``, the time array based on the primary
            orbital period.
            
        """
        if which in ['pri','both']:
            self.orbT = self._compare_param(orbT,self.orbT)
            self.ratRO = self._compare_param(ratRO,self.ratRO)
        if which in ['alt','both']:
            self.orbT_b = self._compare_param(orbT,self.orbT_b)
            self.ratRO_b = self._compare_param(ratRO,self.ratRO_b)
        if which not in ['pri','alt','both']:
            print('I did not adjust orbT or ratRO because which should be \'pri\', \'alt\' or \'both\'.')

        self._orb_min = self._compare_param(orb_min,self._orb_min)
        self._orb_max = self._compare_param(orb_max,self._orb_max)
        self._rot_res = self._compare_param(rot_res,self._rot_res)

        steps_per_orbit = self._rot_res*abs(self.ratRO)
        n_steps = round(max(steps_per_orbit,360)*(self._orb_max-self._orb_min)) + 1
        self.times = np.linspace(self._orb_min,self._orb_max,n_steps)*abs(self.orbT)
    
    
    def Setup_ProRet_Degeneracy(self):
        """Sets your alternate params for a specific light curve degeneracy.

        The degeneracy involves the albedo map and is **usually** prograde
        vs. retrograde rotation (see note below). Discussed in Section 4.5
        and Appendix B3 of S16.

        When a planet has zero obliquity and its orbit is edge-on to you
        (inclination 90 degrees), you cannot tell from a light curve whether:
        
            - the planet does *N* spins per orbit (``ratRO``) with an albedo
              map *A*, or
            - it does 1.0--*N* spins with an East-West flipped *A*.
        
        Most often *N* and 1.0--*N* have opposite signs, so one version spins
        prograde and the other retrograde. This light curve degeneracy breaks
        down if the planet is tilted or its orbit is not edge-on.

        After running this method, test with :func:`Light_Curves`,
        :func:`LightCurve_Plot`, or :func:`Orthographic_Viewer`.

        .. note::
        
            If *N* is between 0 and 1.0, both versions of the planet will
            spin prograde. And when *N* = 0.5, their spins are identical!

        Effect:
            Calls :func:`InvertFlipBlend_Amap`, :func:`Adjust_Geometry`,
            and :func:`Adjust_MotionTimes`, using your primary params to
            setup the alternate params as described.
            
        """
        self.InvertFlipBlend_Amap(flip='EW')
        self.Adjust_Geometry(which='alt',incD=self.incD,oblD=self.oblD,solD=self.solD,
                             longzeroD=-self.longzeroD)
        self.Adjust_MotionTimes(which='alt',ratRO=1.0-self.ratRO)
    
    
    def _describe_amap(self,low,high):
        """Checks if a brightness map is realistic."""
        if low < 0:
            return 'No, some < 0'
        elif high > 1.0:
            return 'Semi, some > 1'
        else:
            return 'Yes'
    
    
    def Info_Printout(self):
        """Prints many of the current model parameters for your planet.

        Grouped by grid, albedo map, motion and geometry. The latter three
        are broken into primary and alternate sets.
        
        """
        print('Below are some parameters you are using to model {}.'.format(self.name))
        print('')
        
        form_cols = '{:^12} {:^14} {:^18}'
        print(form_cols.format('**Grid**','Number','Separation (deg)'))
        form_cols = '{:<12} {:^14} {:^18.2f}'
        print(form_cols.format('Colatitudes',self.n_clat,np.degrees(self.delta_clat)))
        act_long = '{}(+1)'.format(self.n_long-1)
        print(form_cols.format('Longitudes',act_long,np.degrees(self.delta_long)))
        print('')
        
        form_cols = '{:^16} {:^14} {:^14} {:^14} {:^16}'
        print(form_cols.format('**Albedo Map**','Low','Average','High','Realistic?'))
        form_cols = '{:<16} {:^14.3f} {:^14.3f} {:^14.3f} {:^16}'
        a_low,a_avg,a_high = self.albedos.min(),self._amap_average(self.albedos),self.albedos.max()
        print(form_cols.format('Primary',a_low,a_avg,a_high,self._describe_amap(a_low,a_high)))
        a_low_b,a_avg_b,a_high_b = self.albedos_b.min(),self._amap_average(self.albedos_b),self.albedos_b.max()
        print(form_cols.format('Alternate',a_low_b,a_avg_b,a_high_b,self._describe_amap(a_low_b,a_high_b)))
        print('')
        
        form_cols = '{:^14} {:^24} {:^22} {:^17} {:^17}'
        print(form_cols.format('**Motion**','Orbital Period (units)','Rot./Orb. Frequency',
                               'Low t (orbits)','High t (orbits)'))
        form_cols = '{:<14} {:^24.3f} {:^22.4f} {:^17.4f} {:^17.4f}'
        if isinstance(self.times,(int,float)):
            low_t,high_t = self.times,self.times
        else:
            low_t,high_t = self.times[0],self.times[-1]
        print(form_cols.format('Primary',self.orbT,self.ratRO,
                               low_t/abs(self.orbT),high_t/abs(self.orbT)))
        form_cols = '{:<14} {:^24.3f} {:^22.4f} {:^17} {:^17}'
        low_orb_b = '(({:.4f}))'.format(low_t/abs(self.orbT_b))
        high_orb_b = '(({:.4f}))'.format(high_t/abs(self.orbT_b))
        print(form_cols.format('Alternate',self.orbT_b,self.ratRO_b,low_orb_b,high_orb_b))
        print('')
        
        form_cols = '{:^14} {:^20} {:^18} {:^18} {:^22}'
        print(form_cols.format('**Geometry**','Inclination (deg)','Obliquity (deg)',
                               'Solstice (deg)','t=0 Longitude (deg)'))
        form_cols = '{:<14} {:^20.2f} {:^18.2f} {:^18.2f} {:^22.2f}'
        print(form_cols.format('Primary',self.incD,self.oblD,self.solD,self.longzeroD))
        print(form_cols.format('Alternate',self.incD_b,self.oblD_b,self.solD_b,self.longzeroD_b))
    
    
    def Geometry_Diagram(self,which='pri',**kwargs):
        """Makes a diagram of the geometry your planet is in.

        .. image:: _static/geomdiag_example.png
            :width: 60%
            :align: center

        This shows its inclination, obliquity, solstice and spins per orbit.

        Args:
            which (str):
            
                    - 'pri' to use primary params (default),
                    - 'alt' to use alternate params.

        .. note::
                
            Keywords are only used by the interactive function :func:`Sandbox_Reflection`.

        Effect:
            Stores this matplotlib figure as ``fig_geom``, **overwriting**
            the previous version. You can save the image later by
            calling ``fig_geom.savefig(...)``.
            
        """
        ## Takes almost all keywords from Geometry_Reference: _active,incD,oblD,solD,ratRO,phaseD,ph_colors...
        reference = False  # ...except this one; you never make the reference diagram here.
        
        if kwargs.get('_active',False):
            Geometry_Reference(reference=reference,**kwargs)
        else:
            if which == 'pri':
                fig = Geometry_Reference(incD=self.incD,oblD=self.oblD,solD=self.solD,ratRO=self.ratRO,
                                         phaseD=self.solD,name=self.name,reference=reference)
            elif which == 'alt':
                fig = Geometry_Reference(incD=self.incD_b,oblD=self.oblD_b,solD=self.solD_b,ratRO=self.ratRO_b,
                                         phaseD=self.solD_b,name='Alt. '+self.name,reference=reference)

            fig.tight_layout()
            self.fig_geom = fig
            plt.show()
    
    
    def _flat_style(self,fig,ax,albs,v_l,v_h,grat):
        """Styles plots for equirectangular maps."""
        if albs.min() < 0:
            m_c,lab,d_c = darkmid_BrBG_,'value',(1,0,0)
        elif albs.max() > 1.0:
            m_c,lab,d_c = cm.magma,'value',(0,1,0)
        else:
            m_c,lab,d_c = cm.bone,'albedo',(0,1,0)
        
        cart = ax.imshow(albs,cmap=m_c,extent=[-180,180,180,0],vmin=v_l,vmax=v_h)
        cbar = fig.colorbar(cart)
        cbar.set_label(label=lab,size='large')
        if grat:
            ax.axvline(-90,c=d_c,ls=':',lw=1)
            ax.axvline(0,c=d_c,ls='--',lw=1)
            ax.axvline(90,c=d_c,ls=':',lw=1)
            ax.axhline(45,c=d_c,ls=':',lw=1)
            ax.axhline(90,c=d_c,ls='--',lw=1)
            ax.axhline(135,c=d_c,ls=':',lw=1)
        
        ax.set_ylabel('Colatitude',size='x-large')
        ax.set_yticks(np.linspace(0,180,5))
        ax.set_yticklabels(colat_ticks_,size='large')
        ax.set_xlabel('Longitude',size='x-large')
        ax.set_xticks(np.linspace(-180,180,5))
        ax.set_xticklabels(long_ticks_,size='large')
    
    def _single_amap_colorbounds(self,low,high):
        """Gives limits for a brightness map's colorbar."""
        if low < 0:
            bound = max(abs(low),high)
            v_l,v_h = -bound,bound
        elif high > 1.0:
            v_l,v_h = 0,high
        else:
            v_l,v_h = 0,1.0
        return v_l,v_h
    
    def _double_amap_colorbounds(self,alt,same_scale):
        """Gives limits for two colorbars that may be related."""
        mast_l,mast_h = self.albedos.min(),self.albedos.max()
        alt_l,alt_h = self.albedos_b.min(),self.albedos_b.max()
        
        if alt and same_scale:
            if (mast_l < 0) and (alt_l < 0):
                bound = max(abs(mast_l),mast_h,abs(alt_l),alt_h)
                vm_l,vm_h,va_l,va_h = -bound,bound,-bound,bound
            elif (mast_l >= 0) and (alt_l >= 0) and (mast_h > 1.0) and (alt_h > 1.0):
                vm_l,va_l = 0,0
                bound = max(mast_h,alt_h)
                vm_h,va_h = bound,bound
            else:
                vm_l,vm_h = self._single_amap_colorbounds(mast_l,mast_h)
                va_l,va_h = self._single_amap_colorbounds(alt_l,alt_h)
        else:
            vm_l,vm_h = self._single_amap_colorbounds(mast_l,mast_h)
            va_l,va_h = self._single_amap_colorbounds(alt_l,alt_h)
        
        return vm_l,vm_h,va_l,va_h
    
            
    def EquiRect_Amap(self,alt=True,same_scale=True,grat=True):
        """Shows your albedo maps in equirectangular projection.

        .. image:: _static/equirect_example.png
            :align: center

        This projection is a simple rectangle: colatitudes are horizontal lines
        and longitudes are vertical lines. The primary map is always shown, and
        the color schemes adapt to your albedo values (real, semi-real
        or unrealistic).

        Args:
            alt (bool):
                Include the alternate map. Default is True.
            
            same_scale (bool):
                If the primary and alternate maps have the same color scheme,
                then show both on the same color scale. Default is True.
            
            grat (bool):
                Overlay a basic graticule. Default is True.

        Effect:
            Stores this matplotlib figure as ``fig_equi``, **overwriting**
            the previous version. You can save the image later by
            calling ``fig_equi.savefig(...)``.
            
        """
        vm_l,vm_h,va_l,va_h = self._double_amap_colorbounds(alt,same_scale)
        if alt:
            fig = plt.figure(figsize=(16,4))
            ax = fig.add_subplot(121)
        else:
            fig,ax = plt.subplots(figsize=(9,4))
        fig.set_facecolor('w')
        
        self._flat_style(fig,ax,self.albedos,vm_l,vm_h,grat)
        ax.set_title('Map of {}'.format(self.name),size='x-large')
        
        if alt:
            ax2 = fig.add_subplot(122)
            self._flat_style(fig,ax2,self.albedos_b,va_l,va_h,grat)
            ax2.set_title('Alternate Map of {}'.format(self.name),size='x-large')
        
        fig.tight_layout()
        self.fig_equi = fig
        plt.show()
        
    
    def _convert_omega_rad(self,orbT,ratRO,incD,oblD,solD,longzeroD):
        """Converts params to angular frequencies and radians."""
        worb = 2.0*pi/orbT
        wrot = ratRO*abs(worb)
        inc,obl,sol,longzero = np.radians(incD),np.radians(oblD),np.radians(solD),np.radians(longzeroD)
        return worb,wrot,inc,obl,sol,longzero
    
    
    def SubOS_TimeDeg(self,which='pri',times=0,orbT=(24.0*365.0),ratRO=10.0,incD=85,oblD=0,solD=0,longzeroD=0,
                      bypass_time='no'):
        """Calculates an planet's sub-observer and -stellar locations over time.

        Wrapper for :func:`exoplanetsubspots.sub_observerstellar` that
        works with the class :class:`DirectImaging_Planet`. See Appendix A
        of S16.

        Args:
            which (str):
                The param set to use. Can be
            
                    - 'pri' for primary (default),
                    - 'alt' for alternate,
                    - '_c' for custom, see Optional below.
            
            bypass_time (int, float, 1d array, or str):
                Time value(s) in place of the instance ``times``. All
                other primary or alternate params are still used. Canceled
                if any string. Default is 'no'.

        Optional:
            times, orbT, ratRO, incD, oblD, solD, longzeroD:
                Custom set of params to use if ``which`` is '_c'.
                Standard definitions and formats apply.
                See the :class:`class and constructor <DirectImaging_Planet>`
                docstrings.

        Returns:
            Array of trigonometric values with shape (8, # of time steps).
            First dimension is ordered:
            
                - sin theta_obs
                - cos theta_obs
                - sin phi_obs
                - cos phi_obs
                - sin theta_st
                - cos theta_st
                - sin phi_st
                - cos phi_st
        
        """
        if which == 'pri':
            here_times = self.times
            worb,wrot,inc,obl,sol,longzero = self._convert_omega_rad(orbT=self.orbT,ratRO=self.ratRO,
                                                                     incD=self.incD,oblD=self.oblD,
                                                                     solD=self.solD,longzeroD=self.longzeroD)
        elif which == 'alt':
            here_times = self.times
            worb,wrot,inc,obl,sol,longzero = self._convert_omega_rad(orbT=self.orbT_b,ratRO=self.ratRO_b,
                                                                     incD=self.incD_b,oblD=self.oblD_b,
                                                                     solD=self.solD_b,longzeroD=self.longzeroD_b)
        elif which == '_c':
            here_times = times
            worb,wrot,inc,obl,sol,longzero = self._convert_omega_rad(orbT=orbT,ratRO=ratRO,
                                                                     incD=incD,oblD=oblD,
                                                                     solD=solD,longzeroD=longzeroD)
        if (which != '_c') and not isinstance(bypass_time,str):
            here_times = bypass_time
        return exoss.sub_observerstellar(here_times,worb,wrot,inc,obl,sol,longzero)
    
    
    def Kernel2D(self,os_trigs):
        """Calculates a planet's 2D kernel of reflection.

        This kernel is the product of visibility and illumination at each
        location on an exoplanet. See Section 2 of S16.

        Args:
            os_trigs (ndarray):
                Trig values describing the sub-observer and sub-stellar
                points, with shape (8, # of time steps). Should
                be formatted like the output of :func:`SubOS_TimeDeg`.

        Returns:
            2D kernel with shape (# of time steps, ``n_clat``, ``n_long``).
            
        """
        St_o,Ct_o,Sp_o,Cp_o,St_s,Ct_s,Sp_s,Cp_s = os_trigs[:,:,np.newaxis,np.newaxis]
        Vis = (self.sin_clats*St_o*(self.cos_longs*Cp_o + self.sin_longs*Sp_o)) + (self.cos_clats*Ct_o)
        v_ind = (Vis < 0)
        Vis[v_ind] = 0
        Ilu = (self.sin_clats*St_s*(self.cos_longs*Cp_s + self.sin_longs*Sp_s)) + (self.cos_clats*Ct_s)
        i_ind = (Ilu < 0)
        Ilu[i_ind] = 0
        return (Vis*Ilu)/pi
    
    
    def KernelLong(self,k2d):
        """Calculates a planet's longitudinal kernel.

        Marginalizes the 2D kernel over colatitude.

        Args:
            k2d (ndarray):
                2D kernel with shape (# of time steps, ``n_clat``,
                ``n_long``), like output from :func:`Kernel2D`.

        Returns:
            Longitudinal kernel with shape (# of time steps, ``n_long``).
            
        """
        return np.sum(k2d*self.sin_clats,axis=1)*self.delta_clat
    
    
    def KernelClat(self,k2d):
        """Calculates a planet's colatitudinal kernel.

        Marginalizes the 2D kernel over longitude.

        Args:
            k2d (ndarray):
                2D kernel with shape (# of time steps, ``n_clat``,
                ``n_long``), like output from :func:`Kernel2D`.

        Returns:
            Colatitudinal kernel with shape (# of time steps, ``n_clat``).
            
        """
        return np.sum(k2d,axis=2)*self.delta_long
    
    
    def Kernel_WidthDomColat(self,which='pri',keep_kernels=False,times=0,orbT=(24.0*365.0),ratRO=10.0,
                             incD=85,oblD=0,solD=0,longzeroD=0,bypass_time='no'):
        """Calculates characteristics of the kernel over time.

        The kernel of reflection has a longitudinal width (standard
        deviation) and a dominant colatitude (weighted average) that change
        throughout a planet's orbit. See Section 2 of S16.

        Args:
            which (str):
                The param set to use. Can be
            
                    - 'pri' for primary (default),
                    - 'alt' for alternate,
                    - '_c' for custom, see Optional below.
            
            keep_kernels (bool):
                Output all kernel info, not just the characteristics,
                see Returns below. Default is False.
            
            bypass_time (int, float, 1d array, or str):
                Time value(s) in place of the instance ``times``. All other
                primary or alternate params are still used. Canceled if
                any string. Default is 'no'.

        Optional:
            times, orbT, ratRO, incD, oblD, solD, longzeroD:
                Custom set of params to use if ``which`` is '_c'.
                Standard definitions and formats apply.
                See the :class:`class and constructor <DirectImaging_Planet>`
                docstrings.

        Returns:
            sig_long (array):
                Longitudinal widths, shape (# of time steps).
            
            dom_clat (array):
                Dominant colatitudes, shape (# of time steps).
            
            If ``keep_kernels`` is True, also:
                actual_mu (array):
                    Mean longitudes, shape (# of time steps).
                
                klong (array):
                    Longitudinal kernel, shape (# of time steps, ``n_long``).
                
                kclat (array):
                    Colatitudinal kernel, shape (# of time steps, ``n_clat``).
                
                k2d (array):
                    2D kernel, shape (# of time steps, ``n_clat``, ``n_long``).
            
        """
        if which == 'pri':
            os_trigs = self.SubOS_TimeDeg(bypass_time=bypass_time)
        elif which == 'alt':
            os_trigs = self.SubOS_TimeDeg(which=which,bypass_time=bypass_time)
        elif which == '_c':
            os_trigs = self.SubOS_TimeDeg(which=which,times=times,orbT=orbT,ratRO=ratRO,
                                          incD=incD,oblD=oblD,solD=solD,longzeroD=longzeroD)
        
        k2d = self.Kernel2D(os_trigs)
        klong = self.KernelLong(k2d)
        kclat = self.KernelClat(k2d)
        
        twice_long = np.tile(self.long_vec[:-1],2)
        shift_long = _rolling(twice_long[:-1],self.n_long-1)

        klong_norm = np.sum(klong[:,:-1],axis=1)*self.delta_long
        klong_hat = klong/klong_norm[:,np.newaxis]
        virtual_mu = np.sum(shift_long*klong_hat[:,np.newaxis,:-1],axis=2)*self.delta_long

        arg_square = np.absolute(shift_long - virtual_mu[:,:,np.newaxis])
        toshrink = (arg_square > pi)
        arg_square[toshrink] = 2.0*pi - arg_square[toshrink]
        var_long = np.sum((arg_square**2.0)*klong_hat[:,np.newaxis,:-1],axis=2)*self.delta_long
        sig_long = (var_long.min(axis=1))**0.5

        coord_move_i = var_long.argmin(axis=1)
        actual_mu = virtual_mu[np.arange(len(coord_move_i)),coord_move_i] - coord_move_i*self.delta_long
        actual_mu = actual_mu % (2.0*pi)
        
        
        kclat_norm = np.sum(kclat*self.sin_clats[:,0],axis=1)*self.delta_clat
        kclat_hat = kclat/kclat_norm[:,np.newaxis]
        dom_clat = np.sum(self.clat_vec*kclat_hat*self.sin_clats[:,0],axis=1)*self.delta_clat
        
        if keep_kernels:
            return sig_long,dom_clat,actual_mu,klong,kclat,k2d
        else:
            return sig_long,dom_clat
    
    
    def Kernels_Plot(self,phaseD,which='pri',grat=True,fixed_lims=True,force_bright=True,
                     over_amap=False,albs=np.array([[1.0]]),
                     orbT=(24.0*365.0),ratRO=10.0,incD=85,oblD=0,solD=0,longzeroD=0,bypass_time='no'):
        """Diagrams your planet's kernel at a given orbital phase.

        .. image:: _static/kernplot_example.png
            :align: center

        This includes the 2D, longitudinal and colatitudinal versions
        of the kernel. The diagram also shows the kernel's mean
        longitude (pink circle), longitudinal width (red bars), and
        dominant colatitude (blue circle). If you want actual
        data instead, use :func:`Kernel_WidthDomColat`.

        Args:
            phaseD (int or float):
                Orbital phase of the planet in degrees. Standard range
                is [0, 360).
            
            which (str):
                The param set to use. Can be
            
                    - 'pri' for primary (default),
                    - 'alt' for alternate,
                    - '_c' for custom, see Optional below.
            
            grat (bool):
                Overlay basic graticules. Default is True.
            
            fixed_lims (bool):
                Keep the plotted limits for the relative long. and
                colat. kernels fixed at [0, 1.0]. Default is True.
            
            force_bright (bool):
                Use the full color scale to draw the 2D kernel.
                The false brightness can make dark drawings (like
                crescent phases) easier to see. Default is True.
            
            over_amap (bool):
                Draw a dim version of the albedo map with the 2D kernel.
                This map is not affected by ``force_bright``. Default is
                False.
            
            bypass_time (int, float, 1d array, or str):
                Time value(s) in place of the instance ``times``. All
                other primary or alternate params are still used. Canceled
                if any string. Default is 'no'.
            
        Optional:
            albs (2D array):
                Custom albedo map to use if ``which`` is '_c'.
                Its shape should be, or work with, (``n_clat``, ``n_long``).
                Default is ``np.array( [ [ 1.0 ] ] )``.
            
            times, orbT, ratRO, incD, oblD, solD, longzeroD:
                Custom set of params to use if ``which`` is '_c'.
                Standard definitions and formats apply.
                See the :class:`class and constructor <DirectImaging_Planet>`
                docstrings.

        Effect:
            Stores this matplotlib figure as ``fig_kern``, **overwriting**
            the previous version. You can save the image later by
            calling ``fig_kern.savefig(...)``.
            
        """
        if which == 'pri':
            here_albs = self.albedos
            time = self.orbT*(phaseD/360.0)
            here_incD,here_oblD,here_solD = self.incD,self.oblD,self.solD
            sig_long,dom_clat,actual_mu,klong,kclat,k2d = self.Kernel_WidthDomColat(keep_kernels=True,
                                                                                    bypass_time=time)
        elif which == 'alt':
            here_albs = self.albedos_b
            time = self.orbT_b*(phaseD/360.0)
            here_incD,here_oblD,here_solD = self.incD_b,self.oblD_b,self.solD_b
            sig_long,dom_clat,actual_mu,klong,kclat,k2d = self.Kernel_WidthDomColat(which=which,keep_kernels=True,
                                                                                    bypass_time=time)
        elif which == '_c':
            here_albs = albs
            time = orbT*(phaseD/360.0)
            here_incD,here_oblD,here_solD = incD,oblD,solD
            sig_long,dom_clat,actual_mu,klong,kclat,k2d = self.Kernel_WidthDomColat(which=which,keep_kernels=True,
                                                                                    times=time,orbT=orbT,ratRO=ratRO,
                                                                                    incD=incD,oblD=oblD,solD=solD,
                                                                                    longzeroD=longzeroD)
        sig_long,dom_clat,actual_mu,klong,kclat,k2d = sig_long[0],dom_clat[0],actual_mu[0],klong[0],kclat[0],k2d[0]
        tot_k2d = np.sum(k2d[:,:-1]*self.sin_clats[:,:-1])*self.delta_clat*self.delta_long
        kern_frac = tot_k2d/(2.0/3.0)
        
        r,c = 2,3
        fig = plt.figure(figsize=(12,8))
        fig.set_facecolor('w')
        
        ## 2D kernel
        axk = plt.subplot2grid((r,c),(1,0),colspan=2,fig=fig)
        
        if over_amap:
            if force_bright:
                to_view = (0.15*np.absolute(here_albs)/np.absolute(here_albs.max())) + (0.85*k2d/k2d.max())
            else:
                to_view = (0.15*np.absolute(here_albs)/np.absolute(here_albs.max())) + kern_frac*(0.85*k2d/k2d.max())
        else:
            to_view = k2d/k2d.max()
            if not force_bright:
                to_view *= kern_frac
        axk.contourf(np.degrees(self.mono_longs),np.degrees(self.clats),to_view,65,
                     cmap=cm.gray,vmin=0,vmax=1.0)
        if grat:
            d_c = (0,1,0)
            axk.axvline(-90,c=d_c,ls=':',lw=1)
            axk.axvline(0,c=d_c,ls='--',lw=1)
            axk.axvline(90,c=d_c,ls=':',lw=1)
            axk.axhline(45,c=d_c,ls=':',lw=1)
            axk.axhline(90,c=d_c,ls='--',lw=1)
            axk.axhline(135,c=d_c,ls=':',lw=1)
        axk.set_ylabel('Colatitude',size='large')
        axk.set_yticks(np.linspace(0,180,5))
        axk.set_yticklabels(colat_ticks_,size='medium')
        axk.set_ylim([180,0])
        axk.set_xlabel('Longitude',size='large')
        axk.set_xticks(np.linspace(-180,180,5))
        axk.set_xticklabels(long_ticks_,size='medium')
        axk.set_xlim([-180,180])
        
        ## Long. kernel
        axl = plt.subplot2grid((r,c),(0,0),colspan=2,fig=fig)
        
        klong_rel = kern_frac*klong/klong.max()
        axl.plot(np.degrees(self.mono_long_vec),klong_rel,c=cm.Reds(0.5),lw=4)
        if fixed_lims:
            axl.set_yticks(np.linspace(0,1.0,5))
            axl.set_ylim([-0.05,1.15])
            y_mu = 1.075
        else:
            axl.set_ylim([-0.05*klong_rel.max(),1.15*klong_rel.max()])
            y_mu = 1.075*klong_rel.max()
        axl.tick_params(axis='y',labelsize='medium')
        if actual_mu > pi:
            actual_mu -= 2.0*pi
        axl.scatter(np.degrees(actual_mu),y_mu,s=100,color=cm.Reds(0.33),edgecolor='k',marker='o',zorder=3)
        if (actual_mu + sig_long) > pi:
            axl.plot([-180,-180+np.degrees(actual_mu+sig_long-pi)],[y_mu,y_mu],c=cm.Reds(0.75),lw=3)
        if (actual_mu - sig_long) < -pi:
            axl.plot([180,180-np.degrees(-pi-actual_mu+sig_long)],[y_mu,y_mu],c=cm.Reds(0.75),lw=3)
        axl.plot([np.degrees(actual_mu-sig_long),np.degrees(actual_mu+sig_long)],[y_mu,y_mu],c=cm.Reds(0.75),lw=3)
        if grat:
            d_c = '0.33'
            axl.axvline(-90,c=d_c,ls=':',lw=1)
            axl.axvline(0,c=d_c,ls='--',lw=1)
            axl.axvline(90,c=d_c,ls=':',lw=1)
        axl.set_ylabel('Relative Longitudinal Kernel',size='large')
        axl.set_xticks(np.linspace(-180,180,5))
        axl.set_xticklabels(long_ticks_,size='medium')
        axl.set_xlim([-180,180])
        
        ## Colat. kernel
        axc = plt.subplot2grid((r,c),(1,2),fig=fig)
        
        kclat_rel = kern_frac*kclat/kclat.max()
        axc.plot(kclat_rel,np.degrees(self.clat_vec),c=cm.Blues(0.5),lw=4)
        axc.set_yticks(np.linspace(0,180,5))
        axc.set_yticklabels(colat_ticks_,size='medium')
        axc.set_ylim([180,0])
        if fixed_lims:
            axc.set_xticks(np.linspace(0,1.0,5))
            axc.set_xlim([-0.05,1.15])
            y_dom = 1.075
        else:
            axc.set_xlim([-0.05*kclat_rel.max(),1.15*kclat_rel.max()])
            y_dom = 1.075*kclat_rel.max()
        axc.tick_params(axis='x',labelsize='medium')
        axc.scatter(y_dom,np.degrees(dom_clat),s=100,color=cm.Blues(0.75),edgecolor='k',marker='o',zorder=3)
        if grat:
            d_c = '0.33'
            axc.axhline(45,c=d_c,ls=':',lw=1)
            axc.axhline(90,c=d_c,ls='--',lw=1)
            axc.axhline(135,c=d_c,ls=':',lw=1)
        axc.set_xlabel('Relative Colatitudinal Kernel',size='large')
        
        ## Text info
        axt = plt.subplot2grid((r,c),(0,2),fig=fig)
        
        axt.text(0,0.9,r'%s' '\n' 'at $%.2f^{\circ}$ phase' % (self.name,phaseD),color='k',size='x-large',
                 ha='center',va='center',weight='bold')
        axt.text(0,0.7,r'Inclination: $%.2f^{\circ}$' % here_incD,color='k',size='x-large',ha='center',va='center')
        axt.text(0,0.6,'Obliquity: $%.2f^{\circ}$' % here_oblD,color='k',size='x-large',ha='center',va='center')
        axt.text(0,0.5,'Solstice: $%.2f^{\circ}$' % here_solD,color='k',size='x-large',ha='center',va='center')
        axt.text(0,0.325,'Mean Longitude: $%.2f^{\circ}$' % (np.degrees(actual_mu)),
                 color=cm.Reds(0.33),size='x-large',ha='center',va='center')
        axt.text(0,0.225,'Longitudinal Width: $%.2f^{\circ}$' % (np.degrees(sig_long)),
                 color=cm.Reds(0.75),size='x-large',ha='center',va='center')
        axt.text(0,0.05,'Dominant Colatitude: $%.2f^{\circ}$' % (np.degrees(dom_clat)),
                 color=cm.Blues(0.75),size='x-large',ha='center',va='center')
        axt.set_xlim([-0.5,0.5])
        axt.set_ylim([0,1.0])
        axt.axes.get_xaxis().set_visible(False)
        axt.axes.get_yaxis().set_visible(False)
        axt.axis('off')
        
        fig.tight_layout()
        self.fig_kern = fig
        plt.show()
    
    
    def _kcevo_style(self,char,times,sig_long,dom_clat,i,imax,ax1,ax2,_active,phasesD_I,ph_colors):
        """Styles part of plots for kernel characteristics."""
        if char == 'wid':
            ax1.plot(times,np.degrees(sig_long),c=cm.Reds(0.85-0.7*i/(imax-1)),zorder=1)
        elif char == 'dom':
            ax1.plot(times,np.degrees(dom_clat),c=cm.Blues(0.85-0.7*i/(imax-1)),zorder=1)
        elif char == 'both':
            liwi, = ax1.plot(times,np.degrees(sig_long),c=cm.Reds(0.85-0.7*i/(imax-1)),
                            label='Long. Width',zorder=1)
            doco, = ax2.plot(times,np.degrees(dom_clat),c=cm.Blues(0.85-0.7*i/(imax-1)),
                            label='Dom. Colat.',zorder=1)
            if _active:
                n = 0
                for p in phasesD_I:
                    if isinstance(p,(int,float)):
                        pt_ind = round(p % 360.0)
                        ax1.scatter(times[pt_ind],np.degrees(sig_long[pt_ind]),
                                    color=ph_colors[n],edgecolor='k',s=100,marker='o',zorder=2)
                        ax2.scatter(times[pt_ind],np.degrees(dom_clat[pt_ind]),
                                    color=ph_colors[n],edgecolor='k',s=100,marker='o',zorder=2)
                    n += 1
                ax1.legend(handles=[liwi,doco],loc='best',fontsize='medium')
    
    def _kcevo_loop(self,char,explode,gap,times,incD,oblD,solD,ax1,ax2,_active,phasesD_I,ph_colors):
        """Checks which style of kernel characteristics plot to make."""
        if explode == 'inc':
            imax = int(90//gap) + 1
            ex_lab = r' - Inclination: $0^{\circ}$ dark to $%.1f^{\circ}$ light in $%.1f^{\circ}$ gaps' % ((imax-1)*gap,
                                                                                                           gap)
            for i in np.arange(imax):
                now_incD = i*gap
                sig_long,dom_clat = self.Kernel_WidthDomColat(which='_c',times=times,orbT=1,ratRO=1,
                                                              incD=now_incD,oblD=oblD,solD=solD)
                self._kcevo_style(char=char,times=times,sig_long=sig_long,dom_clat=dom_clat,
                                  i=i,imax=imax,ax1=ax1,ax2=ax2,
                                  _active=_active,phasesD_I=phasesD_I,ph_colors=ph_colors)
        elif explode == 'obl':
            imax = int(90//gap) + 1
            ex_lab = r' - Obliquity: $0^{\circ}$ dark to $%.1f^{\circ}$ light in $%.1f^{\circ}$ gaps' % ((imax-1)*gap,
                                                                                                         gap)
            for i in np.arange(imax):
                now_oblD = i*gap
                sig_long,dom_clat = self.Kernel_WidthDomColat(which='_c',times=times,orbT=1,ratRO=1,
                                                              incD=incD,oblD=now_oblD,solD=solD)
                self._kcevo_style(char=char,times=times,sig_long=sig_long,dom_clat=dom_clat,
                                  i=i,imax=imax,ax1=ax1,ax2=ax2,
                                  _active=_active,phasesD_I=phasesD_I,ph_colors=ph_colors)
        elif explode == 'sol':
            imax = int(360//gap)
            ex_lab = r' - Solstice: $0^{\circ}$ dark to $%.1f^{\circ}$ light in $%.1f^{\circ}$ gaps' % ((imax-1)*gap,
                                                                                                        gap)
            for i in np.arange(imax):
                now_solD = i*gap
                sig_long,dom_clat = self.Kernel_WidthDomColat(which='_c',times=times,orbT=1,ratRO=1,
                                                              incD=incD,oblD=oblD,solD=now_solD)
                self._kcevo_style(char=char,times=times,sig_long=sig_long,dom_clat=dom_clat,
                                  i=i,imax=imax,ax1=ax1,ax2=ax2,
                                  _active=_active,phasesD_I=phasesD_I,ph_colors=ph_colors)
        elif explode == 'none':
            ex_lab = ''
            sig_long,dom_clat = self.Kernel_WidthDomColat(which='_c',times=times,orbT=1,ratRO=1,
                                                          incD=incD,oblD=oblD,solD=solD)
            self._kcevo_style(char=char,times=times,sig_long=sig_long,dom_clat=dom_clat,
                              i=1,imax=3,ax1=ax1,ax2=ax2,
                              _active=_active,phasesD_I=phasesD_I,ph_colors=ph_colors)
        return 'Kernel Characteristics of {}'.format(self.name)+ex_lab
    
    def _kcevo_stylewid(self,ax,s_tick,s_lab,_active):
        """Styles part of plots for kernel widths."""
        ax.set_ylim([0,110])
        ax.set_yticks(np.linspace(0,100,5))
        ax.set_yticklabels(wlong_ticks_,size=s_tick)
        if not _active:
            ax.set_ylabel('Longitudinal Width',color=cm.Reds(0.75),size=s_lab)
        ax.tick_params(axis='y',colors=cm.Reds(0.75))
        ax.set_xlim([0,1])
        ax.set_xticks(np.linspace(0,1,5))
        ax.set_xlabel('Time (orbits)',size=s_lab)
        
    def _kcevo_styledom(self,ax,s_tick,s_lab,_active):
        """Styles part of plots for dominant colatitudes."""
        ax.set_ylim([180,0])
        ax.set_yticks(np.linspace(0,180,5))
        ax.set_yticklabels(colat_ticks_,size=s_tick)
        if not _active:
            ax.set_ylabel('Dominant Colatitude',color=cm.Blues(0.75),size=s_lab)
        ax.tick_params(axis='y',colors=cm.Blues(0.75))
        ax.set_xlim([0,1])
        ax.set_xticks(np.linspace(0,1,5))
        ax.set_xlabel('Time (orbits)',size=s_lab)
    
    
    def KChar_Evolve_Plot(self,char,which='pri',explode='none',gap=10,incD=85,oblD=0,solD=0,**kwargs):
        """Plots the kernel's characteristics over a full orbit.

        .. image:: _static/kcharevo_example.png
            :align: center

        If you want actual data instead, use :func:`Kernel_WidthDomColat`.

        Args:
            char (str):
                The characteristic to show. Can be
            
                    - 'wid' for longitudinal width,
                    - 'dom' for dominant colatitude,
                    - 'both'.
                    
            which (str):
                The param set to use. Can be
            
                    - 'pri' for primary (default),
                    - 'alt' for alternate,
                    - '_c' for custom, see Optional below.
                    
            explode (str):
                The geometry param to vary, starting at zero. This shows
                many evolutions instead of one curve. Can be
                
                    - 'inc' for inclination,
                    - 'obl' for obliquity,
                    - 'sol' for solstice,
                    - 'none' to cancel (default).
                
            gap (int or float):
                When you choose to ``explode``, the exploded param's
                spacing in degrees. Default is 10.

        Optional:
            incD, oblD, solD:
                Custom set of params to use if ``which`` is '_c'.
                Standard definitions and formats apply.
                See the :class:`class and constructor <DirectImaging_Planet>`
                docstrings.

        .. note::
            
            Keywords are only used by the interactive function :func:`Sandbox_Reflection`.

        Effect:
            Stores this matplotlib figure as ``fig_kcha``, **overwriting**
            the previous version. You can save the image later by
            calling ``fig_kcha.savefig(...)``.
            
        """
        ## Default keywords
        _active = kwargs.get('_active',False)
        phasesD_I = kwargs.get('phasesD_I',[0])
        ph_colors = kwargs.get('ph_colors',['k'])
        
        times = np.linspace(0,1,361)
        if which == 'pri':
            here_incD,here_oblD,here_solD = self.incD,self.oblD,self.solD
        elif which == 'alt':
            here_incD,here_oblD,here_solD = self.incD_b,self.oblD_b,self.solD_b
        elif which == '_c':
            here_incD,here_oblD,here_solD = incD,oblD,solD
        
        if _active:
            fig = kwargs.get('fig_I','N/A')
            ax1 = fig.add_subplot(236)
            ax2 = ax1.twinx()
            self._kcevo_stylewid(ax1,s_tick='medium',s_lab='medium',_active=_active)
            self._kcevo_styledom(ax2,s_tick='medium',s_lab='medium',_active=_active)
            tit = self._kcevo_loop(char=char,explode=explode,gap=gap,times=times,
                                   incD=here_incD,oblD=here_oblD,solD=here_solD,ax1=ax1,ax2=ax2,
                                   _active=_active,phasesD_I=phasesD_I,ph_colors=ph_colors)
            # 'datalim' continues to be the best option, others mess up the interactive module.
            ax1.set(adjustable='datalim',aspect=1.0/ax1.get_data_ratio())
            ax2.set(adjustable='datalim',aspect=1.0/ax2.get_data_ratio())
            return ax1
        
        else:
            fig,ax1 = plt.subplots(figsize=(10,5))
            fig.set_facecolor('w')

            if char in ['wid','dom']:
                if char == 'wid':
                    self._kcevo_stylewid(ax1,s_tick='large',s_lab='x-large',_active=_active)
                else:
                    self._kcevo_styledom(ax1,s_tick='large',s_lab='x-large',_active=_active)
                tit = self._kcevo_loop(char=char,explode=explode,gap=gap,times=times,
                                       incD=here_incD,oblD=here_oblD,solD=here_solD,ax1=ax1,ax2=0,
                                       _active=_active,phasesD_I=phasesD_I,ph_colors=ph_colors)
            elif char == 'both':
                ax2 = ax1.twinx()
                self._kcevo_stylewid(ax1,s_tick='large',s_lab='x-large',_active=_active)
                self._kcevo_styledom(ax2,s_tick='large',s_lab='x-large',_active=_active)
                tit = self._kcevo_loop(char=char,explode=explode,gap=gap,times=times,
                                       incD=here_incD,oblD=here_oblD,solD=here_solD,ax1=ax1,ax2=ax2,
                                       _active=_active,phasesD_I=phasesD_I,ph_colors=ph_colors)

            ax1.set_title(tit,size='large')
            fig.tight_layout()
            self.fig_kcha = fig
            plt.show()
        
    
    def Light_Curves(self,which='pri',albs=np.array([[1.0]]),
                     times=0,orbT=(24.0*365.0),ratRO=10.0,incD=85,oblD=0,solD=0,longzeroD=0):
        """Calculates light curves of your planet.

        Gives you both the exoplanet's flux (the sum of [*AK*], where *A* is
        the albedo map and *K* is the kernel) and its apparent brightness
        (the flux divided by the sum of *K*) over time.

        Args:
            which (str):
                The param set to use. Can be
            
                    - 'pri' for primary (default),
                    - 'alt' for alternate,
                    - '_c' for custom, see Optional below.
                
        Optional:
            albs (2D array):
                Custom albedo map to use if ``which`` is '_c'.
                Its shape should be, or work with, (``n_clat``, ``n_long``).
                Default is ``np.array( [ [ 1.0 ] ] )``.
            
            times, orbT, ratRO, incD, oblD, solD, longzeroD:
                Custom set of params to use if ``which`` is '_c'.
                Standard definitions and formats apply.
                See the :class:`class and constructor <DirectImaging_Planet>`
                docstrings.

        Returns:
            flux_ak (array):
                flux with shape (# of time steps).
            
            appar_a (array):
                apparent brightness with shape (# of time steps).
            
        """
        if which == 'pri':
            here_albs = self.albedos
            os_trigs = self.SubOS_TimeDeg()
        elif which == 'alt':
            here_albs = self.albedos_b
            os_trigs = self.SubOS_TimeDeg(which=which)
        elif which == '_c':
            here_albs = albs
            os_trigs = self.SubOS_TimeDeg(which=which,times=times,orbT=orbT,ratRO=ratRO,
                                          incD=incD,oblD=oblD,solD=solD,longzeroD=longzeroD)
        k2d = self.Kernel2D(os_trigs)
        flux_ak = np.sum(here_albs[:,:-1]*k2d[:,:,:-1]*self.sin_clats[:,:-1],axis=(1,2))*self.delta_clat*self.delta_long
        marg_k = np.sum(k2d[:,:,:-1]*self.sin_clats[:,:-1],axis=(1,2))*self.delta_clat*self.delta_long
        appar_a = flux_ak/marg_k
        return flux_ak,appar_a
    
    
    def _lc_style(self,which,F_ak,A_app,show,diff,diff_only,ax):
        """Styles plots of light curves."""
        alph = lambda d: 0.25 if d else 1.0
        if which == 'pri':
            orbT,l_c,labf,laba,zo = self.orbT,(1,0,1,alph(diff)),'Flux',r'$A_{\mathrm{apparent}}$',2
        elif which == 'alt':
            orbT,l_c,labf,laba,zo = self.orbT_b,(0,1,1,alph(diff)),'Alt. Flux',r'Alt. $A_{\mathrm{apparent}}$',1
        elif which == 'diff':
            orbT,l_c,labf,laba,zo = self.orbT,'y',r'$\Delta$ Flux',r'$\Delta \ A_{\mathrm{apparent}}$',3
        T = self.times/abs(orbT)
        
        check = (not diff_only) or (diff_only and (which == 'diff'))
        if (show in ['both','flux']) and check:
            ax.plot(T,F_ak,c=l_c,label=labf,zorder=zo)
        if (show in ['both','appar']) and check:
            ax.plot(T,A_app,c=l_c,ls='--',label=laba,zorder=zo)
    
    
    def LightCurve_Plot(self,alt=True,diff=False,diff_only=False,show='flux',**kwargs):
        """Plots light curves of your planet.

        .. image:: _static/lcplot_example.png
            :align: center

        Uses the primary and alternate params to calculate the light curves.
        If you want actual data instead, use :func:`Light_Curves`.

        Args:
            alt (bool):
                Include the alternate case. Default is True.
            
            diff (bool):
                Include the difference between the primary and alternate
                light curves, if ``alt`` is True. Default is False.
            
            diff_only (bool):
                Plot **only** the difference light curve, if ``alt`` is
                True. Default is False.
            
            show (str):
                Which light curves to calculate. Can be
            
                    - 'flux', the sum of [*AK*] where *A* is the albedo map
                      and *K* is the kernel (default),
                    - 'appar' for apparent brightness, or flux divided by
                      sum of the kernel,
                    - 'both'.
                
        .. note::
                
            Keywords are only used by the interactive function :func:`Sandbox_Reflection`.

        Effect:
            Stores this matplotlib figure as ``fig_ligh``, **overwriting**
            the previous version. You can save the image later by
            calling ``fig_ligh.savefig(...)``.
            
        """
        if kwargs.get('_active',False):
            ## Default keywords
            ax_I = kwargs.get('ax_I','N/A')
            times_I = kwargs.get('times_I',0)
            orbT_I = kwargs.get('orbT_I',(24.0*365.0))
            ratRO_I = kwargs.get('ratRO_I',10.0)
            incD_I = kwargs.get('incD_I',90)
            oblD_I = kwargs.get('oblD_I',0)
            solD_I = kwargs.get('solD_I',0)
            longzeroD_I = kwargs.get('longzeroD_I',0)
            ph_color = kwargs.get('ph_color','k')
            now_I = kwargs.get('now_I',0)
            
            flux_ak,appar_a = self.Light_Curves(which='_c',albs=self.albedos,
                                                times=times_I,orbT=orbT_I,ratRO=ratRO_I,
                                                incD=incD_I,oblD=oblD_I,solD=solD_I,longzeroD=longzeroD_I)
            Ph = np.linspace(-2.5,2.5,times_I.size)
            zo = 0
            thick = lambda n: 2 if n == 0 else 1
            if show == 'flux':
                ax_I.plot(Ph,flux_ak,c=ph_color,lw=thick(now_I),zorder=zo)
            elif show == 'appar':
                ax_I.plot(Ph,appar_a,c=ph_color,ls='--',lw=thick(now_I),zorder=zo)
        
        else:
            fig,ax = plt.subplots(figsize=(10,5))
            fig.set_facecolor('w')
            
            flux_ak,appar_a = self.Light_Curves(which='pri')
            self._lc_style('pri',flux_ak,appar_a,show,diff,diff_only,ax)
            if alt:
                flux_ak_b,appar_a_b = self.Light_Curves(which='alt')
                self._lc_style('alt',flux_ak_b,appar_a_b,show,diff,diff_only,ax)
                if diff or diff_only:
                    if self.orbT == self.orbT_b:
                        self._lc_style('diff',flux_ak-flux_ak_b,appar_a-appar_a_b,show,diff,diff_only,ax)
                    else:
                        print('LightCurve_Plot warning: diffs plot only if primary and alternate orbital periods match.')
            
            ax.axhline(0,c='0.67',ls=':',zorder=0)
            ax.legend(loc='best',fontsize='large')
            ax.set_ylabel('Value',size='x-large')
            ax.set_xlabel('Time (orbits)',size='x-large')
            ax.set_title('Light Curves of {}'.format(self.name),size='x-large')
            
            fig.tight_layout()
            self.fig_ligh = fig
            plt.show()
    
    
    def _orth_project(self,phaseD,orbT,which,incD,oblD,solD,_active,ratRO,longzeroD):
        """Sets up an orthographic projection."""
        time = orbT*(phaseD/360.0)  # Note you're calling it 'time' here, *not* 'times'
        if _active:
            os_trigs = self.SubOS_TimeDeg(which=which,times=time,orbT=orbT,ratRO=ratRO,
                                          incD=incD,oblD=oblD,solD=solD,longzeroD=longzeroD)
        else:
            os_trigs = self.SubOS_TimeDeg(which=which,bypass_time=time)
        k2d = self.Kernel2D(os_trigs)[0]
        
        St_o,Ct_o,Sp_o,Cp_o = os_trigs[:4]
        orth_Viz = Ct_o*self.cos_clats + St_o*self.sin_clats*(self.cos_longs*Cp_o + self.sin_longs*Sp_o)
        orth_preX = self.sin_clats*(self.sin_longs*Cp_o - self.cos_longs*Sp_o)
        orth_preY = St_o*self.cos_clats - Ct_o*self.sin_clats*(self.cos_longs*Cp_o + self.sin_longs*Sp_o)
        
        inc,obl,sol = np.radians(incD),np.radians(oblD),np.radians(solD)
        poleN_viz = np.cos(inc)*np.cos(obl) + np.sin(inc)*np.sin(obl)*np.cos(sol)
        poleN_x = np.sin(obl)*np.sin(sol)
        poleN_y = np.sin(inc)*np.cos(obl) - np.cos(inc)*np.sin(obl)*np.cos(sol)
        ang_from_y = np.arctan2(poleN_x,poleN_y)
        orth_X,orth_Y = _rotate_ccw_angle(orth_preX,orth_preY,-ang_from_y)
        
        return k2d,orth_Viz,orth_X,orth_Y,poleN_viz,poleN_x,poleN_y
    
    def _orth_style(self,fig,row,sub,s,which,image,v_l,v_h,
                    orth_Viz,orth_X,orth_Y,poleN_viz,poleN_x,poleN_y,name):
        """Styles plots for orthographic projections."""
        ax = fig.add_subplot(row,sub,s)
        if which == 'kern':
            m_c = cm.gray
        elif image.min() < 0:
            m_c = darkmid_BrBG_
        elif image.max() > 1.0:
            m_c = cm.magma
        else:
            m_c = cm.bone
        
        ma_image = np.ma.masked_array(image,mask=orth_Viz<0)
        cnt_plot = ax.contourf(orth_X,orth_Y,ma_image,65,cmap=m_c,vmin=v_l,vmax=v_h)
        for c in cnt_plot.collections:
            c.set_edgecolor('face')
        if round(poleN_viz,3) >= 0:
            ax.scatter(poleN_x,poleN_y,s=100,color=(0,1,0),edgecolor='k',marker='o')
        if round(poleN_viz,3) <= 0:
            ax.scatter(-poleN_x,-poleN_y,s=70,color=(0,1,0),edgecolor='k',marker='D')
        ax.set_xlim([-1.05,1.05])
        ax.set_ylim([-1.05,1.05])
        if name != 'NONE':
            ax.set_title(name,size='large',x=0.1,y=1.0,va='top',ha='left')
        ax.set_aspect(1.0)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.axis('off')
        return s+1,ax
    
    
    def Orthographic_Viewer(self,phaseD,show='real',alt=False,same_scale=True,force_bright=True,**kwargs):
        """Draws your planet's map and kernel in orthographic projection.

        .. image:: _static/orthview_example.png
            :align: center

        Shows everything from the observer's point of view (with one
        exception), based on your primary and alternate params.
        The North and South poles are drawn as a green circle and diamond,
        respectively.

        Args:
            phaseD (int or float):
                Orbital phase of the planet in degrees. Standard range
                is [0, 360).
            
            show (str):
                The data to draw. Can be
            
                    - 'amap' for the albedo map,
                    - 'kern' for the kernel,
                    - 'both' for the map and kernel separate,
                    - 'real' to multiply the map and kernel (default),
                    - 'sphere' for the whole globe: the visible and opposite
                      hemispheres with no kernel.
                
            alt (bool):
                Include the alternate albedo map. Default is True.
            
            same_scale (bool):
                If the primary and alternate maps have the same color scheme
                (and ``alt`` is True), then show both with the same color
                scale. Default is True.
            
            force_bright (bool):
                Use the full color scale to draw the kernel. Also rescales
                the kernel values into [0, 1.0] when ``show`` is 'real'.
                The false brightness can make dark drawings (like crescent
                phases) easier to see. Default is True.
                
        .. note::
                
            Keywords are only used by the interactive function :func:`Sandbox_Reflection`.

        Effect:
            Stores this matplotlib figure as ``fig_orth``, **overwriting**
            the previous version. You can save the image later by
            calling ``fig_orth.savefig(...)``.
            
        """
        if kwargs.get('_active',False):
            ## Default keywords
            fig = kwargs.get('fig_I','N/A')
            orbT_I = kwargs.get('orbT_I',(24.0*365.0))
            ratRO_I = kwargs.get('ratRO_I',10.0)
            incD_I = kwargs.get('incD_I',90)
            oblD_I = kwargs.get('oblD_I',0)
            solD_I = kwargs.get('solD_I',0)
            longzeroD_I = kwargs.get('longzeroD_I',0)
            
            row,col,s = 2,3,1  # Start on subplot(231)
            
            vm_l,vm_h,va_l,va_h = self._double_amap_colorbounds(alt,same_scale)
            (k2d,orth_Viz,orth_X,orth_Y,
             poleN_viz,poleN_x,poleN_y) = self._orth_project(phaseD=phaseD,orbT=orbT_I,which='_c',
                                                             incD=incD_I,oblD=oblD_I,solD=solD_I,
                                                             _active=True,ratRO=ratRO_I,longzeroD=longzeroD_I)
            
            s,axv = self._orth_style(fig=fig,row=row,sub=col,s=s,which='amap',
                                     image=self.albedos,v_l=vm_l,v_h=vm_h,
                                     orth_Viz=orth_Viz,orth_X=orth_X,orth_Y=orth_Y,
                                     poleN_viz=poleN_viz,poleN_x=poleN_x,poleN_y=poleN_y,name='NONE')
            axv.text(-0.7,1.04,'Visible Map',color='k',size='medium',ha='center',va='center')
            s += 1  # Now on subplot(233)
            up = lambda fb: k2d.max() if fb else 1.0/pi
            s,axk = self._orth_style(fig=fig,row=row,sub=col,s=s,which='kern',
                                     image=k2d,v_l=0,v_h=up(force_bright),
                                     orth_Viz=orth_Viz,orth_X=orth_X,orth_Y=orth_Y,
                                     poleN_viz=poleN_viz,poleN_x=poleN_x,poleN_y=poleN_y,name='NONE')
            axk.text(-0.7,1.04,'Kernel',color='k',size='medium',ha='center',va='center')
        
        else:
            row = 1
            wid = 5
            if show in ['both','sphere']:
                wid += 5
            if alt:
                if show in ['amap','both','real']:
                    wid += 5
                if show == 'sphere':
                    wid += 10
            sub,s = wid//5,1
            fig = plt.figure(figsize=(wid,5))
            fig.set_facecolor('w')
            
            vm_l,vm_h,va_l,va_h = self._double_amap_colorbounds(alt,same_scale)
            
            orbT,incD,oblD,solD = self.orbT,self.incD,self.oblD,self.solD
            (k2d,orth_Viz,orth_X,orth_Y,
             poleN_viz,poleN_x,poleN_y) = self._orth_project(phaseD=phaseD,orbT=orbT,which='pri',
                                                             incD=incD,oblD=oblD,solD=solD,
                                                             _active=False,ratRO=0,longzeroD=0)
            if show in ['kern','both']:
                up = lambda fb: k2d.max() if fb else 1.0/pi
                s,_ax = self._orth_style(fig=fig,row=row,sub=sub,s=s,which='kern',
                                         image=k2d,v_l=0,v_h=up(force_bright),
                                         orth_Viz=orth_Viz,orth_X=orth_X,orth_Y=orth_Y,
                                         poleN_viz=poleN_viz,poleN_x=poleN_x,poleN_y=poleN_y,name='Kernel')
            if show in ['amap','both','sphere']:
                s,_ax = self._orth_style(fig=fig,row=row,sub=sub,s=s,which='amap',
                                         image=self.albedos,v_l=vm_l,v_h=vm_h,
                                         orth_Viz=orth_Viz,orth_X=orth_X,orth_Y=orth_Y,
                                         poleN_viz=poleN_viz,poleN_x=poleN_x,poleN_y=poleN_y,name='Visible Map')
            if show == 'real':
                normk = lambda fb: 1.0/k2d.max() if fb else pi
                s,_ax = self._orth_style(fig=fig,row=row,sub=sub,s=s,which='real',
                                         image=normk(force_bright)*k2d*self.albedos,v_l=vm_l,v_h=vm_h,
                                         orth_Viz=orth_Viz,orth_X=orth_X,orth_Y=orth_Y,
                                         poleN_viz=poleN_viz,poleN_x=poleN_x,poleN_y=poleN_y,name=r'Kernel $\times$ Map')
            if show == 'sphere':
                s,_ax = self._orth_style(fig=fig,row=row,sub=sub,s=s,which='amap',
                                         image=self.albedos,v_l=vm_l,v_h=vm_h,
                                         orth_Viz=-orth_Viz,orth_X=-orth_X,orth_Y=orth_Y,
                                         poleN_viz=-poleN_viz,poleN_x=-poleN_x,poleN_y=poleN_y,name='Far Side of Map')
            if alt:
                orbT,incD,oblD,solD = self.orbT_b,self.incD_b,self.oblD_b,self.solD_b
                (k2d,orth_Viz,orth_X,orth_Y,
                 poleN_viz,poleN_x,poleN_y) = self._orth_project(phaseD=phaseD,orbT=orbT,which='alt',
                                                                 incD=incD,oblD=oblD,solD=solD,
                                                                 _active=False,ratRO=0,longzeroD=0)
                if show in ['amap','both','sphere']:
                    s,_ax = self._orth_style(fig=fig,row=row,sub=sub,s=s,which='amap',
                                             image=self.albedos_b,v_l=vm_l,v_h=vm_h,
                                             orth_Viz=orth_Viz,orth_X=orth_X,orth_Y=orth_Y,
                                             poleN_viz=poleN_viz,poleN_x=poleN_x,poleN_y=poleN_y,name='Visible Alt. Map')
                if show == 'real':
                    s,_ax = self._orth_style(fig=fig,row=row,sub=sub,s=s,which='real',
                                             image=normk(force_bright)*k2d*self.albedos_b,v_l=vm_l,v_h=vm_h,
                                             orth_Viz=orth_Viz,orth_X=orth_X,orth_Y=orth_Y,
                                             poleN_viz=poleN_viz,poleN_x=poleN_x,poleN_y=poleN_y,name=r'Kernel $\times$ Alt. Map')
                if show == 'sphere':
                    s,_ax = self._orth_style(fig=fig,row=row,sub=sub,s=s,which='amap',
                                             image=self.albedos_b,v_l=vm_l,v_h=vm_h,
                                             orth_Viz=-orth_Viz,orth_X=-orth_X,orth_Y=orth_Y,
                                             poleN_viz=-poleN_viz,poleN_x=-poleN_x,poleN_y=poleN_y,name='Far Side of Alt. Map')
                
            fig.suptitle(r'%s at $%.2f^{\circ}$ phase' % (self.name,phaseD),y=0,fontsize='x-large',
                         verticalalignment='bottom')
            fig.tight_layout()
            self.fig_orth = fig
            plt.show()
    
    
    def _spinax_prob_original(self,kchar,k_mu,k_sig,incs,i_mu,i_sig,phases,p_mu,p_sig,obls,
                              yes_p2,phases2,p2_mu,p2_sig):
        """Calculates a 2D PDF of spin axis constraints."""
        full_chi = ((kchar-k_mu)/k_sig)**2.0 + ((incs-i_mu)/i_sig)**2.0 + ((phases-p_mu)/p_sig)**2.0
        if yes_p2:
            full_chi += ((phases2-p2_mu)/p2_sig)**2.0
        prob_like = np.exp(-0.5*full_chi)
        n_p,n_i,n_s,n_o = incs.shape
        dp,di,ds,do = 2.0*pi/n_p,0.5*pi/(n_i-1),2.0*pi/(n_s-1),0.5*pi/(n_o-1)
        if not yes_p2:
            norm = (np.sum(prob_like[:,:,:-1,:]*np.sin(incs[:,:,:-1,:])*
                           np.sin(obls[np.newaxis,np.newaxis,:-1,:]))*dp*di*ds*do)
            prob2d = (1.0/norm)*np.sum(prob_like*np.sin(incs),axis=(0,1))*dp*di
        else:
            norm = (np.sum(prob_like[:,:,:,:-1,:]*np.sin(incs[np.newaxis,:,:,:-1,:])*
                           np.sin(obls[np.newaxis,np.newaxis,np.newaxis,:-1,:]))*dp*dp*di*ds*do)
            prob2d = (1.0/norm)*np.sum(prob_like*np.sin(incs),axis=(0,1,2))*dp*dp*di
        return prob2d
    
    def _spinax_prob_redo(self,prob2d,orig_sols,orig_obls,new_sols,new_obls):
        """Re-calculates a 2D PDF of spin axis constraints."""
        rbs_probfun = RectBivariateSpline(orig_sols[:,0],orig_obls[0,:],prob2d)
        new_prob_like = rbs_probfun(new_sols,new_obls,grid=False)
        new_ns,new_no = new_sols.shape
        new_ds,new_do = 2.0*pi/(new_ns-1),0.5*pi/(new_no-1)
        norm = np.sum(new_prob_like[:-1,:]*np.sin(new_obls[:-1,:]))*new_ds*new_do
        new_prob2d = new_prob_like/norm
        return new_prob2d
        
    def _spinax_leveling(self,prob2d,sigma_probs,res,obls):
        """Calculates n-sigma contour levels for a 2D PDF."""
        hi_prob2d = prob2d.max()
        cut_levels = np.linspace(0,hi_prob2d,res)
        good_to_sum = (prob2d[np.newaxis,:,:] >= cut_levels[:,np.newaxis,np.newaxis])
        good_to_sum[:,-1,:] = False
        n_s,n_o = obls.shape
        ds,do = 2.0*pi/(n_s-1),0.5*pi/(n_o-1)
        total_probs = np.sum(prob2d[np.newaxis,:,:]*good_to_sum*np.sin(obls[np.newaxis,:,:]),axis=(1,2))*ds*do
        args_sigma = np.argmin(np.absolute(total_probs[np.newaxis,:] - sigma_probs[:,np.newaxis]),axis=1)
        levels_sigma = args_sigma*hi_prob2d/(res-1)
        bad_lev_up = ((levels_sigma[1:]-levels_sigma[:-1]) == 0)
        levels_sigma[1:] += bad_lev_up*np.array([1.0e-12,1.1e-12,1.2e-12,1.3e-12])
        return levels_sigma
    
    ## Keep, in case you ever need to make/test kernel characteristic values.
    def _kchar_grider(self,phaseD,incD=60,n_s=73,n_o=19):
        """Calculates an array of kernel characteristics."""
        time = phaseD/360.0  # Because orbT = 1.0 below
        gap_s,gap_o = 360.0/(n_s-1),90.0/(n_o-1)  # Degrees between grid points
        solobl_wgrid,solobl_dgrid = np.zeros((n_s,n_o)),np.zeros((n_s,n_o))
        for s in np.arange(n_s):
            for o in np.arange(n_o):
                solobl_wgrid[s,o],solobl_dgrid[s,o] = self.Kernel_WidthDomColat(which='_c',times=time,orbT=1.0,ratRO=1.0,
                                                                                incD=incD,oblD=o*gap_o,solD=s*gap_s,
                                                                                longzeroD=0)
        return solobl_wgrid,solobl_dgrid
    
    def _spinax_style(self,fig,h,w,s,m_c,kind,ax_combo,sols,obls,prob2d,levs,constraint,orig_sols,orig_obls,
                      kchar,k_mu,now_phaseD,solR,oblR,mark,_active,j,entries):
        """Styles plots for spin axis constraints."""
        if kind == 'combo':
            axs = ax_combo
        else:
            axs = fig.add_subplot(h,w,s,projection='polar')
        axs.set_theta_zero_location('S')
        axs.set_rlabel_position(45)
        c_regs = ('1.0',m_c(0.25),m_c(0.5),m_c(0.75))
        
        if kind == 'info':
            axs.text(np.radians(0),np.radians(0),'Predicted\nSpin Axis\nConstraints',color='k',
                     size='x-large',ha='center',va='center',weight='bold')
            axs.text(np.radians(225),np.radians(110),'Orbital\nPhase(s)',color='0.5',
                     size='x-large',ha='center',va='center')
            axs.text(np.radians(210),np.radians(60),'Radial\ndirection:\nobliquity',color='k',
                     size='x-large',ha='center',va='center')
            axs.text(np.radians(150),np.radians(60),'Compass\ndirection:\nsolstice',color='k',
                     size='x-large',ha='center',va='center')
            if constraint in ['real','both']:
                axs.text(np.radians(270),np.radians(60),'Red\nregions:\nrotational\ninfo',color=cm.Reds(0.75),
                         size='x-large',ha='center',va='center')
                axs.text(np.radians(90),np.radians(60),'Blue\nregions:\norbital\ninfo',color=cm.Blues(0.75),
                         size='x-large',ha='center',va='center')
                axs.text(np.radians(330),np.radians(60),'Dashed\ncontour:\nno uncertainty',color=(0,0.3,0),
                         size='x-large',ha='center',va='center')
            else:
                axs.text(np.radians(270),np.radians(60),'Red\ncontours:\nrotational\ninfo',color=cm.Reds(0.75),
                         size='x-large',ha='center',va='center')
                axs.text(np.radians(90),np.radians(60),'Blue\ncontours:\norbital\ninfo',color=cm.Blues(0.75),
                         size='x-large',ha='center',va='center')
                axs.text(np.radians(330),np.radians(60),'Each\ncontour:\nno uncertainty',color=(0,0.3,0),
                         size='x-large',ha='center',va='center')
            axs.text(np.radians(30),np.radians(60),'Green\nmarker:\ntrue axis',color=(0,0.75,0),
                     size='x-large',ha='center',va='center')
            axs.text(np.radians(180),np.radians(100),'{}'.format(self.name),color='k',
                     size='x-large',ha='center',va='center')
            axs.axes.spines['polar'].set_alpha(0.1)
            axs.grid(alpha=0.1)
            axs.tick_params(axis='x',colors=(0,0,0,0.1))  # Sets alpha level of tick labels
            axs.tick_params(axis='y',colors=(0,0,0,0.1))  #
        else:
            if constraint in ['real','both']:
                axs.contourf(sols,obls,prob2d,levels=levs,colors=c_regs)
                axs.contour(sols,obls,prob2d,levels=levs,colors='0.5')
            if kind == 'single':
                if constraint == 'perf':
                    this_color = m_c(0.33+0.67*(j/entries))
                    axs.contour(orig_sols,orig_obls,kchar,levels=[k_mu],colors=[this_color],
                                linewidths=3,linestyles='solid')
                elif constraint == 'both':
                    axs.contour(orig_sols,orig_obls,kchar,levels=[k_mu],colors=[(0,0.3,0)],
                                linewidths=3,linestyles='dashed')
                if m_c == cm.Reds:
                    axs.text(np.radians(225),np.radians(110),r'$%.0f^{\circ}$' % now_phaseD,color='0.5',
                             size='x-large',ha='center',va='center')
                else:
                    axs.text(np.radians(225),np.radians(110),
                             r'$%.0f^{\circ}$' '\n' r'$%.0f^{\circ}$' % (now_phaseD[0],now_phaseD[1]),color='0.5',
                             size='x-large',ha='center',va='center')
            else:
                if not _active:
                    axs.text(np.radians(225),np.radians(110),'Combined',color='0.5',
                             size='x-large',ha='center',va='center')
            axs.scatter(solR,oblR,s=100,color=(0,1,0),edgecolor='k',marker=mark,zorder=2)

        axs.set_thetalim([0,2.0*pi])
        axs.set_rlim([0,pi/2.0])
        ts = lambda a: 'medium' if a else 'large'
        axs.set_thetagrids(np.linspace(0,315,8),sol_ticks_,size=ts(_active))  # Match numbers to sol_ticks to avoid error.
        axs.set_rgrids(np.linspace(0,pi/2.0,4),obl_ticks_,size=ts(_active))
        return s+1
    
    
    def SpinAxis_Constraints(self,phaseD_list,which='pri',constraint='both',info=True,combine=True,
                             combine_only=False,keep_probdata=False,res=500,n_sol=361,n_obl=91,
                             phaseD_sig=10.0,incD_sig=10.0,kwid_sig=10.0,kddc_sig=20.0,**kwargs):
        """Plots how observations may constrain your planet's spin axis.

        .. image:: _static/spinaxis_example.png
            :align: center

        These predictions use the kernel characteristics and assume your
        planet's light curves **at single orbital phases** are invertible
        (see note below). Discussed in Section 4 of S16.

        Say you are fitting a planet's albedo map. We know the kernel depends
        on the planet's spin axis (its obliquity and solstice). Invert a
        light curve from one orbital phase and you will also fit some
        East-West structure of the kernel, like the longitudinal width.
        Or, invert from two different phases and you fit some North-South
        structure, like the **change in** dominant colatitude. So, kernel
        characteristics help us estimate constraints on the spin axis without
        doing inversions from real data.

        Learn more about the kernel and its characteristics with
        :func:`Kernel_WidthDomColat`, :func:`Kernels_Plot`,
        and :func:`KChar_Evolve_Plot`.

        .. note::

            Inverting a light curve will depend on the quality of the
            observational data. The planet's albedo map matters too:
            East-West markings to sense daily brightness changes,
            North-South markings to sense longer changes.
            
            We have pre-calculated characteristics stored in numpy binary
            files (the obvious two with names ending "values_all5deg.npy").
            So, this method rounds inclination, obliquity and solstice
            to the nearest 5 degrees. It also tracks the North (green
            circle) or South pole (green diamond) when obliquity is less
            than or greater than 90 degrees, respectively.

        Args:
            phaseD_list (list):
                Orbital phases of the planet in degrees. Standard range
                is [0, 360). Phases are integers or floats, and list
                elements can be
                
                    - *phase* for a longitudinal width,
                    - *[phase, phase]* for a change in dominant colatitude.
                
            which (str):
                The param set to use. Can be
            
                    - 'pri' for primary (default),
                    - 'alt' for alternate,
                    - '_c' for custom, see Note below.
            
            constraint (str):
                The type of prediction. Can be
            
                    - 'perf' for perfect constraints with no data
                      uncertainties,
                    - 'real' to use uncertainties and show {1,2,3}--sigma
                      regions,
                    - 'both' (default).
            
            info (bool):
                Include a legend subplot. Default is True.
            
            combine (bool):
                Join all constraints in a separate subplot. Default is True.
            
            combine_only (bool):
                Show **only** the combo constraint. Default is False.
            
            keep_probdata (bool):
                Output all probability data, see Returns below. Default
                is False.

        Optional:
            res (int):
                Resolution when ``constraint`` is 'real', the number of
                probability contours to test. Default is 500.
            
            n_sol (int):
                Number of solstice grid points. Default is 361.
            
            n_obl (int):
                Number of obliquity grid points. Default is 91.

        Very Optional:
            **You should probably check out Section 4.1 of S16 before
            you change any of these.**
            
            phaseD_sig (float):
                Uncertainty on orbital phase, in degrees. Default is 10.0.
            
            incD_sig (float):
                Uncertainty on inclination, in degrees. Default is 10.0.
            
            kwid_sig (float):
                Uncertainty on longitudinal width, in degrees. Default
                is 10.0.
            
            kddc_sig (float):
                Uncertainty on change in dominant colatitude, in degrees.
                Default is 20.0.
                
        .. note::
                
            Keywords are used by the interactive function :func:`Sandbox_Reflection`.
            But if ``which`` is '_c', then enter your custom
            params as ``incD_I``, ``solD_I`` and ``oblD_I``.
            Standard definitions and formats apply.
            See the :class:`class and constructor <DirectImaging_Planet>` 
            docstrings.

        Effect:
            Stores this matplotlib figure as ``fig_spin``, **overwriting**
            the previous version. You can save the image later by
            calling ``fig_spin.savefig(...)``.

        Returns:
            A list (user_file) if ``keep_probdata`` is True and ``constraint``
            is **not** 'perf'.
            
                - First entry is [incD, oblD, solD].
                - Other entries are [*id*, 2D PDF, {1,2,3}--sigma
                  probability levels], where *id* is either a phaseD_list
                  element or 'Combined'.
                    
        """
        ## Default keywords
        _active = kwargs.get('_active',False)
        incD_I = kwargs.get('incD_I',85)
        solD_I = kwargs.get('solD_I',0)
        oblD_I = kwargs.get('oblD_I',0)
        
        made_combo_flag = False
        
        entries = len(phaseD_list)
        if _active:
            h,w,sub,s = 2,3,5,5
        elif combine_only:
            h,w,sub,s = 1,1,1,1
        else:
            ex = lambda x: 1 if x else 0
            sub = entries + ex(info) + ex(combine)
            h,w,s = 1+((sub-1)//3),min(sub,3),1
        p = 0
        
        if which == 'pri':
            incD,solD,oblD = self.incD,self.solD,self.oblD
        elif which == 'alt':
            incD,solD,oblD = self.incD_b,self.solD_b,self.oblD_b
        elif which == '_c':
            incD,solD,oblD = incD_I,solD_I,oblD_I
        mark = 'o'
        if oblD > 90.0:
            solD,oblD,mark = (solD % 360.0) + 180.0,180.0 - oblD,'D'
        
        
        i_i,i_s,i_o = round(incD/5),round((solD%360)/5),round(oblD/5)
        if keep_probdata:
            user_file = [[5*i_i,5*i_o,5*i_s]]
        incR,solR,oblR = np.radians(i_i*5),np.radians(i_s*5),np.radians(i_o*5)
        incR_sig = np.radians(incD_sig)
        
        combo_prob2d = np.ones(obl_2mesh_.shape)
        sigma_probs = np.array([1,0.9973,0.9545,0.6827,0])
        new_sols,new_obls = np.meshgrid(np.linspace(0,2.0*pi,n_sol),np.linspace(0,pi/2.0,n_obl),indexing='ij')
        
        if _active:
            fig = kwargs.get('fig_I','N/A')
        else:
            fig = plt.figure(figsize=(5*w,5*h))
            fig.set_facecolor('w')
            if info and not combine_only:
                s = self._spinax_style(fig=fig,h=h,w=w,s=s,m_c=cm.gray,kind='info',ax_combo='0',
                                       sols=new_sols,obls=new_obls,
                                       prob2d=0,levs=0,constraint=constraint,
                                       orig_sols=sol_2mesh_,orig_obls=obl_2mesh_,
                                       kchar=0,k_mu=0,now_phaseD=0,
                                       solR=solR,oblR=oblR,mark=0,
                                       _active=_active,j=0,entries=entries)
        
        for j in np.arange(entries):
            now_phaseD = phaseD_list[j]
            
            if isinstance(now_phaseD,(int,float)):
                p += 1
                m_c = cm.Reds
                i_p = round((now_phaseD%360)/5)
                sav_phaseD = 5*i_p
                phaseR = np.radians(i_p*5)
                phaseR_sig = np.radians(phaseD_sig)
                wid_mu = kernel_widths_[i_p,i_i,i_s,i_o]
                kchar,k_mu = kernel_widths_[i_p,i_i,:,:],wid_mu
                wid_sig = np.radians(kwid_sig)
                if constraint in ['real','both']:
                    prob2d = self._spinax_prob_original(kchar=kernel_widths_,k_mu=wid_mu,k_sig=wid_sig,
                                                        incs=inc_4mesh_,i_mu=incR,i_sig=incR_sig,
                                                        phases=phase_4mesh_,p_mu=phaseR,p_sig=phaseR_sig,obls=obl_2mesh_,
                                                        yes_p2=False,phases2='no',p2_mu='no',p2_sig='no')
                else:
                    prob2d = 1
            else:
                p += 2
                m_c = cm.Blues
                i_p,i_p2 = round((now_phaseD[0]%360)/5),round((now_phaseD[1]%360)/5)
                sav_phaseD = [5*i_p,5*i_p2]
                phaseR,phaseR2 = np.radians(i_p*5),np.radians(i_p2*5)
                phaseR_sig = np.radians(phaseD_sig)
                dom1,dom2 = kernel_domcolats_[i_p,i_i,i_s,i_o],kernel_domcolats_[i_p2,i_i,i_s,i_o]
                ddc_mu = abs(dom1-dom2)
                kchar,k_mu = np.absolute(kernel_domcolats_[i_p,i_i,:,:]-kernel_domcolats_[i_p2,i_i,:,:]),ddc_mu
                ddc_sig = np.radians(kddc_sig)
                if constraint in ['real','both']:
                    prob2d = self._spinax_prob_original(kchar=kernel_delta_domcolats_,k_mu=ddc_mu,k_sig=ddc_sig,
                                                        incs=inc_4mesh_,i_mu=incR,i_sig=incR_sig,
                                                        phases=phase_4mesh_,p_mu=phaseR,p_sig=phaseR_sig,obls=obl_2mesh_,
                                                        yes_p2=True,phases2=shifted_phase_4mesh_,
                                                        p2_mu=phaseR2,p2_sig=phaseR_sig)
                else:
                    prob2d = 1
            
            if combine or combine_only:
                combo_prob2d *= prob2d
                if made_combo_flag == False:
                    axC = fig.add_subplot(h,w,sub,projection='polar')
                    made_combo_flag = True
                if constraint in ['perf','both']:
                    if constraint == 'perf':
                        this_color = m_c(0.33+0.67*(j/entries))
                        axC.contour(sol_2mesh_,obl_2mesh_,kchar,levels=[k_mu],colors=[this_color],
                                    linewidths=3,linestyles='solid')
                    else:
                        axC.contour(sol_2mesh_,obl_2mesh_,kchar,levels=[k_mu],colors=[(0,0.3,0)],alpha=0.2,
                                    linewidths=3,linestyles='dashed')
            if constraint in ['real','both']:
                new_prob2d = self._spinax_prob_redo(prob2d=prob2d,
                                                    orig_sols=sol_2mesh_,orig_obls=obl_2mesh_,
                                                    new_sols=new_sols,new_obls=new_obls)
            else:
                new_prob2d = 1
            if not combine_only:
                if constraint in ['real','both']:
                    levels_sigma = self._spinax_leveling(prob2d=new_prob2d,sigma_probs=sigma_probs,
                                                         res=res,obls=new_obls)
                    if keep_probdata:
                        user_file.append([sav_phaseD,np.copy(new_prob2d),np.copy(levels_sigma)])
                else:
                    levels_sigma = 1
                s = self._spinax_style(fig=fig,h=h,w=w,s=s,m_c=m_c,kind='single',ax_combo='0',
                                       sols=new_sols,obls=new_obls,
                                       prob2d=new_prob2d,levs=levels_sigma,constraint=constraint,
                                       orig_sols=sol_2mesh_,orig_obls=obl_2mesh_,
                                       kchar=kchar,k_mu=k_mu,now_phaseD=sav_phaseD,
                                       solR=solR,oblR=oblR,mark=mark,
                                       _active=_active,j=j,entries=entries)
        
        if combine or combine_only:
            if constraint in ['real','both']:
                new_combo_prob2d = self._spinax_prob_redo(prob2d=combo_prob2d,
                                                          orig_sols=sol_2mesh_,orig_obls=obl_2mesh_,
                                                          new_sols=new_sols,new_obls=new_obls)
                levels_sigma = self._spinax_leveling(prob2d=new_combo_prob2d,sigma_probs=sigma_probs,
                                                     res=res,obls=new_obls)
                if keep_probdata:
                    user_file.append(['Combined',np.copy(new_combo_prob2d),np.copy(levels_sigma)])
            else:
                new_combo_prob2d,levels_sigma = 1,1
            m_c_here = lambda x: cm.Reds if x == entries else (cm.Blues if x == 2*entries else cm.Purples)
            s = self._spinax_style(fig=fig,h=h,w=w,s=s,m_c=m_c_here(p),kind='combo',ax_combo=axC,
                                   sols=new_sols,obls=new_obls,
                                   prob2d=new_combo_prob2d,levs=levels_sigma,constraint=constraint,
                                   orig_sols=sol_2mesh_,orig_obls=obl_2mesh_,
                                   kchar=kchar,k_mu=k_mu,now_phaseD=0,
                                   solR=solR,oblR=oblR,mark=mark,
                                   _active=_active,j=0,entries=entries)
        
        if _active:
            return axC  # Pass combo axis back to _actmodule_heart
        else:
            fig.tight_layout()
            self.fig_spin = fig
            plt.show()
            
            if keep_probdata:
                return user_file
    
    
    def _savebutton_click(self):
        """Directs a button to save orbital phases."""
        if self._pslot_act.value == 'all':
            self._pword_act.value = '<center><font color="red">Only save to one slot at a time</font></center>'
        else:
            word_start = '<center><font color="limegreen">Saved current phase to '
            wording = word_start+self._pslot_act.value+' slot</font></center>'
            if self._pslot_act.value == 'light':
                self._xph_lig = self._orb_act.value
            elif self._pslot_act.value == 'medium':
                self._xph_med = self._orb_act.value
            elif self._pslot_act.value == 'dark':
                self._xph_drk = self._orb_act.value
            self._pword_act.value = wording
    
    def _clearbutton_click(self):
        """Directs a button to clear orbital phases."""
        word_start = '<center><font color="orange">Cleared phase from '+self._pslot_act.value
        if self._pslot_act.value == 'all':
            self._xph_lig = 'no'
            self._xph_med = 'no'
            self._xph_drk = 'no'
            word_end = ' slots</font></center>'
        else:
            word_end = ' slot</font></center>'
            if self._pslot_act.value == 'light':
                self._xph_lig = 'no'
            elif self._pslot_act.value == 'medium':
                self._xph_med = 'no'
            elif self._pslot_act.value == 'dark':
                self._xph_drk = 'no'
        self._pword_act.value = word_start+word_end
    
    def _check_for_actspin(self,phases,switch):
        """Organizes orbital phases for spin axis constraints."""
        new_ph = []
        if switch == 'wid':
            for p in phases:
                if isinstance(p,(int,float)):
                    new_ph.append(p)
        elif switch == 'dom':
            c,n = 0,1
            for p in phases[1:]:
                if isinstance(p,(int,float)):
                    new_ph.append([phases[c],p])
                    c = n
                n += 1
        elif switch == 'both':
            c,n = 0,1
            lph = len(phases)
            for p in phases:
                if isinstance(p,(int,float)):
                    new_ph.append(p)
                if (n != lph) and isinstance(phases[n],(int,float)):
                    new_ph.append([phases[c],phases[n]])
                    c = n
                n += 1
        return new_ph
    

    def _actmodule_heart(self,phaseD_I,incD_I,oblD_I,solD_I,ratRO_I,res_I,longzeroD_I,lc_swit,spinax_swit):
        """Sets up and combines several plots about your exoplanet."""
        self._pword_act.value = '<center><font color="blue">Ready to save/clear orbital phases</font></center>'
        phasesD_single = [phaseD_I,self._xph_lig,self._xph_med,self._xph_drk]
        phasesD_forspin = self._check_for_actspin(phasesD_single,spinax_swit)
        ph_colors = [(1,0,1),cm.gray(0.6),cm.gray(0.3),cm.gray(0)]
        orbT_I = 24.0*365.0

        see_spins = abs(ratRO_I)/72.0
        num_rel = max(res_I*round(see_spins),self.n_long)
        rel_tphase = np.linspace(-2.5,2.5,num_rel)
        
        fig_I = plt.figure(figsize=(14,9.3))
        fig_I.set_facecolor('w')
        
        ### Geom
        axg = fig_I.add_subplot(232)
        self.Geometry_Diagram(which='N/A',_active=True,ax_I=axg,
                              incD=incD_I,oblD=oblD_I,solD=solD_I,ratRO=ratRO_I,
                              phaseD=phasesD_single,ph_colors=ph_colors)
        
        ### Ortho, sub 231 and 233
        self.Orthographic_Viewer(phaseD_I,show='both',_active=True,fig_I=fig_I,
                                 orbT_I=orbT_I,ratRO_I=ratRO_I,
                                 incD_I=incD_I,oblD_I=oblD_I,solD_I=solD_I,
                                 longzeroD_I=longzeroD_I)
        
        ### Light
        axl = fig_I.add_subplot(234)
        n = 0
        for p in phasesD_single:
            if isinstance(p,(int,float)):
                times_I = orbT_I*((p + rel_tphase)/360.0)
                self.LightCurve_Plot(alt=False,show=lc_swit,_active=True,ax_I=axl,
                                     times_I=times_I,orbT_I=orbT_I,ratRO_I=ratRO_I,
                                     incD_I=incD_I,oblD_I=oblD_I,solD_I=solD_I,
                                     longzeroD_I=longzeroD_I,ph_color=ph_colors[n],now_I=n)
            n += 1
        n = 0
        axl.set_xlim([-2.5,2.5])
        axl.set_xticks(np.linspace(-2,2,5))
        axl.set_xticklabels(relph_ticks_,size='medium')
        axl.set_xlabel('Relative Orbital Phase',size='medium')
        axl.tick_params(axis='y',labelsize='medium')
        ylab = lambda lc: 'Flux' if lc == 'flux' else ('Apparent Brightness' if lc == 'appar' else '')
        axl.set_ylabel(ylab(lc_swit),size='medium')
        axl.set_aspect(1.0/axl.get_data_ratio())
        axl.text(0.25,1.01,'Light Curve',color='k',size='medium',ha='center',va='bottom',
                 transform=axl.transAxes)
        axl.text(0.75,1.01,'Rotations: {:.2f}'.format(see_spins),color='k',size='medium',ha='center',va='bottom',
                 transform=axl.transAxes)
        
        ### Kernel, sub 236
        axk = self.KChar_Evolve_Plot('both',which='_c',incD=incD_I,oblD=oblD_I,solD=solD_I,
                                     _active=True,fig_I=fig_I,phasesD_I=phasesD_single,ph_colors=ph_colors)
        axk.text(0.5,1.01,'Kernel Characteristics',color='k',size='medium',ha='center',va='bottom',
                 transform=axk.transAxes)
        
        ### SpinAx, polar sub 235
        if len(phasesD_forspin) == 0:
            axs = fig_I.add_subplot(235,projection='polar')
            axs.set_theta_zero_location('S')
            axs.set_rlabel_position(45)
            axs.set_xticks(np.linspace(0,1.75*pi,8))  # Match numbers to sol_ticks to avoid error.
            axs.set_xticklabels(sol_ticks_,size='medium',alpha=0.1)
            axs.set_yticks(np.linspace(0,pi/2.0,4))
            axs.set_yticklabels(obl_ticks_,size='medium',alpha=0.1)
            axs.axes.spines['polar'].set_alpha(0.1)
            axs.grid(alpha=0.1)
            bads = ('SPIN AXIS\nCONSTRAINT WARNING:\n\nYOU NEED\nAT LEAST 2 PHASES TO'
                    '\nCALCULATE CHANGES IN\nDOMINANT COLATITUDE')
            axs.text(np.radians(0),np.radians(0),bads,color=(1.0,0.5,0),size='x-large',
                     ha='center',va='center',weight='bold')
        else:
            axs = self.SpinAxis_Constraints(phasesD_forspin,which='_c',constraint='perf',
                                            info=False,combine=False,combine_only=True,_active=True,
                                            fig_I=fig_I,incD_I=incD_I,solD_I=solD_I,oblD_I=oblD_I)
            axs.text(np.radians(225),np.radians(112),'Spin Axis\nConstraints',color='k',size='medium',
                     ha='center',va='center')
        
        fig_I.tight_layout()
        self.fig_sand = fig_I
        plt.show()
    
    def _reset_actmodule(self):
        """Resets attributes for the interactive module."""
        self._xph_lig = 'no'
        self._xph_med = 'no'
        self._xph_drk = 'no'

        self._orb_act.close()
        self._inc_act.close()
        self._obl_act.close()
        self._sol_act.close()
        self._ratRO_act.close()
        self._res_act.close()
        self._zlong_act.close()
        self._ligcur_act.close()
        self._spax_act.close()
        self._pslot_act.close()
        self._pword_act.close()
        self._psav_act.close()
        self._pclr_act.close()
        self._title_act.close()

        self._orb_act.open()
        self._inc_act.open()
        self._obl_act.open()
        self._sol_act.open()
        self._ratRO_act.open()
        self._res_act.open()
        self._zlong_act.open()
        self._ligcur_act.open()
        self._spax_act.open()
        self._pslot_act.open()
        self._pword_act.open()
        self._psav_act.open()
        self._pclr_act.open()
        self._title_act.open()
        

        self._orb_act.value = 0
        self._inc_act.value = 85
        self._obl_act.value = 0
        self._sol_act.value = 0

        self._ratRO_act.value = 72
        self._res_act.value = 101
        self._zlong_act.value = 0

        self._ligcur_act.value = 'flux'
        self._spax_act.value = 'wid'
        self._pslot_act.value = 'light'
        first_pword = '<center><font color="blue">Ready to save/clear orbital phases</font></center>'
        self._pword_act.value = first_pword
    
    
    def Sandbox_Reflection(self):
        """Creates an interactive module about your directly imaged planet.

        .. image:: _static/sandref_example.png
            :align: center

        This module lets you explore how a planet's geometry, motion,
        kernel and light curves are related. You can also see predicted
        constraints on the planet's spin axis (using the kernel and perfect
        data).

        .. note::
        
            The larger your ``n_clat`` and ``n_long``, the longer this
            module takes to update (e.g. seconds with default values).

        The sandbox combines several methods from the class
        :class:`DirectImaging_Planet` into one display. See each
        for details:
        
            - :func:`Geometry_Diagram`
            - :func:`Orthographic_Viewer`
            - :func:`Light_Curves`
            - :func:`KChar_Evolve_Plot`
            - :func:`SpinAxis_Constraints`
        
        The planet and light curves are rendered with your primary albedo
        map. You view a main orbital phase (magenta) and can compare
        up to 3 extra phases (light, medium, dark). Each phase
        has a color-coded marker on the geometry diagram and kernel
        characteristics plot, plus its own light curve.

        There are many controls (all angles in degrees):
        
            - Inclination
            - Obliquity
            - Solstice
            - Orbital Phase
                - [which] Extra Phase Slot
                - Save [extra phase]
                - Clear [extra phase(s)]
            - Spins per Orbit
            - Time Steps per Spin
            - Initial Longitude [at zero phase]
            - [type of] Light Curve
            - [type of] Axis Constraint

        .. note::
                
            For the subplot of spin axis constraints, curves are
            colored lightest to darkest in the phase order [main, light,
            medium, dark]. Red curves are for single phases, blue curves
            pairs of phases.

        Effect:
            Stores this matplotlib figure as ``fig_sand`` **whenever you
            interact with the module**. You can save the image later by
            calling ``fig_sand.savefig(...)``.
            
        """
        self._reset_actmodule()
        
        ios_col = widgets.Box([self._inc_act,self._obl_act,self._sol_act],
                              layout=Layout(flex_flow='column',width='45%'))
        rrz_col = widgets.Box([self._ratRO_act,self._res_act,self._zlong_act],
                              layout=Layout(flex_flow='column',width='30%'))
        tilisp_col = widgets.Box([self._title_act,self._ligcur_act,self._spax_act],
                                 layout=Layout(flex_flow='column',align_self='center',width='25%'))
        top_row = widgets.Box([tilisp_col,ios_col,rrz_col])
        
        savclr_col = widgets.Box([self._psav_act,self._pclr_act],
                                 layout=Layout(flex_flow='column',width='25%'))
        info_col = widgets.Box([self._pword_act,self._pslot_act],
                               layout=Layout(flex_flow='column',width='25%'))
        bot_row = widgets.Box([self._orb_act,info_col,savclr_col])
        
        the_connections = {'phaseD_I':self._orb_act,'incD_I':self._inc_act,'oblD_I':self._obl_act,
                           'solD_I':self._sol_act,'ratRO_I':self._ratRO_act,'res_I':self._res_act,
                           'longzeroD_I':self._zlong_act,'lc_swit':self._ligcur_act,'spinax_swit':self._spax_act}
        inter_out = widgets.interactive_output(self._actmodule_heart,the_connections)
        
        IPy_display(widgets.Box([top_row,bot_row,inter_out],layout=Layout(flex_flow='column')))
        