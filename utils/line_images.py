import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from matplotlib.patches import Ellipse  # this is for the beam contour

from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit



from astropy.cosmology import Planck15
cosmo = Planck15

####################################################################################
####################################################################################

def format_sci(value, error, precision=2):
    import numpy as np
    
    if value == 0:
        return f"(0 ± {error:.{precision}e})"
    
    exponent = int(np.floor(np.log10(abs(value))))
    
    value_scaled = value / 10**exponent
    error_scaled = error / 10**exponent
    
    return f"({value_scaled:.{precision}f} ± {error_scaled:.{precision}f}) × 10^{exponent}"


def ReadFITS(FileName):
    FITSFile = fits.open(FileName, lazy_load_hdu=True)
    Data = FITSFile[0].data[0][0]   # assumes there are four axix (RA, DEC, Freq, Polarisation). Select only the first two.
    Header = FITSFile[0].header
    FITSFile.close()
    return Data, Header


####################################################################################
####################################################################################

def MakePlotCont(File, 
                 Title, 
                 Name, 
                 center_x=0, 
                 center_y=0, 
                 radius=5,
                 contour_levels=None):
    
    print (File)
    Offset = np.asarray((0,0,0,0))
    #read the fits files
    Data, Header = ReadFITS(File) # Data is in Jy/beam
    Bmaj = Header['BMAJ']*3600.0    # major axis, convert deg -> arcsec
    Bmin = Header['BMIN']*3600.0
    Bpa = Header['BPA']+90

    print ("Beam: %.1f %.1f %.1f" % (Bmaj, Bmin, Bpa))

    PxScale = abs(Header['CDELT1']*3600.0) #arcsec
    DimX=Header['NAXIS1']
    DimY=Header['NAXIS2']
    Extent0=np.asarray([-DimX*PxScale/2.,DimX*PxScale/2.,-DimX*PxScale/2.,DimX*PxScale/2.])+Offset
    
    x_coords = np.linspace(Extent0[0], Extent0[1], DimX)
    y_coords = np.linspace(Extent0[2], Extent0[3], DimY)
    X, Y = np.meshgrid(x_coords, y_coords)


    ####### RMS #######

    rms_mask = ((X-center_x)**2 + (Y-center_y)**2) >= (radius + 5)**2
    RMS = np.nanstd(Data[rms_mask])
    print ("rms = %.2e Jy/beam" % RMS)    
    

    ###################

    mask = (X**2 + Y**2) <= 10**2 
    masked_data = np.where(mask, Data, np.nan)

    # index of maximum value inside mask
    peak_index = np.unravel_index(np.nanargmax(masked_data), Data.shape)
    peak_y_idx, peak_x_idx = peak_index

    # convert to coordinate values (arcsec)
    peak_x = x_coords[peak_x_idx]
    peak_y = y_coords[peak_y_idx]

    print(f"Peak position inside 3'': x = {peak_x:.2f} arcsec, y = {peak_y:.2f} arcsec")

    peak = Data[peak_index]
    print(f"Peak flux density : {peak*1e3:.2f} mJy/beam")
    print(f"Peak S/N          : {peak/RMS:.2f}")

    ######################

    # make the actual figure
    fig1 = plt.figure()
    ax=fig1.add_subplot(111)
    # show coordinate ticks
    ax.set_xlabel('RA Offset (arcsec)', fontsize=14)
    ax.set_ylabel('Dec Offset (arcsec)', fontsize=14)

    ax.tick_params(axis='both',
                direction='in',
                length=6,
                width=1,
                colors='black',
                labelsize=12)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

    plt.xlim(-20,20)
    plt.ylim(-20,20)


    # actual picture
    ax1=plt.imshow(Data*1e3, cmap = 'viridis', origin='lower', interpolation = 'none', extent =Extent0)

    # add contours at S/N
    if contour_levels is None:
        contour_levels = [-4,-3,-2,2,3,4,5,6,7,8,9,10,11, 12, 13, 14, 15, 16, 17, 18, 19 ,20]

    contours = plt.contour(Data/RMS, levels=contour_levels, colors =['0'], linewidths =[1], extent =Extent0)

    # Add colorbar
    cb = plt.colorbar(ax1, fraction=0.035)
    cb.set_label('mJy/beam', fontsize = 12)


    # get axis limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # margins so things are not exactly on the edge
    xmargin = 0.07 * (xmax - xmin)
    ymargin = 0.07 * (ymax - ymin)

    circle = plt.Circle((center_x, center_y), radius, color='1', fill=False, linestyle='--', linewidth=1)
    ax.add_patch(circle)


    # add the synthesised beam (bottom-left)
    Beam = Ellipse(
        (xmin + xmargin, ymin + ymargin),
        width=Bmaj,
        height=Bmin,
        angle=Bpa,
        hatch='/////',
        fc='none',
        ec='1',
        lw=1,
        zorder=10
    )
    ax.add_patch(Beam)

    # add the title (top-left)
    ax.text(
        xmin + xmargin,
        ymax - ymargin,
        Title,
        color='1',
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=18
    )
    fig1.savefig(Name,bbox_inches='tight')
    return contours

####################################################################################
####################################################################################

def MakePlot(File, 
             Title, 
             Name, 
             totalwidth, 
             redshift, 
             restfreq, 
             lens_factor, 
             continuum_contours=None, 
             detection=True, 
             aperture=False, 
             radius=5,
             center_x=0, center_y=0,
             twosigma=False
             ):
    
    print (File)
    Offset = np.asarray((0,0,0,0))
    Data, Header = ReadFITS(File) # Data is in Jy/beam
    Bmaj = Header['BMAJ']*3600.0    # major axis, convert deg -> arcsec
    Bmin = Header['BMIN']*3600.0
    Bpa = Header['BPA']+90

    print ("Beam: %.1f %.1f %.1f" % (Bmaj, Bmin, Bpa))

    PxScale = abs(Header['CDELT1']*3600.0) #arcsec
    DimX=Header['NAXIS1']
    DimY=Header['NAXIS2']
    Extent0=np.asarray([-DimX*PxScale/2.,DimX*PxScale/2.,-DimX*PxScale/2.,DimX*PxScale/2.])+Offset
    
    x_coords = np.linspace(Extent0[0], Extent0[1], DimX)
    y_coords = np.linspace(Extent0[2], Extent0[3], DimY)
    X, Y = np.meshgrid(x_coords, y_coords)


    ####### Velocity widths #######
    # Velocity width per channel
    delta_Freq = Header['CDELT3']
    Freq = Header['CRVAL3']
    print("Velocity Width per channel = ", 3e5/Freq*delta_Freq, "km/s")

    print("Total velocity width = ", totalwidth, "km/s")


    ##########################
    peak_x = center_x
    peak_y = center_y

    # Setup a mask areound the correct peak position with the radius we want to use for the flux calculation
    mask = ((X-peak_x)**2 + (Y-peak_y)**2) <= radius**2  

    # beam and pixel areas (arcsec^2)
    # https://science.nrao.edu/facilities/vla/proposing/TBconv
    beam_area = np.pi * Bmaj * Bmin / (4 * np.log(2))
    pixel_area = PxScale**2
    pix_per_beam = beam_area / pixel_area

    # aperture area (arcsec^2)
    aperture_area = np.pi * radius**2
    pix_per_aperture = aperture_area / pixel_area

    # number pixels inside aperture / number of pixels inside beam
    N_beams = aperture_area / beam_area


    ####### RMS #######
    # Estimate RMS outside the source aperture
    rms_mask = ((X - peak_x)**2 + (Y - peak_y)**2) >= radius**2

    # RMS in Jy/beam
    RMS = np.nanstd(Data[rms_mask])
    print(f"RMS = {RMS:.2e} Jy/beam")

    ######## FLUX CALCULATION ########

    # Aperture mask
    mask = ((X - peak_x)**2 + (Y - peak_y)**2) <= radius**2

    # Number of pixels inside aperture
    N_pix = np.count_nonzero(mask)

    # Flux density (Jy)
    flux_jy = np.nansum(Data[mask]) / pix_per_beam

    # Uncertainty on flux density (Jy)
    RMS_jy = RMS * np.sqrt(N_pix) / pix_per_beam

    # Integrated line flux (Jy km/s)
    flux_jykms = flux_jy * totalwidth

    # Uncertainty (Jy km/s)
    RMS_jykms = RMS_jy * totalwidth

    # Peak S/N
    peak_sn = np.nanmax(Data[mask]) / RMS

    # Integrated S/N
    general_sn = flux_jykms / RMS_jykms

    print(f"Flux density : {flux_jy:.3e} ± {RMS_jy:.3e} Jy")
    if detection:
        print(f"I {radius}'' : {flux_jykms:.3e} ± {RMS_jykms:.3e} Jy km/s")
    else:
        print(f"I {radius}'' : {3*RMS_jykms:.3e} Jy km/s")
    print(f"Peak S/N : {peak_sn:.2f}")
    print(f"Integrated S/N : {general_sn:.2f}")

    ############################################################################

    # Luminosity and Prime Luminosity calculation
    luminosity_distance = cosmo.luminosity_distance(redshift).to('Mpc').value
    obsfreq = restfreq / (1+redshift)

    if detection:
        # SKY LUMINOSITY
        L_prime_line =  3.25e7 * flux_jykms* 1.32 * (luminosity_distance**2) / (((1+redshift)**3)*(obsfreq**2))
        L_prime_line_error = 3.25e7 * RMS_jykms* 1.32* (luminosity_distance**2) / (((1+redshift)**3)*(obsfreq**2))

        L_line = 3e-11 * restfreq**3 * L_prime_line
        L_line_error = 3e-11 * restfreq**3 * L_prime_line_error

        print('#################################################')

        print(f"L_sky: {format_sci(L_line, L_line_error)} L_sun")
        print(f"L'_sky: {format_sci(L_prime_line, L_prime_line_error)} K.km/s.pc^2")



        # INTRINSIC LUMINOSITY
        L_line_intrinsic = L_line / lens_factor
        L_line_intrinsic_error = L_line_error / lens_factor

        L_prime_line_intrinsic = L_prime_line / lens_factor
        L_prime_line_intrinsic_error = L_prime_line_error / lens_factor

        print('#################################################')

        print(f"L_intrinsic: {format_sci(L_line_intrinsic, L_line_intrinsic_error)} L_sun")
        print(f"L'_intrinsic: {format_sci(L_prime_line_intrinsic, L_prime_line_intrinsic_error)} K.km/s.pc^2")

        print('#################################################')

    else:
        # SKY LUMINOSITY
        L_prime_line =  3.25e7 * 3*RMS_jykms * 1.32* (luminosity_distance**2) / (((1+redshift)**3)*(obsfreq**2))
        L_prime_line_error = 3.25e7 * RMS_jykms* 1.32*  (luminosity_distance**2) / (((1+redshift)**3)*(obsfreq**2))

        L_line = 3e-11 * restfreq**3 * L_prime_line
        L_line_error = 3e-11 * restfreq**3 * L_prime_line_error

        print('#################################################')

        print(f"L_sky: {format_sci(L_line, L_line_error)} L_sun")
        print(f"L'_sky: {format_sci(L_prime_line, L_prime_line_error)} K.km/s.pc^2")

        # INTRINSIC LUMINOSITY
        L_line_intrinsic = L_line / lens_factor
        L_line_intrinsic_error = L_line_error / lens_factor

        L_prime_line_intrinsic = L_prime_line / lens_factor
        L_prime_line_intrinsic_error = L_prime_line_error / lens_factor

        print('#################################################')

        print(f"L_intrinsic: {format_sci(L_line_intrinsic, L_line_intrinsic_error)} L_sun")
        print(f"L'_intrinsic: {format_sci(L_prime_line_intrinsic, L_prime_line_intrinsic_error)} K.km/s.pc^2")

        print('################################################_')

##############################################################################################
    # make the actual figure
    fig1 = plt.figure()
    ax=fig1.add_subplot(111)
    # show coordinate ticks
    ax.set_xlabel('RA Offset (arcsec)', fontsize=14)
    ax.set_ylabel('Dec Offset (arcsec)', fontsize=14)

    ax.tick_params(axis='both',
                direction='in',
                length=6,
                width=1,
                colors='black',
                labelsize=12)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

    plt.xlim(-20,20)
    plt.ylim(-20,20)


    # actual picture
    ax1=plt.imshow(Data, cmap = 'viridis', origin='lower', interpolation = 'none', extent =Extent0)

    # add contours at S/N
    plt.contour(Data/RMS, levels=[-4,-3,-2,2,3,4,5,6,7,8,9,10], colors =['0'], linewidths =[1], extent =Extent0)

    # Add contours of continuum
    if continuum_contours is not None:
        plt.contour(continuum_contours, levels=[2,3,4,5,6,7,8,9,10], colors =['1'], linewidths =[1], extent =Extent0)


    if aperture:
    # Add circle
        circle = plt.Circle((peak_x, peak_y), radius, color='1', fill=False, linestyle='--', linewidth=1)
        ax.add_patch(circle)

    # Add colorbar
    cb = plt.colorbar(ax1, fraction=0.035)
    cb.set_label('mJy/beam', fontsize = 12)

    # get axis limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # margins so things are not exactly on the edge
    xmargin = 0.05 * (xmax - xmin)
    ymargin = 0.05 * (ymax - ymin)

    # add the synthesised beam (bottom-left)
    Beam = Ellipse(
        (xmin + xmargin, ymin + ymargin),
        width=Bmaj,
        height=Bmin,
        angle=Bpa,
        hatch='/////',
        fc='none',
        ec='1',
        lw=1,
        zorder=10
    )
    ax.add_patch(Beam)

    # add the title (top-left)
    ax.text(
        xmin + xmargin,
        ymax - ymargin,
        Title,
        color='1',
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=18
    )
    fig1.savefig(Name,bbox_inches='tight')

####################################################################################
####################################################################################

def MakePlotClean(File, 
                  Title, 
                  Name, 
                  velwidth, 
                  redshift, 
                  restfreq, 
                  lens_factor, 
                  continuum_contours=None, 
                  detection=True, 
                  aperture=False, 
                  radius=5,
                  center_x=0, center_y=0,
                  twosigma= False,
                  size=20,
                  show_offset=True,
                  contour_levels=None
                  ):
    
    print (File)
    Offset = np.asarray((0,0,0,0))

    #read the fits files
    Data, Header = ReadFITS(File) # Data is in Jy/beam.km/s
    Bmaj = Header['BMAJ']*3600.0    # major axis, convert deg -> arcsec
    Bmin = Header['BMIN']*3600.0
    Bpa = Header['BPA']+90

    print ("Beam: %.1f %.1f %.1f" % (Bmaj, Bmin, Bpa))

    PxScale = abs(Header['CDELT1']*3600.0) #arcsec
    DimX=Header['NAXIS1']
    DimY=Header['NAXIS2']
    Extent0=np.asarray([-DimX*PxScale/2.,DimX*PxScale/2.,-DimX*PxScale/2.,DimX*PxScale/2.])+Offset
    
    print("Pixel scale: ", PxScale, "arcsec/pixel")

    x_coords = np.linspace(Extent0[0], Extent0[1], DimX)
    y_coords = np.linspace(Extent0[2], Extent0[3], DimY)
    X, Y = np.meshgrid(x_coords, y_coords)

    print("###################################")

    ##########################

    peak_x = center_x
    peak_y = center_y

    # Setup a mask areound the correct peak position with the radius we want to use for the flux calculation
    mask = ((X-peak_x)**2 + (Y-peak_y)**2) <= radius**2  

    print(f"Peak position inside 8'': x = {peak_x:.2f} arcsec, y = {peak_y:.2f} arcsec")


    print("###################################")

    # beam and pixel areas (arcsec^2)
    # https://science.nrao.edu/facilities/vla/proposing/TBconv
    beam_area = np.pi * Bmaj * Bmin / (4 * np.log(2))


    pixel_area = PxScale**2
    pix_per_beam = beam_area / pixel_area

    # aperture area (arcsec^2)
    aperture_area = np.pi * radius**2
    pix_per_aperture = aperture_area / pixel_area

    # number pixels inside aperture / number of pixels inside beam
    N_beams = aperture_area / beam_area
    print('Number of Beams', N_beams)

    ####### RMS #######
    # MASK BIGGER THAN 10 ARCSEC TO AVOID THE SOURCE
    rms_mask = (X**2 + Y**2) >= 20**2

    # Jy/beam
    RMS = np.nanstd(Data[rms_mask])
    print ("rms = %.2e Jy/beam" % RMS)    

    # Jy
    RMS_jy = RMS * np.sqrt(N_beams)
    
    # Jy.km/s
    RMS_jykms = RMS_jy * velwidth

    ######## FLUX CALCULATION ########
    # SIZE OF THE GALAXY

    # sum moment-0 values, Jy/beam.km/s
    sum = np.nansum(Data[mask])

    flux_jy = sum / pix_per_beam

    # Jy km/s
    flux_jykms = flux_jy * velwidth

    # Peak S/N
    peak_sn = np.nanmax(Data[mask]) / RMS

    
    print(f"Sum {radius}'' : {sum} Jy/beam")
    print(f"Flux density {radius}'' : {flux_jy:.3e} Jy")

    if detection:
        print(f"I {radius}'' : {flux_jykms:.3e} ± {RMS_jykms:.3e} Jy km/s")
    else:
        print(f"I {radius}'' : {3*RMS_jykms:.3e} Jy km/s")
    
    print(f"Peak S/N : {peak_sn:.2f}")

    ###########################################################################
    if twosigma:
        # sum moment-0 values, Jy/beam.km/s
        detection_mask = mask & (Data >= 2 * RMS)

        beam_area = np.pi * Bmaj * Bmin / (4 * np.log(2))


        pixel_area = PxScale**2
        pix_per_beam = beam_area / pixel_area

        aperture_area = len(Data[detection_mask]) * pixel_area
        pix_per_aperture = aperture_area / pixel_area

        # number pixels inside aperture / number of pixels inside beam
        N_beams = aperture_area / beam_area
        print('Number of Beams', N_beams)

        ####### RMS #######
        # MASK BIGGER THAN 10 ARCSEC TO AVOID THE SOURCE
        rms_mask = (X**2 + Y**2) >= 20**2

        # Jy/beam
        RMS = np.nanstd(Data[rms_mask])
        print ("rms = %.2e Jy/beam" % RMS)    

        # Jy
        RMS_jy = RMS * np.sqrt(N_beams)
        
        # Jy.km/s
        RMS_jykms = RMS_jy * velwidth

        ######## FLUX CALCULATION ########
        # SIZE OF THE GALAXY

        # sum moment-0 values only in significant pixels
        sum = np.nansum(Data[detection_mask])

        flux_jy = sum / pix_per_beam

        # Jy km/s
        flux_jykms = flux_jy * velwidth

        # Peak S/N
        peak_sn = np.nanmax(Data[mask]) / RMS

        
        print(f"Sum {radius}'' : {sum} Jy/beam")
        print(f"Flux density {radius}'' : {flux_jy:.3e} Jy")

        print(f"I {radius}'' : {flux_jykms:.3e} ± {RMS_jykms:.3e} Jy km/s")
        
        print(f"Peak S/N : {peak_sn:.2f}")


    ############################################################################

    # Luminosity and Prime Luminosity calculation
    luminosity_distance = cosmo.luminosity_distance(redshift).to('Mpc').value
    obsfreq = restfreq / (1+redshift)

    if detection:
        # SKY LUMINOSITY
        L_prime_line =  3.25e7 * flux_jykms* 1.32 * (luminosity_distance**2) / (((1+redshift)**3)*(obsfreq**2))
        L_prime_line_error = 3.25e7 * RMS_jykms* 1.32 *  (luminosity_distance**2) / (((1+redshift)**3)*(obsfreq**2))

        L_line = 3e-11 * restfreq**3 * L_prime_line
        L_line_error = 3e-11 * restfreq**3 * L_prime_line_error

        print('#################################################')

        print(f"L_sky: {format_sci(L_line, L_line_error)} L_sun")
        print(f"L'_sky: {format_sci(L_prime_line, L_prime_line_error)} K.km/s.pc^2")



        # INTRINSIC LUMINOSITY
        L_line_intrinsic = L_line / lens_factor
        L_line_intrinsic_error = L_line_error / lens_factor

        L_prime_line_intrinsic = L_prime_line / lens_factor
        L_prime_line_intrinsic_error = L_prime_line_error / lens_factor

        print('#################################################')

        print(f"L_intrinsic: {format_sci(L_line_intrinsic, L_line_intrinsic_error)} L_sun")
        print(f"L'_intrinsic: {format_sci(L_prime_line_intrinsic, L_prime_line_intrinsic_error)} K.km/s.pc^2")

        print('#################################################')

    else:
        # SKY LUMINOSITY
        L_prime_line =  3.25e7 * 3*RMS_jykms * 1.32* (luminosity_distance**2) / (((1+redshift)**3)*(obsfreq**2))
        L_prime_line_error = 3.25e7 * RMS_jykms*  (luminosity_distance**2) / (((1+redshift)**3)*(obsfreq**2))

        L_line = 3e-11 * restfreq**3 * L_prime_line
        L_line_error = 3e-11 * restfreq**3 * L_prime_line_error

        print('#################################################')

        print(f"L_sky: {format_sci(L_line, L_line_error)} L_sun")
        print(f"L'_sky: {format_sci(L_prime_line, L_prime_line_error)} K.km/s.pc^2")

        # INTRINSIC LUMINOSITY
        L_line_intrinsic = L_line / lens_factor
        L_line_intrinsic_error = L_line_error / lens_factor

        L_prime_line_intrinsic = L_prime_line / lens_factor
        L_prime_line_intrinsic_error = L_prime_line_error / lens_factor

        print('#################################################')

        print(f"L_intrinsic: {format_sci(L_line_intrinsic, L_line_intrinsic_error)} L_sun")
        print(f"L'_intrinsic: {format_sci(L_prime_line_intrinsic, L_prime_line_intrinsic_error)} K.km/s.pc^2")

        print('################################################_')

##############################################################################################
    # make the actual figure
    fig1 = plt.figure()
    ax=fig1.add_subplot(111)

    if show_offset:
        ax.set_xlabel('RA Offset (arcsec)', fontsize=14)
        ax.set_ylabel('Dec Offset (arcsec)', fontsize=14)

        ax.tick_params(
            axis='both',
            direction='in',
            length=6,
            width=1,
            colors='black',
            labelsize=12
        )

        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

    else:
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Remove tick marks and tick labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw a 5" scale bar in the bottom-right
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        bar_length = 5.0  # arcsec
        margin = 0.5      # arcsec from the edge

        # Bottom-right position
        x_end = max(x0, x1) - margin
        x_start = x_end - bar_length
        y_bar = min(y0, y1) + margin

        ax.plot([x_start, x_end], [y_bar, y_bar],
                color='white', lw=2.5, solid_capstyle='butt')

        ax.text((x_start + x_end) / 2, y_bar + 0.5,
                "5''",
                color='white',
                ha='center',
                va='bottom',
                fontsize=12)

    half_size = size / 2

    plt.xlim(-half_size+center_x, half_size+center_x)
    plt.ylim(-half_size+center_y, half_size+center_y)


    # actual picture
    ax1=plt.imshow(Data*1e3, cmap = 'viridis', origin='lower', interpolation = 'none', extent =Extent0)

    if contour_levels is None:
        contour_levels = [-4,-3,-2,2,3,4,5,6,7,8,9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19 ,20]


    # add contours at S/N
    plt.contour(Data/RMS, levels=contour_levels, colors =['0'], linewidths =[1], extent =Extent0)

    # Add contours of continuum
    if continuum_contours is not None:
        plt.contour(continuum_contours, levels=[2,3,4,5,6,7,8,9,10], colors =['1'], linewidths =[1], extent =Extent0)


    if aperture:
    # Add circle
        circle = plt.Circle((peak_x, peak_y), radius, color='1', fill=False, linestyle='--', linewidth=1)
        ax.add_patch(circle)

    # Add colorbar
    cb = plt.colorbar(ax1, fraction=0.035)
    cb.set_label('mJy/beam', fontsize = 12)

    # get axis limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # margins so things are not exactly on the edge
    xmargin = 0.13 * (xmax - xmin)
    ymargin = 0.13 * (ymax - ymin)

    # add the synthesised beam (bottom-left)
    Beam = Ellipse(
        (xmin + xmargin, ymin + ymargin),
        width=Bmaj,
        height=Bmin,
        angle=Bpa,
        hatch='/////',
        fc='none',
        ec='1',
        lw=1,
        zorder=10
    )
    ax.add_patch(Beam)

    # margins so things are not exactly on the edge
    xmargin = 0.05 * (xmax - xmin)
    ymargin = 0.05 * (ymax - ymin)


    # add the title (top-left)
    ax.text(
        xmin + xmargin,
        ymax - ymargin,
        Title,
        color='1',
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=18
    )
    fig1.savefig(Name,bbox_inches='tight')


#############################################################################
##############################################################################

def MakeSpectrum(File, 
                 Title, 
                 Z, 
                 lines_dict, 
                 xlim=None, ylim=None,
                 bin_size=10, fit_window=0.1, 
                 center_guess=None,
                 minTicks=50,
                 maxTicks=100,
                 fit=True):

    C = 299792.458  # km/s
    FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2))

    # Load data
    data = np.genfromtxt(File, usecols=(2, 4), skip_header=1)
    freq_raw = data[:, 0]
    flux_raw = data[:, 1] * 1.e6  # Jy → uJy

    # Binning
    Freq = freq_raw
    Flux = flux_raw
    if bin_size > 1:
        n = len(freq_raw) // bin_size
        Freq = freq_raw[:n*bin_size].reshape(n, bin_size).mean(axis=1)
        Flux = flux_raw[:n*bin_size].reshape(n, bin_size).mean(axis=1)

    # Channel width information
    channel_width = 2e6  # Hz
    nu_mean = np.median(Freq) * 1e9  # GHz → Hz

    dv_channel = C * (channel_width / nu_mean)
    dv_bin = dv_channel * bin_size

    print(f"Channel width = {dv_channel:.2f} km/s")
    print(f"Binned velocity width = {dv_bin:.2f} km/s")

    # Gaussian model
    def gaussian(x, amp, mean, sigma):
        return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    # Plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.ylabel('Flux density [uJy]', fontsize=10)
    plt.xlabel('Frequency (observed) [GHz]', fontsize=10)
    
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(maxTicks))
    ax.yaxis.set_minor_locator(MultipleLocator(minTicks))

    plt.tick_params(axis='both', which='major',
                    length=10, direction='in', width=1, labelsize=10)
    plt.tick_params(axis='both', which='minor',
                    length=5, direction='in', width=0.5)


    plt.xlim(*xlim)
    YMin, YMax = ylim
    plt.ylim(YMin, YMax)


    ###################################################
    # Plot spectral lines
    colors = ['#4285f4', '#D72000FF', '#2CA030FF']

    for i, (line_name, nu_rest) in enumerate(lines_dict.items()):

        nu_obs = nu_rest / (1 + Z)
        color = colors[i % len(colors)]

        ax.axvline(nu_obs, c=color, lw=1,
                   linestyle='dashed', zorder=5)

        plt.text(
            nu_obs+0.015,
            YMax,
            line_name,
            color=color,
            fontsize=8,
            rotation=90,          # make text vertical
            va='top',             # align top of text to y-position
            ha='center'           # keep it centered on x
        )
        
        if fit:
            # Determine center guess
            if isinstance(center_guess, dict):
                mean_guess = center_guess.get(line_name, nu_obs)
            else:
                mean_guess = center_guess if center_guess is not None else nu_obs

            # Fit window
            fit_mask = (Freq >= mean_guess - fit_window) & \
                    (Freq <= mean_guess + fit_window)

            if fit_mask.sum() >= 5:

                x_fit = Freq[fit_mask]
                y_fit = Flux[fit_mask]

                offset0 = np.median(y_fit)
                y_fit_zero = y_fit - offset0

                # Robust amplitude guess
                amp0 = y_fit_zero[np.argmax(np.abs(y_fit_zero))]

                try:
                    popt, pcov = curve_fit(
                        gaussian,
                        x_fit,
                        y_fit_zero,
                        p0=[amp0, mean_guess, fit_window/4],
                        maxfev=5000
                    )

                    amp, mean, sigma = popt

                    fwhm = FWHM_FACTOR * abs(sigma)
                    fwhm_kms = C * fwhm / mean
                    sigma_kms = C * abs(sigma) / mean

                    integral = amp * abs(sigma) * np.sqrt(2*np.pi)

                    # Convert to Jy km/s
                    area_jykms = integral * (C / mean) / 1000

                    # Flux density estimate by dividing by line width (FWHM in km/s)
                    flux_density_jy = area_jykms / 3*fwhm_kms if fwhm_kms != 0 else np.nan

                    ###########

                        # Luminosity and Prime Luminosity calculation
                    luminosity_distance = cosmo.luminosity_distance(Z).to('Mpc').value

                    L_prime_line =  3.25e7 * area_jykms * (luminosity_distance**2) / (((1+Z)**3)*(nu_obs**2))
                    L_line = 3e-11 * nu_rest**3 * L_prime_line

                    L_line_intrinsic = L_line / 25
                    L_prime_line_intrinsic = L_prime_line / 25

                    print('################################')

                    print(f"Luminosity: {L_line:.3e} L_sun")
                    print(f"Prime Luminosity: {L_prime_line:.3e} L_sun")

                    print('################################')

                    print(f"Intrinsic Luminosity: {L_line_intrinsic:.3e} L_sun")
                    print(f"Intrinsic Prime Luminosity: {L_prime_line_intrinsic:.3e} L_sun")

                    print('################################')

                    ###########

                    print(
                        f"{line_name} gaussian fit:\n"
                        f" amp={amp:.5f} uJy\n"
                        f" mean={mean:.5f} GHz\n"
                        f" sigma={sigma:.5f} GHz ({sigma_kms:.5f} km/s)\n"
                        f" FWHM={fwhm:.4f} GHz ({fwhm_kms:.5f} km/s)\n"
                        f" Integrated flux={area_jykms:.5f} Jy km/s\n"
                    )

                    # Plot fit
                    x_line = np.linspace(x_fit.min(),
                                        x_fit.max(), 300)

                    y_line = gaussian(x_line, *popt)

                    ax.plot(x_line, y_line,
                            color=color, lw=1,
                            linestyle='-', alpha=0.8,
                            zorder=4)

                except Exception as e:
                    print(f"Gaussian fit failed for {line_name}: {e}")

    ###################################################
    # Plot spectrum

    plt.step(Freq, Flux,
             where='mid',
             color='#FFAD0AFF',
             lw=1,
             zorder=3)
    
    ax.fill_between(
    Freq,
    Flux,
    0,                      # fill down to baseline
    step='mid',             # match plt.step
    color='#FFAD0AFF',
    alpha=0.3,              # transparency
    zorder=1                # behind the line
    )

    ax.axhline(0, color='0',
               lw=1,
               linestyle='dashed',
               zorder=3)

    ###################################################
    # Rest-frame frequency axis

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()[0]*(1+Z),
                 ax.get_xlim()[1]*(1+Z))

    ax2.set_xlabel('Frequency (rest-frame) [GHz]',
                   fontsize=10)

    ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
    ###################################################
    # Save figure

    plt.gcf().set_size_inches(7, 3)

    plt.savefig(f'{Title}.png',
                dpi=100,
                bbox_inches='tight')

    plt.show()


##############################################################################
###############################################################################

def MakeStackedProfiles(File, Title, Z, lines_dict,
                        fit_window=0.08, bin_size=10,
                        y_limits=None,
                        skip_fit_lines=None,
                        n_gaussians=2,
                        edge=[0, 0, 0],
                        colors=None,
                        target_name=None):

    if skip_fit_lines is None:
        skip_fit_lines = []

    C = 299792.458  # km/s
    FWHM_FACTOR = 2.354820045

    # -----------------------------
    # Models
    # -----------------------------
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

    def multi_gaussian(x, *params):
        n = len(params) // 3
        result = np.zeros_like(x)
        for i in range(n):
            result += gaussian(
                x,
                params[3*i],
                params[3*i + 1],
                params[3*i + 2]
            )
        return result

    # -----------------------------
    # Load data
    # -----------------------------
    data = np.genfromtxt(File, usecols=(2, 4), skip_header=1)
    freq_raw = data[:, 0]
    flux_raw = data[:, 1] * 1e6  # Jy → uJy

    # -----------------------------
    # Binning
    # -----------------------------
    if bin_size > 1:
        n = len(freq_raw) // bin_size
        freq_raw = freq_raw[:n*bin_size].reshape(n, bin_size).mean(axis=1)
        flux_raw = flux_raw[:n*bin_size].reshape(n, bin_size).mean(axis=1)

    Freq = freq_raw
    Flux = flux_raw

    # -----------------------------
    # Collect profiles
    # -----------------------------
    profiles = []

    for line_name, nu_rest in lines_dict.items():

        nu_obs = nu_rest / (1 + Z)

        mask = (Freq >= nu_obs - fit_window) & \
               (Freq <= nu_obs + fit_window)

        if np.sum(mask) == 0:
            continue

        x = Freq[mask]
        y = Flux[mask]

        # velocity convention (your choice)
        vel = -C * (x - nu_obs) / nu_obs

        profiles.append((line_name, vel, y))

    if len(profiles) == 0:
        print("No valid lines found.")
        return

    # -----------------------------
    # Plot setup
    # -----------------------------
    n_lines = len(profiles)

    fig, axes = plt.subplots(
        n_lines, 1,
        sharex=True,
        figsize=(6, 2*n_lines),
        gridspec_kw={'hspace': 0}
    )

    if n_lines == 1:
        axes = [axes]

    if colors==None:
        colors = ['#4285f4', '#D72000FF', '#2CA030FF']
    else:
        colors = colors
    

    # -----------------------------
    # Loop over lines
    # -----------------------------
    for i, (line_name, vel, y) in enumerate(profiles):

        ax = axes[i]
        color = colors[i % len(colors)]

        # Data
        ax.step(vel, y, where='mid', color=color, lw=1)

        ax.axhline(0, color='0.5', linestyle='dashed', linewidth=1)
        ax.axvline(0, color='0.5', linestyle='dashed', linewidth=1)

        #
        ax.axvline(edge[i], color=color, linestyle='dashed', linewidth=1)
        ax.axvline(-edge[i], color=color, linestyle='dashed', linewidth=1)

        ax.fill_between(
            vel, 0, y,
            color=color,
            step='mid',
            alpha=0.25,
            linewidth=0
        )

        # -----------------------------
        # Fit
        # -----------------------------
        if line_name not in skip_fit_lines:

            try:
                peak = y.max()

                # initial guesses
                p0 = []
                mu0 = vel[np.argmax(y)]

                for k in range(n_gaussians):
                    A = peak / (k + 1)
                    mu = mu0 + k * 80
                    sigma = 50
                    p0.extend([A, mu, sigma])

                popt, _ = curve_fit(
                    multi_gaussian,
                    vel,
                    y,
                    p0=p0,
                    maxfev=10000
                )

                v_fit = np.linspace(vel.min(), vel.max(), 400)

                # total fit
                g_total = multi_gaussian(v_fit, *popt)
                ax.plot(v_fit, g_total, color=color, lw=1.5)

                # components
                for k in range(n_gaussians):
                    A = popt[3*k]
                    mu = popt[3*k + 1]
                    sigma = popt[3*k + 2]

                    g = gaussian(v_fit, A, mu, sigma)
                    ax.plot(v_fit, g, color=color, ls='--', lw=1)

                # -----------------------------
                # Fit annotation (bottom-left)
                # -----------------------------
                fit_text = ""

                for k in range(n_gaussians):
                    A = popt[3*k]
                    sigma = popt[3*k + 2]
                    fwhm = FWHM_FACTOR * sigma

                    fit_text += f"A={A:.2f} $\\mu$Jy, FWHM={fwhm:.1f} km/s\n"

                ax.text(
                    0.02, 0.02,
                    fit_text.strip(),
                    transform=ax.transAxes,
                    fontsize=9,
                    ha='left',
                    va='bottom',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
                )

            except RuntimeError:
                pass

        # -----------------------------
        # Labels
        # -----------------------------
        ax.text(
            0.98, 0.98,
            line_name,
            transform=ax.transAxes,
            color=color,
            fontsize=15,
            ha='right',
            va='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
        )

        ax.text(
            0.02, 0.98,
            target_name,
            transform=ax.transAxes,
            color='black',
            fontsize=12,
            ha='left',
            va='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
        )

        if y_limits is not None:
            ax.set_ylim(y_limits)

        ax.set_ylabel('Flux [$\\mu$Jy]')

        if i < n_lines - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel('Velocity [km/s]')

    plt.tight_layout()
    plt.savefig(f"{Title}_stacked.png", dpi=120, bbox_inches='tight')
    plt.show()


###########################################################################################
###########################################################################################
def PlotThreeMaps(files,
                  titles=None,
                  continuum_contours=None,
                  show_aperture=False,
                  radius=5,
                  center_x=0,
                  center_y=0,
                  save_name=None,
                  show_offset=True,
                  size=20
                  ):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    for i, file in enumerate(files):
        ax = axes[i]

        # --- Read FITS ---
        Data, Header = ReadFITS(file)

        Bmaj = Header['BMAJ'] * 3600.0
        Bmin = Header['BMIN'] * 3600.0
        Bpa  = Header['BPA'] + 90

        PxScale = abs(Header['CDELT1'] * 3600.0)
        DimX = Header['NAXIS1']
        DimY = Header['NAXIS2']

        extent = np.array([
            -DimX * PxScale / 2., DimX * PxScale / 2.,
            -DimY * PxScale / 2., DimY * PxScale / 2.
        ])

        # --- Coordinates ---
        x_coords = np.linspace(extent[0], extent[1], DimX)
        y_coords = np.linspace(extent[2], extent[3], DimY)
        X, Y = np.meshgrid(x_coords, y_coords)

        # --- RMS ---
        rms_mask = (X**2 + Y**2) >= 20**2
        RMS = np.nanstd(Data[rms_mask])

        # --- Image ---
        im = ax.imshow(Data * 1e3,
                       cmap='viridis',
                       origin='lower',
                       extent=extent)

        # --- S/N contours ---
        ax.contour(Data / RMS,
                   levels=[-6, -5, -4,-3,-2, 2,3,4,5,6,7,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                   colors='black',
                   linewidths=1,
                   extent=extent)

        # --- Continuum contours ---
        if continuum_contours is not None:
            ax.contour(continuum_contours[i],
                       levels=[2,3,4,5,6,7,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                       colors='white',
                       linewidths=1,
                       extent=extent)

        # --- Aperture ---
        if show_aperture:
            circle = plt.Circle((center_x, center_y),
                                radius,
                                color='white',
                                fill=False,
                                linestyle='--')
            ax.add_patch(circle)

        # --- Beam ---
        
        beam = Ellipse((-size+Bmaj, -size+Bmaj),
                       width=Bmaj,
                       height=Bmin,
                       angle=Bpa,
                       hatch='/////',
                       fc='none',
                       ec='white',
                       lw=1)
        ax.add_patch(beam)

        # --- Title ---
        ax.text(0.05, 0.95,
                titles[i],
                transform=ax.transAxes,
                color='white',
                fontsize=14,
                ha='left',
                va='top')
        
    
        ax.set_xlim(-size, size)
        ax.set_ylim(-size, size)

        if show_offset:
            ax.set_xlabel('RA Offset (arcsec)', fontsize=14)
            ax.set_ylabel('Dec Offset (arcsec)', fontsize=14)

            ax.tick_params(
                axis='both',
                direction='in',
                length=6,
                width=1,
                colors='black',
                labelsize=12
            )

            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

        else:
            ax.set_xlabel('')
            ax.set_ylabel('')

            # Remove tick marks and tick labels
            ax.set_xticks([])
            ax.set_yticks([])

            

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('mJy/beam')

    plt.tight_layout()

    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')

    plt.show()


##############################################################################
###############################################################################

def MakeStackedProfilesSameLine(Files, Title, Z, line_name, nu_rest,
                        labels=None,
                        fit_window=0.08, bin_size=10,
                        y_limits=None,
                        skip_fit=False,
                        n_gaussians=2,
                        edge=[0, 0],
                        colors=None):

    C = 299792.458  # km/s
    FWHM_FACTOR = 2.354820045

    if labels is None:
        labels = [f"File {i+1}" for i in range(len(Files))]

    # -----------------------------
    # Normalize y_limits: allow either
    #   - a single (ymin, ymax) tuple applied to all panels
    #   - a list of (ymin, ymax) tuples, one per panel
    #   - None (no limits set)
    # -----------------------------
    if y_limits is not None:
        if isinstance(y_limits[0], (list, tuple)):
            y_limits_list = list(y_limits)
        else:
            y_limits_list = [y_limits] * len(Files)
    else:
        y_limits_list = [None] * len(Files)

    # -----------------------------
    # Models
    # -----------------------------
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

    def multi_gaussian(x, *params):
        n = len(params) // 3
        result = np.zeros_like(x)
        for i in range(n):
            result += gaussian(
                x,
                params[3*i],
                params[3*i + 1],
                params[3*i + 2]
            )
        return result

    nu_obs = nu_rest / (1 + Z)

    # -----------------------------
    # Load + process each file
    # -----------------------------
    profiles = []

    for File in Files:

        data = np.genfromtxt(File, usecols=(2, 4), skip_header=1)
        freq_raw = data[:, 0]
        flux_raw = data[:, 1] * 1e3  # Jy → mJy

        if bin_size > 1:
            n = len(freq_raw) // bin_size
            freq_raw = freq_raw[:n*bin_size].reshape(n, bin_size).mean(axis=1)
            flux_raw = flux_raw[:n*bin_size].reshape(n, bin_size).mean(axis=1)

        Freq = freq_raw
        Flux = flux_raw

        mask = (Freq >= nu_obs - fit_window) & \
               (Freq <= nu_obs + fit_window)

        if np.sum(mask) == 0:
            print(f"No data found in window for {File}")
            continue

        x = Freq[mask]
        y = Flux[mask]

        # velocity convention (your choice) — same nu_obs for both files
        vel = -C * (x - nu_obs) / nu_obs

        profiles.append((File, vel, y))

    if len(profiles) == 0:
        print("No valid data found.")
        return

    # -----------------------------
    # Plot setup
    # -----------------------------
    n_panels = len(profiles)

    fig, axes = plt.subplots(
        n_panels, 1,
        sharex=True,
        figsize=(6, 2*n_panels),
        gridspec_kw={'hspace': 0}
    )

    if n_panels == 1:
        axes = [axes]

    if colors is None:
        colors = ['#4285f4', '#D72000FF', '#2CA030FF']

    # -----------------------------
    # Loop over files (top -> bottom)
    # -----------------------------
    for i, (File, vel, y) in enumerate(profiles):

        ax = axes[i]
        color = colors[i % len(colors)]

        ax.step(vel, y, where='mid', color=color, lw=1)

        # ax.axvline(edge[i], color=color, linestyle='dashed', linewidth=1)
        # ax.axvline(-edge[i], color=color, linestyle='dashed', linewidth=1)

        ax.fill_between(
            vel, 0, y,
            color=color,
            step='mid',
            alpha=0.25,
            linewidth=0
        )

        # -----------------------------
        # Fit
        # -----------------------------
        if not skip_fit:

            try:
                peak = y.max()

                p0 = []
                mu0 = vel[np.argmax(y)]

                for k in range(n_gaussians):
                    A = peak / (k + 1)
                    mu = mu0 + k * 80
                    sigma = 50
                    p0.extend([A, mu, sigma])

                popt, pcov = curve_fit(
                    multi_gaussian,
                    vel,
                    y,
                    p0=p0,
                    maxfev=10000
                )

                v_fit = np.linspace(vel.min(), vel.max(), 400)

                g_total = multi_gaussian(v_fit, *popt)
                ax.plot(v_fit, g_total, color=color, lw=1.5)

                for k in range(n_gaussians):
                    A = popt[3*k]
                    mu = popt[3*k + 1]
                    sigma = popt[3*k + 2]

                    g = gaussian(v_fit, A, mu, sigma)
                    ax.plot(v_fit, g, color=color, ls='--', lw=1)
           

                fit_text = ""
                for k in range(n_gaussians):
                    A = popt[3*k]
                    mu = popt[3*k + 1]
                    mu_err = np.sqrt(pcov[3*k + 1, 3*k + 1])
                    sigma = popt[3*k + 2]

                    fwhm = FWHM_FACTOR * sigma
                    nu_fit = nu_obs * (1 - mu / C)
                    z_fit = nu_rest / nu_fit - 1
                    z_err = (nu_rest / nu_fit**2) * (nu_obs / C) * mu_err

                    fit_text += (
                        f"A = {A:.2f} mJy  "
                        f"FWHM = {fwhm:.1f} km/s  "
                        f"z = {z_fit:.6f} ± {z_err:.6f}"
                    )
                    ax.axvline(mu, color=color, linestyle='dashed', linewidth=1)

                ax.text(
                    0.02, 0.02,
                    fit_text.strip(),
                    transform=ax.transAxes,
                    fontsize=9,
                    ha='left',
                    va='bottom',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
                )
                

            except RuntimeError:
                pass

        # -----------------------------
        # Labels
        # -----------------------------
        ax.text(
            0.98, 0.98,
            labels[i],
            transform=ax.transAxes,
            color=color,
            fontsize=15,
            ha='right',
            va='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
        )

        if y_limits_list[i] is not None:
            ax.set_ylim(y_limits_list[i])

        ax.set_ylabel('Flux [mJy]')

        if i < n_panels - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel('Velocity [km/s]')

    fig.suptitle(line_name, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{Title}_stacked.png", dpi=120, bbox_inches='tight')
    plt.show()