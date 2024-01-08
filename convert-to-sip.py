import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from matplotlib.patches import Polygon

import numpy as np
from scipy.optimize import minimize
import os
import sys
import pickle
import re
import argparse # pip install argparse
from pandas import read_csv

from read_ybc import readYBC
from fit_wcs import wcs_air
from fisheye_map import Fisheye, load_fisheye
from utils import radec2altaz, altaz2radec

from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, Longitude, Latitude
import astropy.units as u
from astropy.time import Time
from astropy import units as u
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.modeling import models, fitting
from astropy.utils.data import get_pkg_data_filename

import cv2 # pip install opencv-python
from skimage import exposure # pip install skimage

from photutils import CircularAperture


if __name__ == "__main__":

	# Command line arguments
	parser = argparse.ArgumentParser(description='Fit WCS with distortion table to WCS with SIP.')
	parser.add_argument('path', help='path to wcs-initial.pkl and wcs-refined.npz')
	parser.add_argument('-f', '--frame', help='FITS frame for preview')
	parser.add_argument('-d', '--dark', help='master dark frame for preview')
	parser.add_argument('-g', '--longitude', type=float, help='longitude of observation station, default 1.2062 degrees', default=1.2062)
	parser.add_argument('-t', '--latitude', type=float, help='latitude of observation station, default 52.7721 degrees', default=52.7721)
	parser.add_argument('-v', '--preview', help='preview existing wcs-sip.fits only', action='store_true')
	args = parser.parse_args()
	light_frame_file = args.frame
	dark_data = args.dark
	preview_only = args.preview
	path_to_wcs = args.path
	path_to_frame, filename = os.path.split(os.path.splitext(light_frame_file)[0]) # Get file name without the extension
	location = EarthLocation(lat=args.latitude*u.degree, lon=args.longitude*u.degree, height=72.0*u.meter)

	wcs_sip = None
	if not preview_only:
		# Load initial WCS
		try:
			wcs_initial = pickle.load(open(path_to_wcs + '/' + "wcs-initial.pkl", "rb"))
			print(path_to_wcs + '/wcs-initial.pkl loaded')
		except (OSError, IOError) as e:
			print(path_to_wcs + '/wcs-initial.pkl does not exist. Run fit-initial-wcs.py first.')
			exit()

		# Load refined WCS
		wcs_refined = load_fisheye(path_to_wcs + '/' + 'wcs-refined.npz') # Load alt-az coordinate system (after running calibration)
		if not wcs_refined is None:
			print(path_to_wcs + '/wcs-refined.npz loaded: The refined WCS is in use.')
		else:
			print(path_to_wcs + '/wcs-refined.npz does not exist. Run refine-wcs.py first.')
			exit()

		# def printx(x):
			# print(x)
		
		# Convert distortion table to SIP
		print('Converting WCS with distortion table to WCS with SIP...')
		width = np.int(np.max(wcs_refined.x)); height = np.int(np.max(wcs_refined.y))
		x = np.linspace(0, width, 20)
		y =  np.linspace(0, height, 20)
		X, Y = np.meshgrid(x,y)
		xx = X.ravel()
		yy = Y.ravel()
		xy_to_alt, xy_to_az = wcs_refined.all_pix2world(xx, yy, 0.)
		good = np.where(~(np.isnan(xy_to_az)) & ~(np.isnan(xy_to_alt)))

		# Doesn't work reliably - use own optimization instead
		# wcs_sip1.to_fits().writeto(path_to_wcs + '/' + 'wcs_sip.fits', overwrite=True)
		# from astropy.wcs.utils import fit_wcs_from_points
		# pixels = [xx[good], yy[good]]
		# world = SkyCoord(xy_to_az[good], xy_to_alt[good], unit='deg', frame='fk5')
		# wcs_sip = fit_wcs_from_points(pixels, world, proj_point='center', projection='AIR', sip_degree=2)

		sip_order1 = 2
		fun1 = wcs_air(xx[good], yy[good], xy_to_alt[good], xy_to_az[good], crpix1=width/2, crpix2=height/2, a_order=sip_order1, b_order=sip_order1)
		x0 = np.concatenate(np.array([ fun1.wcs2x0(wcs_initial) , np.zeros((sip_order1+1)**2), np.zeros((sip_order1+1)**2) ]))
		fit_result1 = minimize(fun1, x0, method='Powell', options={'disp':True, 'maxiter':10000})#, callback=printx)
		wcs_sip = fun1.return_wcs(fit_result1.x)
		wcs_sip.to_fits(relax=True).writeto(path_to_wcs + '/wcs_sip.fits', overwrite=True)
	
	else:
		print('Preview only using existing wcs_sip.fits. If this fails, run without --preview (-v) first.')
		wcs_sip = wcs.WCS(fits.open(path_to_wcs + '/wcs_sip.fits')[0].header)

	if args.frame is not None and args.dark is not None:

		# Load light frame
		image_file = get_pkg_data_filename(light_frame_file)
		image_data = fits.getdata(image_file)
		image_data = cv2.flip(image_data, 0) # flip vertical - FITS are stored 'upside down'
		image_data = cv2.cvtColor(image_data, cv2.COLOR_BAYER_RG2RGB) # debayer into RGB - Decode XY2RGB bayer pattern using figure at https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
		height, width = image_data.shape[0:2] # size of image

		# Time stamp of image
		hdr = fits.getheader(image_file,0 )
		date = hdr['DATE-OBS']
		t = Time(date, format='isot', scale='utc')
		mjd = t.mjd # Modified Julian Date of the frame with which to calibrate lens distortion

		# Dark frame subtraction
		dark_data = plt.imread(dark_data) # need to 'pip install image' for PIL
		dark_data = cv2.cvtColor(dark_data, cv2.COLOR_BAYER_RG2RGB) # debayer into RGB - See comment above
		calibrated = cv2.subtract(image_data, dark_data) # subtract master dark frame
		calibrated = cv2.flip(calibrated, 1) # flip horizontal
		calibrated = exposure.rescale_intensity(cv2.bitwise_not(calibrated), in_range=(180,255)) # invert and set the black point to N (i.e. stretch the original range of N to 255 so that it clips pixels which are <N and then stretches what's left into the range 0 to 255)
		calibrated_blurred = cv2.GaussianBlur(calibrated, (9, 9), 0) # change blur level to denoise - a main parameter for adjustment
		calibrated_blurred_gray = cv2.cvtColor(calibrated_blurred, cv2.COLOR_RGB2GRAY) # convert to grayscale

		# Star detection
		params = cv2.SimpleBlobDetector_Params() # Set up SimpleBlobDetector parameters
		params.filterByColor = 0 # Filter by color
		params.blobColor = 255
		params.minThreshold = 100 # Change thresholds - main parameter for adjustment
		params.maxThreshold = 255
		params.thresholdStep = 10
		params.filterByArea = True # Filter by area
		params.minArea = 4
		params.maxArea = 100
		detector = cv2.SimpleBlobDetector_create(params) # Set up the detector with default parameters.
		keypoints = detector.detect(calibrated_blurred_gray) # Detect blobs

		# Loop through keypoints saving in a numpy array and correcting for coordinates on original image
		star_coords = np.empty((0,3), int)
		for idx, item in enumerate(keypoints):
			star_coords = np.append(star_coords, np.array([[item.pt[0], item.pt[1], item.size]]), axis=0) # extract star coordinates and size from keypoints into numpy array

		phot_appertures = CircularAperture( zip(star_coords[:,0], star_coords[:,1]), r=5.)
		zp = -2. # Tune this value to convert star radii on chip into magnitudes
		measured_mags = -2.5*np.log10(star_coords[:,2]) - zp

		# Plot catalogue stars in expected location (in green) vs detected stars (in blue)
		ybc = readYBC() # Read in Yale Bright Star Catalogue
		ybc_sub = ybc[ybc['Vmag'] < 5] # filter out stars that are dimmer than this visual magnitude
		
		# Calculate expected locations of these catalogue stars
		alt_cat, az_cat = radec2altaz(ybc_sub['RA'], ybc_sub['Dec'], mjd, location=location)
		above = np.where(alt_cat > 5.)
		mags = np.array(ybc_sub['Vmag'])[above]
		x_expected, y_expected = wcs_sip.all_world2pix(az_cat[above], alt_cat[above], 0.)
		good = np.where(~(np.isnan(x_expected)) & ~(np.isnan(y_expected)))
		x_expected = x_expected[good]
		y_expected = y_expected[good]
		mags = mags[good]

		# Let's take a look at where the stars are, and where predicted stars are:
		dpi = 50 # nothing really to do with dpi or inches or actual figure size - matplotlib is just strange in this regard
		fig = plt.figure(figsize=(width/dpi,height/dpi), dpi=dpi)
		ax = plt.Axes(fig, [0., 0., 1., 1.])
		ax.set_axis_off()
		fig.add_axes(ax)
		plt.imshow(calibrated, origin='upper')

		# Indicate stars:
		# green if catalogue star matches detected star to within 'tol' pixels
		# red if catalogue star doesn't match up with detected star to within 'tol' pixels
		tol = 2.0; zp = -2.
		for i, expected in enumerate(np.array([x_expected, y_expected]).T): # loop through expected stars
			found = False
			radius = 10.**(-(mags[i] + zp)/2.5) + 2 # add 2 px to expected star radius so encircles the star
			for detected in star_coords: # loop through detected stars
				if np.sqrt( (detected[0] - expected[0])**2 + (detected[1] - expected[1])**2 ) < tol: # if expected star corresponds closely enough to a detected star...
					found = True
					CircularAperture( (expected[0], expected[1]), r=radius).plot(color='green', lw=1, alpha=0.75) # plot it in green...
					break
			if not found:
				CircularAperture( (expected[0], expected[1]), r=radius).plot(color='red', lw=1, alpha=0.75) # otherwise plot it in red...

		# Draw constellations (in expected location)
		constellations = []
		with open("ConstellationLines.dat", "r") as f: # Read constellations into an array
			lines = f.readlines()
			for l in lines:
				l = l.split()
				constellations.append([l[0], l[1], l[2:]])
		f.close()
		for i, con in enumerate(constellations): # Loop through constellations: Name, Number of star pairs, List of stars pairs
			for i in range(0, int(con[1]), 1): # Loop through star pairs in pairs
				ybc_star1 = ybc[ybc['HR'] == int(con[2][2*i])] # Look up first star in YBC
				alt1, az1 = radec2altaz(float(ybc_star1['RA']), float(ybc_star1['Dec']), mjd, location=location)
				if alt1 > 5.:
					x1, y1 = wcs_sip.all_world2pix(az1, alt1, 0.) # Calculate x y coordinates of first star on chip
					ybc_star2 = ybc[ybc['HR'] == int(con[2][2*i+1])] # Look up second star in YBC
					alt2, az2 = radec2altaz(float(ybc_star2['RA']), float(ybc_star2['Dec']), mjd, location=location)
					if alt2 > 5.:
						x2, y2 = wcs_sip.all_world2pix(az2, alt2, 0.) # Calculate x y coordinates of second star on chip
						if not np.isnan([x1, y1, x2, y2]).any():
							vertices = Polygon(np.array([[x1, y1], [x2, y2]]), False, edgecolor='black', facecolor='none', lw = 1, ls='solid', alpha=0.75)
							ax.add_patch(vertices)

		plt.draw()
		plt.show()