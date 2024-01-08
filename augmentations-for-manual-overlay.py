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
import glob

ybc = readYBC() # read in from catalogue
ybc_sub = ybc[ybc['Vmag'] < 5] # filter out stars that are dimmer than this visual magnitude

if __name__ == "__main__":

	# Command line arguments
	parser = argparse.ArgumentParser(description='Fit a rough WCS to an all-sky FITS frame.')
	parser.add_argument('path', help='path to search *.FITS')
	parser.add_argument('dark', help='master dark frame')
	parser.add_argument('-p', '--pattern', help='Unix style pathname pattern expansion', default="*.FITS")
	parser.add_argument('-g', '--longitude', type=float, help='longitude of observation station, default 1.2062 degrees', default=1.2062)
	parser.add_argument('-t', '--latitude', type=float, help='latitude of observation station, default 52.7721 degrees', default=52.7721)
	args = parser.parse_args()
	path_to = args.path
	dark_data = args.dark
	location = EarthLocation(lat=args.latitude*u.degree, lon=args.longitude*u.degree, height=72.0*u.meter)

	fits_files = glob.glob(path_to + '/' + args.pattern)
	wcs = load_fisheye(path_to + '/' + 'wcs-refined.npz') # Load alt-az coordinate system (after running calibration)

	for fits_file in fits_files: # loop through all fits files in directory

		# Load light frame
		image_file = get_pkg_data_filename(fits_file)
		image_data = fits.getdata(image_file)
		image_data = cv2.flip(image_data, 0) # flip vertical - FITS are stored 'upside down'
		image_data = cv2.cvtColor(image_data, cv2.COLOR_BAYER_RG2RGB) # debayer into RGB - Decode XY2RGB bayer pattern using figure at
		height, width = image_data.shape[0:2] # size of image

		# Time stamp of image
		hdr = fits.getheader(image_file,0 )
		date = hdr['DATE-OBS']
		t = Time(date, format='isot', scale='utc')
		mjd = t.mjd # Modified Julian Date of the frame with which to calibrate lens distortion

		# Dark frame subtraction
		dark_data = plt.imread(args.dark) # need to 'pip install image' for PIL
		dark_data = cv2.cvtColor(dark_data, cv2.COLOR_BAYER_RG2RGB) # debayer into RGB - See comment above
		calibrated = cv2.subtract(image_data, dark_data) # subtract master dark frame
		calibrated = cv2.flip(calibrated, 1) # flip horizontal
		calibrated = exposure.rescale_intensity(cv2.bitwise_not(calibrated), in_range=(220,255)) # invert and set the black point to N (i.e. stretch the original range of N to 255 so that it clips pixels which are <N and then stretches what's left into the range 0 to 255)
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

		# Loop through keypoints saving in a numpy array
		star_coords = np.empty((0,3), int)
		for idx, item in enumerate(keypoints):
			star_coords = np.append(star_coords, np.array([[item.pt[0], item.pt[1], item.size]]), axis=0) # extract star coordinates and size from keypoints into numpy array
		phot_appertures = CircularAperture( zip(star_coords[:,0], star_coords[:,1]), r=5.)

		# Calculate expected positions of catalogue stars
		alt_cat, az_cat = radec2altaz(ybc_sub['RA'], ybc_sub['Dec'], mjd, location=location)
		above = np.where(alt_cat > 5.)
		mags = np.array(ybc_sub['Vmag'])[above]
		x_expected, y_expected = wcs.all_world2pix(az_cat[above], alt_cat[above], 0.)
		good = np.where(~(np.isnan(x_expected)) & ~(np.isnan(y_expected)))
		x_expected = x_expected[good]
		y_expected = y_expected[good]
		mags = mags[good]

		# Plot refined fit of alt-az coordinate system
		dpi = 50 # nothing really to do with dpi or inches or actual figure size - matplotlib is just strange in this regard
		fig = plt.figure(figsize=(width/dpi,height/dpi), dpi=dpi)
		ax = plt.Axes(fig, [0., 0., 1., 1.])
		ax.set_axis_off()
		fig.add_axes(ax, projection=wcs)
		plt.imshow(np.ones((height,width,3)), origin='upper')

		# # Indicate stars:
		# # green if catalogue star matches detected star to within 'tol' pixels
		# # red if catalogue star doesn't match up with detected star to within 'tol' pixels
		# tol = 2.0; zp = -2.
		# for i, expected in enumerate(np.array([x_expected, y_expected]).T): # loop through expected stars
			# found = False
			# radius = 10.**(-(mags[i] + zp)/2.5) + 2 # add 2 px to expected star radius so encircles the star
			# for detected in star_coords: # loop through detected stars
				# if np.sqrt( (detected[0] - expected[0])**2 + (detected[1] - expected[1])**2 ) < tol: # if expected star corresponds closely enough to a detected star...
					# found = True
					# CircularAperture( (expected[0], expected[1]), r=radius).plot(color='green', lw=1, alpha=0.75) # plot it in green...
					# break
			# if not found:
				# CircularAperture( (expected[0], expected[1]), r=radius).plot(color='red', lw=1, alpha=0.75) # otherwise plot it in red...

		# Add grid lines
		resolution = 100
		# Add Alt grid lines
		for alt in range(15, 166, 15):
			az = np.linspace(0, 361, resolution)
			alt = np.repeat(90 - alt, resolution)
			x, y = wcs.all_world2pix(az, alt, 0.)
			alt_vertices = Polygon(np.vstack([x, y]).T, False, edgecolor='blue', facecolor='none', lw = 2, ls='dotted', alpha=0.75)
			ax.add_patch(alt_vertices)

		# Add Az grid lines
		# from LongitudeCircle import LongitudeCircle
		for az in range(0, 361, 60):
			az = np.repeat(az, resolution)
			alt = np.linspace(90, -90, resolution)
			x, y = wcs.all_world2pix(az, alt, 0.)
			az_vertices = Polygon(np.vstack([x, y]).T, False, edgecolor='blue', facecolor='none', lw = 2, ls='dotted', alpha=0.75)
			ax.add_patch(az_vertices)

		# Add Dec grid lines
		for dec in range(15, 166, 15):
			ra = np.linspace(0, 361, resolution)
			dec = np.repeat(90 - dec, resolution)
			alt, az = radec2altaz(ra, dec, mjd, location=location)
			x, y = wcs.all_world2pix(az, alt, 0.)
			dec_vertices = Polygon(np.vstack([x, y]).T, False, edgecolor='black', facecolor='none', lw = 2, ls='dotted', alpha=0.75)
			ax.add_patch(dec_vertices)

		# Add RA grid lines
		for ra in range(0, 361, 60):
			ra = np.repeat(ra, resolution)
			dec = np.linspace(90., -90., resolution)
			alt, az = radec2altaz(ra, dec, mjd, location=location)
			x, y = wcs.all_world2pix(az, alt, 0.)
			ra_vertices = Polygon(np.vstack([x, y]).T, False, edgecolor='black', facecolor='none', lw = 2, ls='dotted', alpha=0.75)
			ax.add_patch(ra_vertices)

		# Draw constellations
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
				x1, y1 = wcs.all_world2pix(az1, alt1, 0.) # Calculate x y coordinates of first star on chip
				ybc_star2 = ybc[ybc['HR'] == int(con[2][2*i+1])] # Look up second star in YBC
				alt2, az2 = radec2altaz(float(ybc_star2['RA']), float(ybc_star2['Dec']), mjd, location=location)
				x2, y2 = wcs.all_world2pix(az2, alt2, 0.) # Calculate x y coordinates of second star on chip
				if not np.isnan([x1, y1, x2, y2]).any():
					vertices = Polygon(np.array([[x1, y1], [x2, y2]]), False, edgecolor='black', facecolor='none', lw = 2, ls='solid', alpha=0.75)
					ax.add_patch(vertices)

		# Az and RA Grid labels
		for deg in range(0, 301, 60):
			x, y = wcs.all_world2pix(deg, 67.5, 0.) # Az label coordinates
			if not np.isnan([x, y]).any():
				ax.annotate('Az ' + str(deg), (x, y), color='blue', fontsize=16, alpha=0.75)
			alt, az = radec2altaz(deg, 67.5, mjd, location=location)
			x, y = wcs.all_world2pix(az, alt, 0.) # RA label coordinates
			if not np.isnan([x, y]).any():
				ax.annotate('RA ' + str(deg), (x, y), color='black', fontsize=16, alpha=0.75)

		# Alt and Dec Grid labels
		for deg in range(15, 166, 15):
			x, y = wcs.all_world2pix(30, 90 - deg, 0.) # Az label coordinates
			if not np.isnan([x, y]).any():
				ax.annotate('Alt ' + str(90 - deg), (x, y), color='blue', fontsize=16, alpha=0.75)
			x, y = wcs.all_world2pix(210, 90 - deg, 0.) # Az label coordinates - 180 degrees apart (to avoid being off-screen)
			if not np.isnan([x, y]).any():
				ax.annotate('Alt ' + str(90 - deg), (x, y), color='blue', fontsize=16, alpha=0.75)

			alt, az = radec2altaz(30, 90 - deg, mjd, location=location)
			x, y = wcs.all_world2pix(az, alt, 0.) # RA label coordinates
			if not np.isnan([x, y]).any():
				ax.annotate('Dec ' + str(90 - deg), (x, y), color='black', fontsize=16, alpha=0.75)
			alt, az = radec2altaz(210, 90 - deg, mjd, location=location)
			x, y = wcs.all_world2pix(az, alt, 0.) # RA label coordinates - 180 degrees apart (to avoid being off-screen)
			if not np.isnan([x, y]).any():
				ax.annotate('Dec ' + str(90 - deg), (x, y), color='black', fontsize=16, alpha=0.75)

		# Timestamp
		ax.annotate(str(t.iso), (5, 5), xycoords='axes pixels', color='black', weight='bold', fontsize=30)

		plt.savefig(os.path.splitext(fits_file)[0] +'.png', format='png')
		plt.close(fig)