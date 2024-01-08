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


# Global variables
ybc = readYBC() # Read in Yale Bright Star Catalogue
corrections = np.array([],dtype=[('x1', '<f8'), ('y1', '<f8'), ('x2', '<f8'), ('y2', '<f8')])
keypoints = None
wcs_initial = None
wcs_refined = None


# This class allows us to distinguish between clicks and click-drags, etc.
class Click():
	def __init__(self, ax, func, button=1):
		self.ax=ax
		self.func=func
		self.button=button
		self.press=False
		self.move = False
		self.c1=self.ax.figure.canvas.mpl_connect('button_press_event', self.onpress)
		self.c2=self.ax.figure.canvas.mpl_connect('button_release_event', self.onrelease)
		self.c3=self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onmove)

	def onclick(self,event):
		if event.inaxes == self.ax:
			if event.button == self.button:
				self.func(event, self.ax)
	def onpress(self,event):
		self.press=True
	def onmove(self,event):
		if self.press:
			self.move=True
	def onrelease(self,event):
		if self.press and not self.move:
			self.onclick(event)
		self.press=False; self.move=False
		

firstclick = True
def onclick(event, ax):
	if event.xdata is None or event.ydata is None:
		return
	
	global firstclick, corrections
	if firstclick:
		radius = 5
		d = -1; point = 0; radius = 0; xc = 0; yc = 0
		for idx, x in enumerate( x_expected ): # Check catalogue stars
			y = y_expected[idx]
			di = np.sqrt((event.xdata - x)**2 + (event.ydata - y)**2)
			if di < d or d < 0:
				d = di
				point = idx
				radius = 10.**(-(mags[point] + zp)/2.5) + 2
				xc = x; yc = y;
		xshift = 0; yshift = 0
		if show_refined:
			xshift = wcs_refined.reverse_xinterp(xc,yc) # the shift that has already been applied
			yshift = wcs_refined.reverse_yinterp(xc,yc)
		corrections = np.append(corrections, np.array((xc+xshift, yc+yshift, 0, 0), dtype=[('x1', '<f8'), ('y1', '<f8'), ('x2', '<f8'), ('y2', '<f8')]))
		print('(' + str(xc+xshift) + ', ' + str(yc+yshift) + ', ', end = '')
		circ = patches.Circle((xc,yc), radius, linewidth=1, edgecolor='r', facecolor='none')
		ax.add_patch(circ)
		plt.draw()
		firstclick = False
	else:
		d = -1; point = 0; radius = 0; xc = 0; yc = 0
		for idx, x in enumerate( star_coords[:,0] ): # Check detected stars
			y = star_coords[:,1][idx]
			di = np.sqrt((event.xdata - x)**2 + (event.ydata - y)**2)
			if di < d or d < 0:
				d = di
				point = idx
				radius = star_coords[:,2][idx]
				xc = x; yc = y;
		print(str(xc) + ', ' + str(yc) + '),')
		corrections[-1][2] = xc
		corrections[-1][3] = yc
		circ = patches.Circle((xc,yc), radius, linewidth=1, edgecolor='r', facecolor='none')
		ax.add_patch(circ)
		plt.draw()
		firstclick = True



if __name__ == "__main__":

	# Command line arguments
	parser = argparse.ArgumentParser(description='Fit a rough WCS to an all-sky FITS frame.')
	parser.add_argument('frame', help='FITS frame to fit a rough WCS to')
	parser.add_argument('dark', help='master dark frame')
	parser.add_argument('-g', '--longitude', type=float, help='longitude of observation station, default 1.2062 degrees', default=1.2062)
	parser.add_argument('-t', '--latitude', type=float, help='latitude of observation station, default 52.7721 degrees', default=52.7721)
	args = parser.parse_args()
	light_frame_file = args.frame
	dark_data = args.dark
	path_to, filename = os.path.split(os.path.splitext(light_frame_file)[0]) # Get file name without the extension
	location = EarthLocation(lat=args.latitude*u.degree, lon=args.longitude*u.degree, height=72.0*u.meter)

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
	params.minThreshold = 50 # Change thresholds - main parameter for adjustment
	params.maxThreshold = 255
	params.thresholdStep = 5
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

	# Load initial WCS
	try:
		wcs_initial = pickle.load(open(path_to + '/' + "wcs-initial.pkl", "rb"))
		print(path_to + '/wcs-initial.pkl loaded')
	except (OSError, IOError) as e:
		print(path_to + '/wcs-initial.pkl does not exist. Run fit-initial-wcs.py first.')
		exit()

	# Load refined WCS
	global show_refined
	show_refined = False
	wcs_refined = load_fisheye(path_to + '/' + 'wcs-refined.npz') # Load alt-az coordinate system (after running calibration)
	if not wcs_refined is None:
		show_refined = True
		print(path_to + '/wcs-refined.npz loaded: The refined WCS is in use.')
	else:
		print(path_to + '/wcs-refined.npz does not exist. A refined WCS is not yet in use.')

	# Select which WCS to use
	if show_refined:
		wcs = wcs_refined
	else:
		wcs = wcs_initial

	# Plot catalogue stars in expected location (in green) vs detected stars (in blue)
	ybc_sub = ybc[ybc['Vmag'] < 5] # filter out stars that are dimmer than this visual magnitude
	
	# Calculate expected locations of these catalogue stars
	alt_cat, az_cat = radec2altaz(ybc_sub['RA'], ybc_sub['Dec'], mjd, location=location)
	above = np.where(alt_cat > 5.)
	mags = np.array(ybc_sub['Vmag'])[above]
	x_expected, y_expected = wcs.all_world2pix(az_cat[above], alt_cat[above], 0.)
	good = np.where(~(np.isnan(x_expected)) & ~(np.isnan(y_expected)))
	x_expected = x_expected[good]
	y_expected = y_expected[good]
	mags = mags[good]

	# Let's take a look at where the stars are, and where predicted stars are:
	fig, ax = plt.subplots()
	plt.imshow(calibrated)
	for i, x in enumerate(star_coords[:,0]): # detected objects in blue
		y = star_coords[:,1][i]
		r = star_coords[:,2][i] + 1 # add 1 px to expected star radius so encircles the star
		circ = patches.Circle((x, y), r, linewidth=1.5, edgecolor='blue', alpha=0.5, facecolor='none')
		ax.add_patch(circ)
	for i, x in enumerate(x_expected): # predicted locations in green
		y = y_expected[i]
		r = 10.**(-(mags[i] + zp)/2.5) + 2 # add 2 px to expected star radius so encircles the star
		circ = patches.Circle((x, y), r, linewidth=1.5, edgecolor='green', alpha=0.75, facecolor='none')
		ax.add_patch(circ)

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
			x1, y1 = wcs.all_world2pix(az1, alt1, 0.) # Calculate x y coordinates of first star on chip
			ybc_star2 = ybc[ybc['HR'] == int(con[2][2*i+1])] # Look up second star in YBC
			alt2, az2 = radec2altaz(float(ybc_star2['RA']), float(ybc_star2['Dec']), mjd, location=location)
			x2, y2 = wcs.all_world2pix(az2, alt2, 0.) # Calculate x y coordinates of second star on chip
			if not np.isnan([x1, y1, x2, y2]).any():
				vertices = Polygon(np.array([[x1, y1], [x2, y2]]), False, edgecolor='black', facecolor='none', lw = 1, ls='solid', alpha=0.75)
				ax.add_patch(vertices)
	click = Click(ax, onclick, button=1)

	# Load existing distortion table
	try:
		corrections = pickle.load(open(path_to + '/' + "distortion.pkl", "rb"))
		print(path_to + '/distortion.pkl loaded')
	except (OSError, IOError) as e:
		print(path_to + '/distortion.pkl does not exist. It will be created.')

	plt.draw()
	plt.show()

	# Either quit; save the distortions and quit; or save the distortions, attempt to interpolate corrections and quit
	selection = input("Select how to proceed:\n1\tQuit without saving\n2\tSave distortions and quit\n3\tSave the distortions, interpolate corrections and quit\nSelection: ")
	if int(selection) == 2:
		with open(path_to + '/' + 'distortion.pkl', 'wb') as f:
			pickle.dump(corrections, f)
	elif int(selection) == 3:
		with open(path_to + '/' + 'distortion.pkl', 'wb') as f:
			pickle.dump(corrections, f)

		# Interpolate corrections onto 2D meshgrid of sensor and calculate a new corrected coordinate system
		x = np.linspace(0,1280,320)
		y =  np.linspace(0,960,240)
		X, Y = np.meshgrid(x,y)
		xx = X.ravel()
		yy = Y.ravel()
		p_init = models.Polynomial2D(degree=4)
		fit_p = fitting.LevMarLSQFitter()
		fx = fit_p(p_init, corrections['x1'], corrections['y1'], (corrections['x2'] - corrections['x1']))
		fy = fit_p(p_init, corrections['x1'], corrections['y1'], (corrections['y2'] - corrections['y1']))
		xshifts = fx(X, Y)
		yshifts = fy(X, Y)

		# Look at the distortion map that we have to apply after the WCS solution
		# plt.figure()
		# plt.scatter(X, Y, c=xshifts)#, vmin=-15, vmax=10, s=40)
		# cb = plt.colorbar()
		# cb.set_label('x-shift (pix)')
		# plt.figure()
		# plt.scatter(X, Y, c=yshifts)#, vmin=-15, vmax=10, s=40)
		# cb = plt.colorbar()
		# cb.set_label('y-shift (pix)')

		# Calculate new coordinate system including the mapped distortion
		xx = X.ravel()
		yy = Y.ravel()
		wcs_refined = Fisheye(wcs_initial, xx, yy, np.reshape(xshifts,-1), np.reshape(yshifts,-1))

		# Save coordinate system
		wcs_refined.save(path_to + '/' + 'wcs-refined.npz')
