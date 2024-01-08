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
HabHYG = read_csv('HabHYG.csv', sep=',', engine='python')#, dtype='str')
stars = np.array([], dtype=[('star_name', '<U10'), ('HD', 'uint32'), ('x', '<f8'), ('y', '<f8'), ('alt', '<f8'), ('az', '<f8')]) # Array to store identified star locations
keypoints = None


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
		

def onclick(event, ax):
	d = -1;
	di = 0;
	point = -1;
	if event.xdata is None or event.ydata is None:
		return
	global keypoints
	for idx, item in enumerate(keypoints):
		di = np.sqrt((event.xdata - item.pt[0])**2 + (event.ydata - item.pt[1])**2)
		if di < d or d < 0:
			d = di
			point = idx
	circ = patches.Circle((keypoints[point].pt[0],keypoints[point].pt[1]), keypoints[point].size, linewidth=1, edgecolor='r', facecolor='none')
	global stars
	ax.add_patch(circ)
	plt.draw()
	
	searchterm = input('Search for star in catalogue: ')
	match1 = HabHYG['Display Name'].str.contains(searchterm, flags=re.IGNORECASE)
	match2 = HabHYG['Hip'].astype('str').str.contains(searchterm, flags=re.IGNORECASE)
	match3 = HabHYG['HD'].astype('str').str.contains(searchterm, flags=re.IGNORECASE)
	match = match1 | match2 | match3
	if len(searchterm) > 0:
		for i, entry in enumerate(HabHYG[match].iterrows()):
			name = entry[1]['Display Name']
			hd = entry[1]['HD']
			hip = str(entry[1]['Hip'])
			if not np.isnan(hd):
				print(str(i + 1) + ': ' + name + ', HIP ' + hip + ', HD ' + str(int(hd)))
			else:
				print(str(i + 1) + ': ' + name + ', HIP ' + hip)
		selection = input('Select: ')
		try:
			val = int(selection)
			if val > 0 and val <= np.sum(match) and len(selection) > 0:
				name = HabHYG[match].iloc[[val - 1]]['Display Name'].values[0]
				HD = int(HabHYG[match].iloc[[val - 1]]['HD'].values[0])
				print('Added: ' + name + " at (" + str(keypoints[point].pt[0]) + ', ' + str(keypoints[point].pt[1]) + ")")
				stars = np.append(stars, np.array((name, HD, keypoints[point].pt[0], keypoints[point].pt[1], 0, 0), dtype=[('star_name', '<U10'), ('HD', 'uint32'), ('x', '<f8'), ('y', '<f8'), ('alt', '<f8'), ('az', '<f8')]))
			else:
				ax.patches[-1].remove()
				plt.draw()
		except ValueError:
			print('Invalid input')
			ax.patches[-1].remove()
			plt.draw()
	else:
		ax.patches[-1].remove()
		plt.draw()


if __name__ == "__main__":

	# Command line arguments
	parser = argparse.ArgumentParser(description='Fit an initial WCS to an all-sky FITS frame.')
	parser.add_argument('frame', help='FITS frame to fit an initial WCS to')
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
	#image_data = np.tranpose(image_data, (1,2,0))
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

	# Display calibration frame and make detected stars clickable
	fig, ax = plt.subplots()
	plt.imshow(calibrated)
	click = Click(ax, onclick, button=1)

	# Load existing files
	try:
		stars, ax.patches = pickle.load(open(path_to + '/' + "brightstars.pkl", "rb"))
		print(path_to + '/brightstars.pkl loaded')
	except (OSError, IOError) as e:
		print(path_to + '/brightstars.pkl does not exist. It will be created.')

	plt.draw()
	plt.show()

	# Calculate altitude azimuth coordinates of identified stars
	for star in stars: # Loop through the identified stars
		ybc_star = ybc[ybc['HD'] == star['HD']] # Look up each star in YBC
		alt, az = radec2altaz(float(ybc_star['RA']), float(ybc_star['Dec']), mjd, location=location)
		star['alt'] = alt # As 'star' is a reference, these numbers are stored back in the stars array above
		star['az'] = az

	# Save current state so it can be reloaded later
	with open(path_to + '/' + 'brightstars.pkl', 'wb') as f:
		pickle.dump([stars, ax.patches], f)

	# Fit an initial Zenith Equal Area projection to the identified stars
	print('Fitting WCS...')
	fun = wcs_air(stars['x'], stars['y'], stars['alt'], stars['az'], crpix1=width/2, crpix2=height/2, a_order=0, b_order=0)
	# rranges = (slice(0, width, 100), slice(0, height, 100), slice(-2, 2, 0.5), slice(-2, 2, 0.5), slice(-0.2, 0.2, 0.05), slice(-0.2, 0.2, 0.05), slice(-0.02, 0.02, 0.005), slice(-0.02, 0.02, 0.005))
	# from scipy import optimize
	# resbrute = optimize.brute(fun, rranges, full_output=True, finish=None)
	# x0 = resbrute[0] # np.array([width/2, height/2, 1., 1., 0.036, 0.0027, 0.00295, 0.0359])
	x0 = np.array([width/2, height/2, -1., 1., 0.036, 0.0027, 0.00295, 0.0359]) # try flipping signs of 3rd and 4th arguments if it won't converge
	fit_result = minimize(fun, x0, method='Powell')
	wcs_initial = fun.return_wcs(fit_result.x) # Convert the fit to a full WCS object to transform alt, az to chip x,y. Note, header says RA, Dec but is really Az, Alt.

	# Save coordinate system
	print('Saving WCS in ' + path_to + '/' + 'wcs-initial.pkl')
	with open(path_to + '/' + 'wcs-initial.pkl', 'wb') as f:
		pickle.dump(wcs_initial, f)