import matplotlib.pylab as plt
from matplotlib.patches import Polygon
import numpy as np
import os
import argparse # pip install argparse
from astropy.io import fits
from astropy import wcs
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
import cv2 # pip install opencv-python
# from skimage import exposure # pip install scikit-image
from utils import radec2altaz, altaz2radec
from skimage.transform import warp
from fisheye_map import Fisheye, load_fisheye
import glob

if __name__ == "__main__":

	# Command line arguments
	parser = argparse.ArgumentParser(description='Reproject an image from its fitted WCS to a gnomonic projection (TAN).')
	parser.add_argument('path', help='path to search *.FITS')
	parser.add_argument('dark', help='master dark frame')
	parser.add_argument('reference', help='reference frame (for alignment)')
	parser.add_argument('-p', '--pattern', help='Unix style pathname pattern expansion', default="*.FITS")
	parser.add_argument('-g', '--longitude', type=float, help='longitude of observation station, default 1.2062 degrees', default=1.2062)
	parser.add_argument('-t', '--latitude', type=float, help='latitude of observation station, default 52.7721 degrees', default=52.7721)
	parser.add_argument('-m', '--mask', help='mask')
	args = parser.parse_args()
	dark_data = args.dark
	location = EarthLocation(lat=args.latitude*u.degree, lon=args.longitude*u.degree, height=72.0*u.meter)
	fits_files = glob.glob(args.path + '/' + args.pattern)

	# Choose one of the following fitted wcs
	# print('Attemping to load ' + args.path + '/wcs-refined.npz. If this fails, run refine-wcs.py first.')
	# wcs_altaz = load_fisheye(args.path + '/' + 'wcs-refined.npz') # Load alt-az coordinate system (after running calibration)
	print('Attemping to load ' + args.path + '/wcs_sip.fits. If this fails, run convert-to-sip.py first.')
	hdu_altaz = fits.open(args.path + '/wcs_sip.fits')
	wcs_altaz = wcs.WCS(hdu_altaz[0])

	# Time stamp of reference frame
	hdr = fits.getheader(args.reference, 0)
	date = hdr['DATE-OBS']
	t = Time(date, format='isot', scale='utc')
	mjd2 = t.mjd # Modified Julian Date of the reference frame

	mask = None
	stacksize = None
	if args.mask is not None:
		mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)==0
	
	stack = None
	for fits_file in fits_files: # loop through all fits files in directory
		# Load light frame
		image_file = get_pkg_data_filename(fits_file)
		image_data = fits.getdata(image_file)
		image_data = cv2.flip(image_data, 0) # flip vertical - FITS are stored 'upside down'
		image_data = cv2.cvtColor(image_data, cv2.COLOR_BAYER_RG2RGB) # debayer into RGB - Decode XY2RGB bayer pattern
		height, width = image_data.shape[0:2] # size of image

		# Time stamp of image
		hdr = fits.getheader(fits_file,0 )
		date = hdr['DATE-OBS']
		t = Time(date, format='isot', scale='utc')
		mjd = t.mjd # Modified Julian Date of the input frame

		# Dark frame subtraction
		dark_data = plt.imread(args.dark) # need to 'pip install image' for PIL
		dark_data = cv2.cvtColor(dark_data, cv2.COLOR_BAYER_RG2RGB) # debayer into RGB - See comment above
		calibrated = cv2.subtract(image_data, dark_data) # subtract master dark frame
		calibrated = cv2.flip(calibrated, 1) # flip horizontal

		if stacksize is None: # First image in stack
			if args.mask is None: # No mask supplied
				mask = np.zeros((height, width), dtype=bool)
			stacksize = np.zeros((height, width))
			stack = np.zeros((height, width, 3))

		def imw(xy): # Note this is the *inverse transformation*
			xx1 = xy[:,0] # Output image coordinates
			yy1 = xy[:,1]
			az1, alt1 = wcs_altaz.all_pix2world(xx1, yy1, 0.)
			# good = np.where(~(np.isnan(az1)) & ~(np.isnan(alt1)))
			ra1, dec1 = altaz2radec(alt1, az1, mjd2, location=location)
			alt2, az2 = radec2altaz(ra1, dec1, mjd, location=location)
			xx2, yy2 = wcs_altaz.all_world2pix(az2, alt2, 0.) # Input image coordinates
			xy2 = np.hstack((np.reshape(xx2,(len(xx2),1)),np.reshape(yy2,(len(yy2),1))))
			return xy2

		# Transform current frame to align with reference frame
		calibrated[mask] = (0,0,0)
		calibrated_remapped = warp(calibrated, imw, order=3) # bi-cubic
		
		# Transform mask if it exists
		mask_warped = mask
		if not args.mask is None:
			mask_warped = (calibrated_remapped[:,:,0] + calibrated_remapped[:,:,1] + calibrated_remapped[:,:,2] < 1e-2) # mask is set to be where the background level is < 3
			mask_warped = cv2.dilate(mask_warped.astype(np.uint8),np.ones((5,5),np.uint8),iterations = 1).astype(bool)

		# Stack using mean averaging
		# mean = (stack*stacksize + next)/(stacksize + 1) = stack*stacksize/(stacksize + 1) + next/(stacksize + 1)
		oldstacksize = np.copy(stacksize)
		include = np.invert(mask_warped)
		stacksize[include] += 1
		for i in range(0, 3, 1): # Cycle through RGB
			stack[include, i] = np.multiply(stack[include, i],np.divide(oldstacksize[include],stacksize[include])) + np.divide(calibrated_remapped[include, i],stacksize[include])

		# Write out image as 32-bit tiff
		cv2.imwrite(os.path.splitext(fits_file)[0] +'-aligned+stacked.tiff', cv2.cvtColor(stack.astype('float32'), cv2.COLOR_RGB2BGR))
		# cv2.imwrite(os.path.splitext(fits_file)[0] +'-mask.tiff', mask_warped.astype('float32'))

		# # Plot
		# dpi = 50 # nothing really to do with dpi or inches or actual figure size - matplotlib is just strange in this regard
		# fig = plt.figure(figsize=(width/dpi,height/dpi), dpi=dpi)
		# ax = plt.Axes(fig, [0., 0., 1., 1.])
		# ax.set_axis_off()
		# fig.add_axes(ax, projection=wcs)
		# ax.imshow(stack, origin='upper')
		
		# plt.draw()
		# plt.savefig(os.path.splitext(fits_file)[0] +'-reproject.png', format='png')
		# # plt.show()
		# plt.close(fig)
