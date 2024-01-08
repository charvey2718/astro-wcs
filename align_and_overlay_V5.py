import cv2
import numpy as np
import os
import matplotlib.pylab as plt
import astropy.units as u
from astropy import wcs
from astropy.io import fits
from astropy.time import Time
from astropy.utils.data import get_pkg_data_filename
from astropy.coordinates import EarthLocation
from PIL import Image, ImageEnhance 
from utils import radec2altaz, altaz2radec
from skimage.transform import warp

path = r"C:\Users\samue\OneDrive - Loughborough University\Part D\FYP\Python_code\Photometry\2020-01-19\Capture\00_11_08"
images= []
for file in os.listdir(path):
    if file.endswith(".fits"):
        images.append(file)
images = images[:-1]
diff_images = ['Capture_00043.fits', 'Capture_00200.fits','Capture_00300.fits', 'Capture_00417.fits', 'Capture_00418.fits', 'Capture_00419.fits', 'Capture_00555.fits', 'Capture_00556.fits', 'Capture_00557.fits', 'Capture_00558.fits', 'Capture_00594.fits', 'Capture_00595.fits', 'Capture_00596.fits', 'Capture_00597.fits', 'Capture_00613.fits']
diff_prev_images = ['Capture_00042.fits', 'Capture_00199.fits','Capture_00299.fits', 'Capture_00416.fits', 'Capture_00417.fits', 'Capture_00418.fits', 'Capture_00554.fits', 'Capture_00555.fits', 'Capture_00556.fits', 'Capture_00557.fits', 'Capture_00593.fits', 'Capture_00594.fits', 'Capture_00595.fits', 'Capture_00596.fits', 'Capture_00612.fits']
stacksize = None
longitude = 1.2062
latitude = 52.7721
location = EarthLocation(lat=latitude*u.degree, lon=longitude*u.degree, height=72.0*u.meter)
# Time stamp of reference frame
hdr = fits.getheader(path + "/" + images[500], 0)
date = hdr['DATE-OBS']
t = Time(date, format='isot', scale='utc')
mjd2 = t.mjd # Modified Julian Date of the reference frame
diffs = []

for image in images:
    original_img = get_pkg_data_filename(path + "/" + image)
    image_data = fits.getdata(original_img)
    image_data = cv2.flip(image_data, 0) # flip vertical - FITS are stored 'upside down'
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BAYER_RG2RGB) # debayer into RGB - Decode XY2RGB bayer pattern
    height, width = image_data.shape[0:2] # size of image
    dark = plt.imread(r"C:\Users\samue\OneDrive - Loughborough University\Part D\FYP\Python_code\Photometry\2020-01-19\Capture\00_11_08\pipp_20200119_113611\logs\pipp_master_dark.tif") # need to 'pip install image' for PIL
    dark = cv2.cvtColor(dark, cv2.COLOR_BAYER_RG2RGB)
    
    calibrated_image = cv2.subtract(image_data, dark) # subtract master dark frame
    calibrated_image = cv2.flip(calibrated_image, 1) # flip horizontal          
    mask = cv2.imread(path + r"\Mask.tif", cv2.IMREAD_GRAYSCALE)
    mask = cv2.flip(mask, 1)
    # calibrated_image = cv2.bitwise_and(calibrated_image, calibrated_image, mask = mask)
    hdu_altaz = fits.open(path + '/wcs_sip.fits')
    wcs_altaz = wcs.WCS(hdu_altaz[0]) 
    
    # Time stamp of image
    hdr = fits.getheader(path + "/" + image, 0)
    date = hdr['DATE-OBS']
    t = Time(date, format='isot', scale='utc')
    mjd = t.mjd # Modified Julian Date of the input frame
    
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
        
    # Align with reference image
    calibrated_image = cv2.bitwise_and(calibrated_image, calibrated_image, mask = mask)
    remapped_img = warp(calibrated_image, imw, order=3)
    
    if image in diff_images:
        # Previous Image
        prev_img = get_pkg_data_filename(path + "/" + images[images.index(image)-1])
        prev_image_data = fits.getdata(prev_img)
        prev_image_data = cv2.flip(prev_image_data, 0) # flip vertical - FITS are stored 'upside down'
        prev_image_data = cv2.cvtColor(prev_image_data, cv2.COLOR_BAYER_RG2RGB) # debayer into RGB - Decode XY2RGB bayer pattern
        prev_calibrated_image = cv2.subtract(prev_image_data, dark) # subtract master dark frame
        prev_calibrated_image = cv2.flip(prev_calibrated_image, 1) # flip horizontal
        diff = cv2.subtract(calibrated_image, prev_calibrated_image) 
        # cv2.imwrite('diff.jpg', diff)
        # im = Image.open("diff.jpg")
        # enhancer = ImageEnhance.Brightness(im)
        # enhanced_diff = enhancer.enhance(2.5)
        #enhanced_im.save("enhanced_diff.jpg")
        # cv2.namedWindow('diff',cv2.WINDOW_NORMAL)
        # cv2.imshow('diff',diff)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray_diff, 5, 255, cv2.THRESH_TOZERO)
        thresh_diff = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
    
        # cv2.namedWindow('thresh_diff',cv2.WINDOW_NORMAL)
        # cv2.imshow('thresh_diff',thresh_diff)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
        
        remapped_diff = warp(thresh_diff, imw, order=3)
        # remapped_diff = cv2.GaussianBlur(remapped_diff, (5,5), 0)
        # Want to save all the diffs and add them at the end of the for loop
        diffs.append(remapped_diff)
        
    # First image in the stack
    if stacksize is None:
        stacksize = np.zeros((height, width))
        stack = np.zeros((height, width, 3))
    
    # Transform the mask
    mask_warped = mask
    mask_warped = (remapped_img[:,:,0] + remapped_img[:,:,1] + remapped_img[:,:,2] < 1e-2)
    mask_warped = cv2.dilate(mask_warped.astype(np.uint8),np.ones((5,5),np.uint8),iterations = 1).astype(bool)
    
    # Stack 
    oldstacksize = np.copy(stacksize)
    include = np.invert(mask_warped)
    stacksize[include] += 1
    for i in range(0, 3, 1): # Cycle through RGB
        stack[include, i] = np.multiply(stack[include, i],np.divide(oldstacksize[include],stacksize[include])) + np.divide(remapped_img[include, i],stacksize[include])
        
    print((images.index(image)/len(images))*100)

    #stack = cv2.add(stack, remapped_diff)

for diff in diffs:
   stack = cv2.add(stack, diff)
   
dpi = 50 # nothing really to do with dpi or inches or actual figure size - matplotlib is just strange in this regard
fig = plt.figure(figsize=(width/dpi,height/dpi), dpi=dpi)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax, projection=wcs)
ax.imshow(stack, origin='upper')
		
plt.draw()
plt.savefig('align_and_overlay_V5.png', format='png')
plt.show()
plt.close(fig)