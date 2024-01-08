# astro-wcs
Fits a world coordinate systems (WCS) to photographs from all-sky meteor cameras.

The motivation behind this was to replace the need for plate solving since I was finding it very difficult to get traditional plate solving tools to work on my all-sky photographs.

## Files

### Run files
1. `fit-initial.wcs.py`
1. `refine-wcs.py`
1. `augment-all.py`
1. `convert-to-sip.py`
1. `reproject-image.py`

### Data files
- `bsc5.dat`
- `HabHYG.csv`
- `ConstellationLines.dat`

### Utility files
- `fisheye_map.py`
- `fit_wcs.py`
- `read_ybc.py`
- `utils.py`


## Basic operation
You need to run the files on the command line as follows (after changing directory to the folder that contains the `.py` files). Obviously, install the pre-requisite libraries first (see the initial lines of code in each file). Some of the scripts use command line arguments and so you can run the script with "--help" to get the help message with usage guidance.

1. `python fit-initial-wcs.py path/to/image.fits path/to/master_dark.tiff` : This will plot an image, detect all (or some) of the stars, and allow you to click on a star, type in its name or identifier, select it from the list, and log the coordinates. When you close the window, it will compute an initial WCS fit (for fisheye all-sky images, the fit is usually OK but not great), and save the bright star locations. You can reload the same image and continue adding more stars. If you make a mistake when adding a star, just hit enter instead of typing a star identifier or number, and it will deselect the star, and allow you to try another one instead. The vector of bright stars and the initial WCS are both saved in the same folder as the fits file.

1. `python refine-wcs.py path/to/image.fits path/to/master_dark.tiff` : This allows you to open any image (as long as the camera hasn’t moved since fitting the initial wcs), and overlays the expected star locations and constellations. You first need to click on the catalogue star (green), and then the detected star (blue), and it will log the difference. When you quit, it will ask you if you want to quit or save. If you save, it will create a WCS with a distortion table and save it in the same folder as the fits file. You can rerun this command as many times as you like, and with different images so you can refine the WCS for different times of the night, or for when the stars have moved into an area that was previously a bit sparse. I find it helpful to make a few refinements, and then save and reload. You can see how effective your refinements have been. If you make a mistake, you can just quit and not save, and then start again. (Another good reason to save a few, and then stop and save – but note, the first pass needs at least 15 stars before you can save.)

1. `augment-all.py` : Overlays RA-Dec and Alt-Az grids, and plots stars and constellations.

1. `reproject-image.py` kind of works. It doesn’t use command line arguments (so the parameters are hard coded). If you run it, it will reproject an image into a new projection, e.g. gnomonic, or mercator. The problems are: (1) The original distortion is high and so the images get severely stretched. (2) It doesn’t work with WCS with distortion tables. That’s why I had a look at converting the WCS to a SIP WCS, using polynomial corrections instead of a distortion table. Polynomial corrections via SIP is standard, but I can’t get the WCS fit to converge.

1. `convert-to-sip.py` : Doesn't work very well – I tried to fit a SIP WCS instead of using a distortion table (which is what `refine-wcs.py` does). It kind of works, but only for an order 2 polynomial and no higher – I think it needs at least order 3 to be accurate for the high distortion.

1. `reproject-image.py` :  This should work better when `convert-to-sip.py` works properly. Note that this script doesn’t work with distortion tables, only polynomials. I reported the [distortion tables bug](https://github.com/astropy/reproject/issues/213) to the Astropy GitHub page.
