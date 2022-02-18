# brownian motion analysis

helper analysis package for BMC

## setup

install this package with either

```
git clone https://github.com/luclepot/brownian_motion_analysis.git
```

(non-ssh setup) or 

```
git clone git@github.com:luclepot/brownian_motion_analysis.git
```

Then cd into the directory and use your latest version of python (>= 3.6) to run the command

```
pip install .
```

to install required dependencies. 

## analysis

some introductory analysis of the tracked paths is given in `analysis.ipynb` notebook. Run this with jupyter as usual. There you can do tracking etc.

## driver usage (OLD, probably buggy)

To run the tracking driver, use

```
python tracker.py --input=<YOUR_INPUT_GLOB_STRING> --output=<YOUR_OUTPUT_FILE>
```

Make sure to include the equals signs if you're using glob strings, as they'll expand otherwise. 

If you wanted to process a folder of bitmap images named `data_1`, an example of good usage would be

```
python tracker.py --input=data_1/*.bmp --output=data_1/tracks.csv
```
