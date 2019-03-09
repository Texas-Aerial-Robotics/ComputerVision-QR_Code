# Purpose: This bash script will generate the quarters of each qr code for all of the possible 4-digit codes. The result of running this script
#          is a folder for each code that will contain 4 .png files. Each file will contain a fourth (top-left, top-right, bottom-left, bottom-right)
#          of the qr code for the 4-digit code.

# How to Use: Place this file in the folder that you want all of the test cases to be located in. Then run: bash qrGen.sh



# Mechanics of this Script:
# qr command format:
#	qr <data>

# convert command format:
#	convert - -crop <height>x<width>+<x-offset>+<y-offset> <output file name>



for ((i = 0; i < 10000; i++)); do
	# Image Constraints (edit these for your needs)
	height=145              # height of cropped image
	width=145               # width of cropped image

	# Create folder for qr code
	mkdir qr_$i
	cd qr_$i

	# Generate top left image
	qr "$i" | convert - -crop "$height"x"$width"+0+0 qr_topLeft_$i.png

	# Generate top right image
	qr "$i" | convert - -crop "$height"x"$width"+"$width"+0 qr_topRight_$i.png

	# Generate bottom left image
	qr "$i" | convert - -crop "$height"x"$width"+0+"$height" qr_bottomLeft_$i.png

	# Generate bottom right image
	qr "$i" | convert - -crop "$height"x"$width"+"$width"+"$height" qr_bottomRight_$i.png

	# Go back to overall directory for next code
	cd ..

done
