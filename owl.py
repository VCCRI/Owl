#This is the beginning of the VDT written by Vivian Bakiris 

import sys
import os
import skimage as ski
from skimage import io, color, util, img_as_uint
import imghdr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time 
import prox_tv as ptv
import warnings
from PIL import Image
import argparse
import cv2
import SimpleITK as sitk
import sys
import os
from math import pi
from numpy import diff
import glob
import numpy as np
import re 
import subprocess
from math import factorial


metrics = list() 
# A list of all the metric values obtained from the image registration 
images = list()
# A list of the names of all the images developed in the denoising process 

types = {'png': 'png', 'tiff': 'tif', 'jpeg': 'jpg', 'dicom':'dcm'}
#A dictionary of common file conversions 

#Given a file name this function will return an np.array representation of this image 
def input(filename):
		
	#Define a path to the current working directory
	here = os.path.dirname(os.path.abspath(__file__))

	#Add folder name to path, if applicable 
	if args.folder: 
		here = here + "/" + args.folder

	#Include the file name of interest to the path 
	filepath = here + "/" + filename 

	#Determine filetype
	filetype = imghdr.what(filepath)


	#If the filetype has not been tested for compatabilitiy program will exit. 

	if filetype in types:
		#print("Your file type is compatible")
		pass
	else:
		print("Your file type is not compatible yet, sorry!")
		exit(1)

	#Read Image 
	X = io.imread(filepath)

	#Convert to float for processing
	with warnings.catch_warnings():
		warnings.simplefilter("ignore") 
		X = ski.img_as_float(X)
		X = color.rgb2gray(X)


	return X 
#Input file function to be used during 3D slice batch processing
def input2(filename):
	#Define a path to the current working directory
	here = os.path.dirname(os.path.abspath(__file__))

	filepath = here + "/" + filename 

	filetype = imghdr.what(filepath)
	X = io.imread(filepath)

	if filetype in types:
		#print("Your file type is compatible")
		pass
	else:
		print("Your file type is not compatible yet, sorry!")

	X = ski.img_as_float(X)
	X = color.rgb2gray(X)

	return X 
def command_iteration(method) :
    #if (method.GetOptimizerIteration()==0):
    #    print("Scales: ", method.GetOptimizerScales())
    #print("{0:3} = {1:7.5f} : {2}".format(method.GetOptimizerIteration(),
                                           #method.GetMetricValue(),
                                           #method.GetOptimizerPosition()))
	pass
def register(moving, fixed): 


	#Register start time 
	start = time.time()

	#Determine file path to fixed image and read. 
	here = os.path.dirname(os.path.abspath(__file__))
	filepath = here + "/" + str(fixed)
	fixed = sitk.ReadImage(filepath, sitk.sitkFloat32)

	#Determine filepath to moving image and read
	filepath2 = here + "/" + str(moving)
	moving = sitk.ReadImage(filepath2, sitk.sitkFloat32)


	R = sitk.ImageRegistrationMethod()
	R.SetMetricAsMattesMutualInformation(numberOfHistogramBins = 50)
	
	sample_per_axis=12
	if fixed.GetDimension() == 2:
		tx = sitk.Euler2DTransform()
		# Set the number of samples (radius) in each dimension, with a
		# default step size of 1.0
		R.SetOptimizerAsExhaustive([sample_per_axis//2,0,0])
		# Utilize the scale to set the step size for each dimension
		R.SetOptimizerScales([2.0*pi/sample_per_axis, 1.0,1.0])
	elif fixed.GetDimension() == 3:
		tx = sitk.Euler3DTransform()
		R.SetOptimizerAsExhaustive([sample_per_axis//2,sample_per_axis//2,sample_per_axis//4,0,0,0])
		R.SetOptimizerScales([2.0*pi/sample_per_axis,2.0*pi/sample_per_axis,2.0*pi/sample_per_axis,1.0,1.0,1.0])

	# Initialize the transform with a translation and the center o
	# rotation from the moments of intensity

	tx = sitk.CenteredTransformInitializer(moving, fixed, tx)
	R.SetInitialTransform(tx)
	R.SetInterpolator(sitk.sitkLinear)
	#R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )
	outTx = R.Execute(moving,fixed)
	#print("-------")
	#print(outTx)
	#print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
	#print(" Iteration: {0}".format(R.GetOptimizerIteration()))
	#print(" Metric value: {0}".format(R.GetMetricValue()))
	metrics.append(abs(R.GetMetricValue()))
	end = time.time()

	print('Time to Register ' + str(end-start))
#Denoises a given file, based off specified lambda 
#Determined by user. 
def denoise (inputFile):

	print("Starting denoising")

	start = time.time()

	F = ptv.tv1_2d(inputFile, args.lamb,1,3)

	end = time.time()

	print('Time to denoise ' + str(end-start))
	return F 
#Denoises a given file, based of specified lambda value
def denoiseManual(inputFile,lamb):


	start = time.time()

	F = ptv.tv1_2d(inputFile, lamb,1,2)

	end = time.time()

	print('Time to denoise ' + str(end-start))

	return F 

#Automatically denoises a file within a given range 
#Need to change this to take in start and fin parameters. 

def denoiseAutomated(inputFile):
	if args.automationmethod == "Pairwise":
		
		denoiseAutomatedPairWise(inputFile)
	else: 
		
		denoiseAutomatedOriginal(inputFile)	

def denoiseAutomatedOriginal(inputFile):
	print("Starting Original Method of Automated Denoising...")
	fileToDenoise = inputFile
	previousFile = inputFile
	original = "hello"
	count = 0 
	for x in np.arange(args.startscope,args.endscope,args.increment):

		step = denoiseManual(fileToDenoise,x)
		name = "Temporary" + str(x) + ".png"

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			io.imsave(str(name), step)

		images.append(name)
		if count == 0:
			original = name 
		
		register(name, original)
		count = count + 1

def denoiseAutomatedPairWise(inputFile):
	print("Starting Pairwise method of Automated Denoising...")

	fileToDenoise = inputFile
	prevx = 0.0 

	for x in np.arange(args.startscope,args.endscope,args.increment):

		step = denoiseManual(fileToDenoise,x)

		current = "Temporary" + str(x) + ".png"

		previous = "Temporary" + str(prevx) + ".png"

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			io.imsave(str(current), step)

		images.append(current)

		register(current, previous)

		prevx = x 

def calculateBest(metricsx):

	dif=np.diff(metricsx)
	dif2=np.diff(dif)
	
	fn=True
	fa = True 
	for k,i in enumerate(dif2):
		#print(k)
		if i>0:
			if not fn: 
				fa = False 
		if i<0:
			#print("Reaching correct", k)
			if not fn and not fa: 
				break 
				
			fn=False
	#print("k is: ", k)
	return (k)

def showFile(file):

	print("Show file function!")

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		out = img_as_uint(file)
	
	if args.file: 
	
		outname = outfileName(args.file)
	else: 
		outname = "hello.png"

	#You need to save the image you want to see 

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")

		io.imsave(str(outname), out)
	

	name = "showFile" #+ str(number)
	img = cv2.imread(str(outname),0)
	imS = cv2.resize(img, (700, 760))  
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	cv2.moveWindow(name, 05,20);
	cv2.imshow(name,imS)

	cv2.waitKey(0) #closes after a key press 
	cv2.destroyAllWindows()

def openFile(filenameNew, filenameOriginal):

	print("Open file function!")
	

	name = "Denoised File" #+ str(number)
	img = cv2.imread(filenameNew,0)
	imS = cv2.resize(img, (700, 760))  
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	cv2.moveWindow(name, 710,20);
	cv2.imshow(name,imS)
	#cv2.waitKey(0) #closes after a key press 
	#cv2.destroyAllWindows()

	imgOriginal = cv2.imread(filenameOriginal,0)
	imOriginal = cv2.resize(imgOriginal,(700,760))
	cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
	cv2.moveWindow('original image', 05,20)
	cv2.imshow('original image', imOriginal)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def denoiseManualIteration(inputFile):

	#Change this so only one question is asked - makes it easier. 

	#Maybe we could display on a scale?

	#We should create another value - manual or lambda 

	if args.iterationlambda:
		lamb = args.iterationlambda
	else: 
		lamb = 0.5


	userIsNotHappy = True 
	
	image = inputFile
	#showFile(inputFile)

	while userIsNotHappy:
		print("Denoising with lambda value", lamb)
		image = denoiseManual(inputFile, lamb)
		showFile(image)
		#showFile(inputFile)
		happiness = raw_input("Are you happy though? ")

		if happiness == "yes":
			userIsNotHappy = False
		else:
			userIsNotHappy = True
			moreOrLess = raw_input("Would you like to denoise more or less? ")
			if moreOrLess == "more":
				lamb = lamb + float(args.increment)
			else:
				lamb = lamb - float(args.increment) 
				if lamb < 0:
					lamb = 0
					print("Minimum lambda value reached")
			


	return image
#File Output
def output(outputFile):

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		out = img_as_uint(outputFile)
	
	outname = outfileName()

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		io.imsave(str(outname), out)

def outfileName(file):

	filetype = getFileType(file)

	fileExtension = filetypeConversions(filetype)
	outfileName = "outputFile." + fileExtension

	return outfileName 

def filetypeConversions(filetype):

	if args.outfiletype: 
		return types[args.outfiletype]
	else: 
		return types[filetype]

def getFileType (filename):

	here = os.path.dirname(os.path.abspath(__file__))

	if args.folder:
		filepath = here + "/" + args.folder + "/" + filename
	else: 
		filepath = here + "/" + filename
	

	filetype = imghdr.what(filepath)

	if args.outfiletype:
		return args.outfiletype

	return filetype 

def saveFileFinal(outfilename, filename):
	fileToSave = input2(outfilename)	
	newFileName = "Denoised_" + filename 
	if args.outfiletype:
		newFileName = newFileName + "." + str(filetypeConversions(args.outfiletype))
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		io.imsave(newFileName, fileToSave)

	#pass

#Change this name
def checkfiletype(outfiletype):

	if outfiletype in types: 
		return outfiletype
	else:
		raise argparse.ArgumentTypeError("This output file type is not yet supported. Please check your spelling e.g. TIFF not TIF ")


def checkFile(filename):

	#Create the filepath

	if args.file:
		here = os.path.dirname(os.path.abspath(__file__))
		path = here + "/" + str(filename)

	else:
		path = os.path.dirname(os.path.abspath(__file__))
		path = here + "/" + str(args.folder) + "/"+ str(filename)


	if os.path.exists(path):
		return filename
	else: 
		raise argparse.ArgumentTypeError("Not a valid file")

def savitzky_golay(y, window_size, order, deriv=0, rate=1):

	try:
	    window_size = np.abs(np.int(window_size))
	    order = np.abs(np.int(order))
	except ValueError, msg:
	    raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
	    raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
	    raise TypeError("window_size is too small for the polynomials order")
	order_range = range(order+1)
	half_window = (window_size -1) // 2
	# precompute coefficients
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
	# pad the signal at the extremes with
	# values taken from the signal itself
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m[::-1], y, mode='valid')


#A function to display the graph produced by automated denoising 
def showGraph(metrics,fileIndex):

	plt.plot(metrics, "bo")
	plt.title('Automated Denoising Graph')
	plt.ylabel('Metric')
	plt.xlabel('Iteration of Denoising')
	plt.savefig("outputgraph.png")
	#plt.show()

def checkRegistration(registrationoption):

	#Change the case first and then check 

	original = ("ORIGINAL","Original", "original", "O")
	pairwise = ("PAIRWISE", "pairwise", "Pairwise", "Pair Wise", "p", "pw", "pair wise")

	if registrationoption in original:
		return "Original"
	elif registrationoption in pairwise: 

		return "Pairwise"
	else: 
		raise argparse.ArgumentTypeError("This is not a valid option, please try either Pairwise or Original")

def dicom_to_array(filename):
	import pydicom 
	d = pydicom.read_file(filename)
	a = d.pixel_array
	return np.array(a)

def demo():

	# Load image
	here = os.path.dirname(os.path.abspath(__file__))
	X = io.imread(here + '/small.png')
	X = ski.img_as_float(X)
	X = color.rgb2gray(X)

	# Filter using 2D TV-L1
	lam=0.005;
	print('Filtering image with 2D TV-L1...')
	start = time.time()
	F = ptv.tv1_2d(X, lam,1,3)
	end = time.time()
	print('Elapsed time ' + str(end-start))
	io.imshow(X)
	plt.show()
	
	io.imshow(F)
	plt.show()



#Run
if __name__ == "__main__":

	start = time.time()

	parser = argparse.ArgumentParser()
	parser.add_argument("--file", help = "The file you want to denoise")
	parser.add_argument("--folder", help = "The folder containing the files you want to denoise")
	parser.add_argument("--lamb", help ="The lambda value for denoising", type = float)
	parser.add_argument('--outfiletype', help = "The output file type", type = checkfiletype)
	parser.add_argument('--showfile', help = "Display the end result denoised file", action = 'store_true')
	parser.add_argument('--automationmethod', help = "Select either pairwise or original registration", type = checkRegistration)
	#parser.add_argument('--confirmfirst', help = "Confirm the automation before proceeding with the whole file", type = checkTrue)
	parser.add_argument('--confirmfirst', help = "Confirm the automation before proceeding with the whole file", action = 'store_true')
	parser.add_argument('--startscope', help = "The point to begin denoising", type = float, default = 0.0)
	parser.add_argument('--endscope', help ="The point to end denoising", type = float, default = 0.2)
	parser.add_argument('--increment', help ="The increment between denoising iterations", type = float, default = 0.01)
	parser.add_argument('--specifyfile', help ="To specify a start file to obtain the lambda value in an unskewed way", type = checkFile)
	parser.add_argument('--demo', help = "Run a demo version of the program", action = 'store_true')
	parser.add_argument('--manualIteration', help = "To manually iterate through and select the best image", action = 'store_true')
	parser.add_argument('--iterationlambda', help = "To be used as a starting point for manual iteration", type = float)
	parser.add_argument('--showGraph', help = "Display the automation graphs", action = 'store_true')
	args = parser.parse_args()


	if args.demo: 
		print("Entering Demonstration Program")
		demoFile = "demo.png"
		newfile = input(demoFile)
		denoiseAutomated(newfile)
		fileIndex = calculateBest()	
		saveFileFinal(images[fileIndex], demoFile)
		openFile(images[fileIndex], demoFile)
		print("Demonstration Completed!")
		exit(0)

	#Due to argparse we should be able to remove this try.
	if args.file: 
		try: 
			newfile = input(args.file)
		except IndexError:
			print("You need to give us a file mate, otherwise how will I know what to denoise?")
			print("Soon we will add the ability to use one of our test files as default!")
			exit(1)

		if args.lamb:

			denoisedFile = denoiseManual(newfile, args.lamb)

			#Move this into function 
			savefilename = "Denoised_" + args.file
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				io.imsave(str(savefilename), denoisedFile)

			if args.showfile:
				showFile(denoisedFile)
			print("Manual Denoising of one file completed!")

			#demo()
			exit(0)


		elif args.manualIteration: 

			denoisedFile = denoiseManualIteration(newfile)
			#Change so its appropriate for the other file options 
			savefilename = "Denoised_" + args.file
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				io.imsave(str(savefilename), denoisedFile)

			if args.showfile:
				showFile(denoisedFile)

			print("Manual Iteration Denoising Completed!")
			exit(0)

		else: 
			denoiseAutomated(newfile)
			fileIndex = calculateBest(metrics)	

			print("The lambda value selected is:", int(fileIndex) * float(args.increment))

			saveFileFinal(images[fileIndex], args.file)

			if args.showfile:
				openFile(images[fileIndex], args.file)

			if args.showGraph:
				showGraph(metrics,fileIndex)

			print("Automated Denoising for one file completed!")

		for fileDelete in images:
					os.remove(fileDelete)





	elif args.folder:

		arr = os.listdir('.')
		givenFolder =  args.folder + '/'
		obtainedLambda = False 

		for filename in os.listdir(givenFolder):

			#Change this to the filetype that thing is. 
			if filename.endswith(".png"):
				newfile = input(filename)
				if obtainedLambda is False: 
					denoiseAutomated(newfile)
					fileIndex = calculateBest(metrics)
					lambdaValue = args.increment * int(fileIndex)

					if args.showGraph:
						showGraph(metrics, fileIndex)	

					if args.confirmfirst:
						iterationfile = input2(images[fileIndex])
						manitfile = denoiseManualIteration(iterationfile,lambdaValue)
						newFileName = "Denoised_" + filename 
						with warnings.catch_warnings():
							warnings.simplefilter("ignore")
							io.imsave(newFileName, manitfile)

					else:
						saveFileFinal(images[fileIndex], filename)
					
					obtainedLambda = True 

				else:
					fileToSave = denoiseManual(newfile, lambdaValue)
					newFileName = "Denoised_" + filename 
					io.imsave(newFileName, fileToSave)
	

				#Remove temporary files 
				for fileDelete in images:
					os.remove(fileDelete)

				#Clear metrics and list of images. 
				metrics = list()
				images = list() 
		print("Denoising of 3D slices completed!")

	else:
		print("Please select either a file or a folder to submit. ")
		print("You can do that with the --file or --folder flag")
		print("Alternatively, you can type python owl.py --demo for a demo run!")
		exit(1)
		#Denoising a sample image 

