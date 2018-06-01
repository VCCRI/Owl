#This is the beginning of the VDT written by Vivian Bakiris 

import sys
import os
import skimage as ski
from skimage import io, color, util, img_as_uint
import imghdr
import matplotlib.pyplot as plt
import time 
import prox_tv as ptv
import warnings
from PIL import Image
import argparse
import cv2
import SimpleITK as sitk
from math import pi
from numpy import diff
import glob
import numpy as np
import re 
import subprocess



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
		print("Your file type is compatible")
	else:
		print("Your file type is not compatible yet, sorry!")
		exit(1)

	#Read Image 
	X = io.imread(filepath)

	#Convert to float for processing
	with warnings.catch_warnings():
		warnings.simplefilter("ignore") 
		X = ski.img_as_float(X)
	return X 
#Input file function to be used during 3D slice batch processing
def input2(filename):
	#Define a path to the given file 
	here = os.path.dirname(os.path.abspath(__file__))
	filepath = here + "/" + filename 
	#Determine the filetype 
	filetype = imghdr.what(filepath)
	#Read the image 
	X = io.imread(filepath)

	#Check file compatability 
	if filetype in types:
		print("Your file type is compatible")
	else:
		print("Your file type is not compatible yet, sorry!")

	X = ski.img_as_float(X)

	return X 

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
	outTx = R.Execute(moving,fixed)
	metrics.append(abs(R.GetMetricValue()))
	end = time.time()

#Denoises a given file, based off specified lambda 
#Determined by user. 
def denoise (inputFile):

	print("Starting denoising")

	start = time.time()

	F = ptv.tv1_2d(inputFile, args.lamb,1,3)

	end = time.time()

	#print('Time to denoise ' + str(end-start))
	return F 

#Denoises a given file, based of specified lambda value
def denoiseManual(inputFile,lamb):

	start = time.time()

	F = ptv.tv1_2d(inputFile, lamb,1 ,3)

	end = time.time()

	#print('Time to denoise ' + str(end-start))

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
	for x in np.arange(args.startscope,args.endscope,args.increment):

		step = denoiseManual(fileToDenoise,x)
		name = "Temporary" + str(x) + ".png"

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			io.imsave(str(name), step)

		images.append(name)
		if x == 0:
			original = name 
		
		register(name, original)

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


def calculateBest():

	dif=np.diff(metrics)
	dif2=np.diff(dif)
	
	fn=True
	for k,i in enumerate(dif2):
		if i<0:
			if not fn:
				if ((k+1) > 3):
					break
				

			fn=False

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

	#Save the image in order to view the file without distortion 

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")

		io.imsave(str(outname), out)
	

	name = "showFile" 
	img = cv2.imread(str(outname),0)
	imS = cv2.resize(img, (700, 760))  
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	cv2.moveWindow(name, 05,20);
	cv2.imshow(name,imS)

	cv2.waitKey(0) #closes after a key press 
	cv2.destroyAllWindows()

def openFile(filenameNew, filenameOriginal):

	print("Open file function!")
	

	name = "Denoised File"
	img = cv2.imread(filenameNew,0)
	imS = cv2.resize(img, (700, 760))  
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	cv2.moveWindow(name, 710,20);
	cv2.imshow(name,imS)


	imgOriginal = cv2.imread(filenameOriginal,0)
	imOriginal = cv2.resize(imgOriginal,(700,760))
	cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
	cv2.moveWindow('original image', 05,20)
	cv2.imshow('original image', imOriginal)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def denoiseManualIteration(inputFile):

	if args.iterationlambda:
		lamb = args.iterationlambda
	else: 
		lamb = 0.5


	userIsNotHappy = True 	
	image = inputFile


	while userIsNotHappy:
		image = denoiseManual(inputFile, lamb)
		showFile(image)
		happiness = raw_input("Are you happy though? ")

		if happiness == "Yes":
			userIsNotHappy = False
		else:
			userIsNotHappy = True
			moreOrLess = raw_input("Would you like to denoise more or less? ")
			if moreOrLess == "more":
				lamb = lamb + 0.01
			else:
				lamb = lamb - 0.01
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

#A function to display the graph produced by automated denoising 
def showGraph(metrics,fileIndex):

	plt.plot(metrics,'ro')
	plt.title('Automated Denoising Graph')
	plt.ylabel('Metric')
	plt.xlabel('Iteration of Denoising')
	plt.show()


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
	parser.add_argument('--startscope', help = "The point to begin denoising", type = float, default = 0.00)
	parser.add_argument('--endscope', help ="The point to end denoising", type = float, default = 0.15)
	parser.add_argument('--increment', help ="The increment between denoising iterations", type = float, default = 0.05)
	parser.add_argument('--specifyfile', help ="To specify a start file to obtain the lambda value in an unskewed way", type = checkFile)
	parser.add_argument('--demo', help = "Run a demo version of the program", action = 'store_true')
	parser.add_argument('--manualIteration', help = "To manually iterate through and select the best image", action = 'store_true')
	parser.add_argument('--iterationlambda', help = "To be used as a starting point for manual iteration", type = float)
	parser.add_argument('--showGraph', help = "Display the automation graphs", action = 'store_true')
	args = parser.parse_args()

	defaultfiletype = ".png"

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
			fileIndex = calculateBest()	
			saveFileFinal(images[fileIndex], args.file)

			if args.showfile:
				openFile(images[fileIndex], args.file)

			if args.showGraph:
				showGraph(metrics,fileIndex)

			print("Automated Denoising for one file completed!")




	elif args.folder:

		arr = os.listdir('.')
		givenFolder =  args.folder + '/'
		obtainedLambda = False 

		for filename in os.listdir(givenFolder):

			if filename.endswith(defaultfiletype):
				newfile = input(filename)
				if obtainedLambda is False: 
					denoiseAutomated(newfile)
					fileIndex = calculateBest()
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

