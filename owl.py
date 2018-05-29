#This is the beginning of the VDT written by Vivian Bakiris 

import sys
import os
import skimage as ski
from skimage import io, color, util 
import imghdr
from matplotlib import pyplot as plt
import time 
import prox_tv as ptv
from skimage import img_as_uint
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
import matplotlib.pyplot as plt
import re 
import subprocess



metrics = list() 
# A list of all the metric values obtained from the image registration 
images = list()
# A list of the names of all the images developed in the denoising process 

types = {'png': 'png', 'tiff': 'tif', 'jpeg': 'jpg'}

#File Input 
def input(filename):
	
	print ("Welcome to VDT: Vivian's Denoising Thing")
	print("You would like to denoise " + filename)	
	
	#Define a path to the current working directory
	here = os.path.dirname(os.path.abspath(__file__))

	#Add folder name to path, if applicable 
	if args.folder: 
		here = here + "/" + args.folder

	#Include the file name of interest to the path 
	filepath = here + "/" + filename 

	#Determine filetype
	filetype = imghdr.what(filepath)


	#If the filetype has not been tested for compatabilitiy 
	#Program will exit. 

	if filetype in types:
		print("Your file type is compatible")
	else:
		print("Your file type is not compatible yet, sorry!")
		exit(1)

	#Read Image 
	X = io.imread(filepath)

	#Convert to float for processing 
	X = ski.img_as_float(X)
	#X = color.rgb2gray(X)

	return X 

def input2(filename):
	here = os.path.dirname(os.path.abspath(__file__))

	filepath = here + "/" + filename 

	filetype = imghdr.what(filepath)
	X = io.imread(filepath)

	if filetype in types:
		print("Your file type is compatible")
	else:
		print("Your file type is not compatible yet, sorry!")

	X = ski.img_as_float(X)

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
	print(abs(R.GetMetricValue()))

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

	F = ptv.tv1_2d(inputFile, lamb,1 ,3)

	end = time.time()

	print('Time elapsed ' + str(end-start))

	return F 

#Automatically denoises a file within a given range 
#Need to change this to take in start and fin parameters. 

def denoiseAutomated(inputFile):
	if args.automationmethod == "Pairwise":
		
		denoiseAutomatedPairWise(inputFile)
	else: 
		
		denoiseAutomatedOriginal(inputFile)	

def denoiseAutomatedOriginal(inputFile):
	print("Start Original Method of Automated Denoising")

	fileToDenoise = inputFile
	previousFile = inputFile
	original = "hello"

	for x in np.arange(args.startscope,args.endscope,args.increment):

		print x

		step = denoiseManual(fileToDenoise,x)
		name = "Temporary" + str(x) + ".png"

		io.imsave(str(name), step)

		images.append(name)
		if x == 0:
			original = name 

		register(name, original)



def denoiseAutomatedPairWise(inputFile):
	print("Start Pairwise method of Automated Denoising")

	fileToDenoise = inputFile
	prevx = 0.0 

	for x in np.arange(args.startscope,args.endscope,args.increment):

		step = denoiseManual(fileToDenoise,x)

		current = "Temporary" + str(x) + ".png"

		previous = "Temporary" + str(prevx) + ".png"

		print("Saving the file name", str(current))

		io.imsave(str(current), step)

		images.append(current)

		register(current, previous)

		prevx = x 

#Should remove the global variables - have it take in a lambda 

def calculateBest():

	dif=np.diff(metrics)
	dif2=np.diff(dif)
	
	fn=True
	for k,i in enumerate(dif2):
		if i<0:
			if not fn:
				print(k+1)
				if ((k+1) > 5):
					break

			fn=False

	print (k+1)

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

	io.imsave(str(outname), out)
	

	name = "showFile" #+ str(number)
	img = cv2.imread(str(outname),0)
	imS = cv2.resize(img, (700, 760))  
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	cv2.moveWindow(name, 05,20);
	cv2.imshow(name,imS)

	cv2.waitKey(0) #closes after a key press 
	cv2.destroyAllWindows()
	'''
	imgOriginal = cv2.imread(args.file,0)
	imOriginal = cv2.resize(imgOriginal,(700,760))
	cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
	cv2.moveWindow('original image', 710,20)
	cv2.imshow('original image', imOriginal)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''

	#image = Image.open("tempfile.png")
	#image.show()


	#fig=plt.figure(figsize=(14, 4))
	#columns = 3
	#rows = 1
	#for i in range(1, columns*rows +1):
	#	fig.add_subplot(rows, columns, i)
	#	io.imshow(file)
	#plt.show()


def openFile(filename):

	print("Open file function!")
	

	name = "Denoised File" #+ str(number)
	img = cv2.imread(filename,0)
	imS = cv2.resize(img, (700, 760))  
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	cv2.moveWindow(name, 05,20);
	cv2.imshow(name,imS)
	#cv2.waitKey(0) #closes after a key press 
	#cv2.destroyAllWindows()

	imgOriginal = cv2.imread(args.file,0)
	imOriginal = cv2.resize(imgOriginal,(700,760))
	cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
	cv2.moveWindow('original image', 710,20)
	cv2.imshow('original image', imOriginal)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def denoiseManualIteration(inputFile,lamb):

	#Change this so only one question is asked - makes it easier. 

	#Maybe we could display on a scale?

	#We should create another value - manual or lambda 


	userIsNotHappy = True 

	print("Before denoise manual")
	
	image = inputFile
	#showFile(inputFile)

	while userIsNotHappy:

		image = denoiseManual(inputFile, lamb)
		showFile(image)
		#showFile(inputFile)
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
			

			print(lamb)


	return image

#File Output
def output(outputFile):

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		out = img_as_uint(outputFile)
	
	outname = outfileName()

	io.imsave(str(outname), out)


def outfileName(file):

	filetype = getFileType(file)

	fileExtension = filetypeConversions(filetype)
	outfileName = "outtathisworld." + fileExtension

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

	return filetype 

def saveFileFinal(outfilename, filename):
	fileToSave = input2(outfilename)
	newFileName = "Denoised_" + filename 
	io.imsave(newFileName, fileToSave)

	#pass

#Change this name
def checkfiletype(outfiletype):

	if outfiletype in types: 
		return outfiletype
	else:
		raise argparse.ArgumentTypeError("This output file type is not yet supported. Please check your spelling e.g. TIFF not TIF ")

def checkTrue(showfileresponse):

	#Change the case first and then check 

	yes = ("Yes","yes","YES","Y","y", "True","TRUE","T")
	no = ("No","NO","N","no","FALSE","False","F","f","false")

	if showfileresponse in yes:
		return 'T'
	elif showfileresponse in no:
		return 'F'
	else:
		raise argparse.ArgumentTypeError("This is not a valid option, please try either yes or no")

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

#Run
if __name__ == "__main__":

	#int lambdaValue 
	#increment = 0.01 

	start = time.time()

	parser = argparse.ArgumentParser()
	parser.add_argument("--file", help = "The file you want to denoise")
	parser.add_argument("--folder", help = "The folder containing the files you want to denoise")
	parser.add_argument("--lamb", help ="The lambda value for denoising", type = float)
	parser.add_argument('--outfiletype', help = "The output file type", type = checkfiletype)
	parser.add_argument('--showfile', help = "Display the end result denoised file", type = checkTrue)
	parser.add_argument('--automationmethod', help = "Select either pairwise or original registration", type = checkRegistration)
	parser.add_argument('--confirmfirst', help = "Confirm the automation before proceeding with the whole file", type = checkTrue)
	parser.add_argument('--startscope', help = "The point to begin denoising", type = float, default = 0.00)
	parser.add_argument('--endscope', help ="The point to end denoising", type = float, default = 0.12)
	parser.add_argument('--increment', help ="The increment between denoising iterations", type = float, default = 0.05)
	parser.add_argument('--specifyfile', help ="To specify a start file to obtain the lambda value in an unskewed way", type = checkFile)
	

	args = parser.parse_args()

	#Create a show graphs thing 
	#print(args.showfile)

	#Due to argparse we should be able to remove this try.
	if args.file: 
		try: 
			newfile = input(args.file)
		except IndexError:
			print("You need to give us a file mate, otherwise how will I know what to denoise?")
			print("Soon we will add the ability to use one of our test files as default!")
			exit(1)

	elif args.folder: 

		pass
	else:
		print("Please select either a file or a folder to submit")
		#Denoising a sample image 



	if args.folder: 

		arr = os.listdir('.')
		givenFolder =  args.folder + '/'

		obtainedLambda = False 

		for filename in os.listdir(givenFolder):
			print filename

			#Change this to the filetype that thing is. 
			if filename.endswith(".png"):

				newfile = input(filename)

				if obtainedLambda is False: 
					print ("ITS FALSE MATE")
					denoiseAutomated(newfile)
					fileIndex = calculateBest()
					lambdaValue = args.increment * int(fileIndex)

					#Let's see. 


					plt.plot(metrics,'ro')
					plt.show()
					

					if args.confirmfirst == "T":
						iterationfile = input2(images[fileIndex])
						manitfile = denoiseManualIteration(iterationfile,lambdaValue)
						newFileName = "Denoised_" + filename 
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
				

	else:
		#print("Does it go here?")

		if args.lamb:
			#print("What about here??")
			#print(args.lamb)

			denoisedFile = denoiseManual(newfile, args.lamb)

			savefilename = "Denoised_" + args.file

			io.imsave(str(savefilename), denoisedFile)

			if args.showfile == "T":
				showFile(denoisedFile)

			#showFile(denoisedFile)

	
		else: 
			#print("Does this go here?")
			denoiseAutomated(newfile)
			fileIndex = calculateBest()
			
			saveFileFinal(images[fileIndex], args.file)

			if args.showfile == "T":
				openFile(images[fileIndex])

			plt.plot(metrics,'ro')
			plt.show()
		
			f = open('datafull.csv','a')
			data = images[fileIndex] + '\t' + args.file + '\n'
			f.write(data)
			f.close()
			


		
#Open file is used for automated
#Show file is used for manual
