# User Requirements Check 

#### Input 2D Images

As a bioinformatician, I want to input an image directly into the application so that I can utilise this denoising application on all my datasets.   

##### Demonstration Command

`python owl.py --file demo.png`  

The above command will implement owl with a user specified image   

#### Output 2D Images 

As a bioinformatician, I want to be able to output an image in multiple file formats, so that I can easily input them into other image analysis pipelines in future. 

##### Demonstration Command 
`python owl.py --file demo.png --outfiletype jpeg`   

The above command will input a png file, and output a denoised jpeg file. 

#### Apply the ITV Algorithm 

As a bioinformatician, I want to apply the ITV algorithm, on my input images, so that my pictures are clear 

##### Demonstration Command
 `python owl.py --file demo.png`  
 
 The above command will apply the ITV Algorithm. 

#### Manually Input Lambda 

As a bioinformatician, I want to be able to manually input a lambda value into the denoising algorithm, so that I can moderate the output files to suit my personal needs. 

##### Demonstration Command 
`python owl.py --file demo.png --lamb 0.1` 

The above command will manually set the lambda value as 0.1. 

#### Manually Select Best Image 
As a bioinformatician, I want to be able to manually select the "best" output image, so that I can produce results that meet my needs. 

##### Demonstration Command
`python owl.py --file demo.png --manualIteration --iterationlambda 0.1`

After typing this command, you will be prompted with multiple questions to allow for manual iteration. For example "Are you happy?" "Would you like to denoise more or less". Iterationlambda is the value that denoising will begin. This parameter is option for the manual iteration method.  


#### Pairwise Automation Method 
As a bioinformatician, I want to be able to selet the pairwise automation method, so I can use subtle differences to determine the "best" image. 

##### Demonstration Command 
`python owl.py --file demo.png --automationmethod pairwise`

The above command will invoke the pairwise automation method on the file of choice.

#### Original Automation Method
As a bioinformatician, I want to be able to select the original comparison automation method, so I can use larger differences to determine the "best" images. 

##### Demonstration Command
`python owl.py --file demo.png --automationmethod original`

Alternatively 

`python owl.py --file demo.png` 

Owl will automatically denoise the given files, unless a manual lambda value is inputted using the --lamb flag. The default automation method is original. 


#### Select Automation Method
As a bioinformatician, I want to be able to select which automatino method is used on my data, so that I can produce the best image for my needs. 

##### Demonstration Command

`python owl.py --file demo.png --automationmethod original`

Alternative   

`python owl.py --file demo.png --automationmethod pairwise`  

The above commands demonstrate the different automation choices. 

#### View Best Option 

As a bioinformatician I would like to view which option was selected as the best by the image registration method   

##### Demonstration Command

`python owl.py --file demo.png --showFile`  

`python owl.py --file demo.png --automationmethod pairwise --showFile` 
 
The above commands will display the option selected as the best by each respective automation method. 

#### Display Automation Graphs 
As a bioinformatician, I want to be able to display the automation graphs, so that I have an understanding of how the "Best" image was selected. 

##### Demonstration Command 

`python owl.py --DO THIS`


#### Input 3D Image Series 
As a bioinformatician, I want to be able to input a series of images directly into the pipeline, so that I can utilise this application to denoise 3D images.

`python owl.py --folder sample`

##### Demonstration Command 

#### Output 3D Image Series 
As a bioinformatician, I want to be able to output a series of images in the same order that they were inputted, so that I can utilise this application to denoise 3D images.

##### Demonstration Command 
`python owl.py --folder sample` 
To view these commands in action a video has been created: 

