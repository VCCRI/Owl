# Owl
![Owl](/images/owl.png)
Owl is an automated denoising application utilising ITV Algorithm and Image Registration. 
 
Traditionally, user input is provided to identify a stopping points for extent of denoising. Owl utilises image registration to calculate the optimal level of denoising.  

### How it is calculated: 

When two images undergo im age registration, they are compared and a similarity metric is derived. An image is iteratively denoised, obtaining a similarity metric at each stage. A graph is derived plotting the similarity metric against the lambda value used to denoise the image. The turning point from this the optimal lambda value. 

## Installation 
### Linux 

1.	Installing Prox_TV 
Detailed installation instructions can be found at:  https://github.com/albarji/proxTV 

sudo apt install cmake  
sudo pip install numpy scipy scikit-image cffi  
sudo apt-get install liblapacke-dev  
sudo pip install prox-tv	 

2.	Installing Simple Elastix 

sudo apt-get python python-dev monodevelop r-base r-base-dev ruby ruby-dev tcl tcl-dev tk tk-dev   
git clone https://github.com/SuperElastix/SimpleElastix   
mkdir build  
cd build  
cmake ../SimpleElastix/SuperBuild   
make -j4  

Note: -j4 means we are compiling SimpleElastix with 4 cores. You need 4GB of free memory per core. This process takes 1 hour on a quad core machine.     
To install the python module into your system go to the following directory:   
{BUILD_DIRECTORY}/SimpleITK-build/Wrapping/Python/Packaging   
and type:    
sudo python setup.py install   

3.	Download owl   
Git clone https://github.com/VCCRI/Owl.git   


### Mac:
1.	Prox_TV    
Detailed installation instructions can be found at:  https://github.com/albarji/proxTV    

sudo easy_install pip     
sudo pip install  tornado nose numpy scipy cffi matplotlib scikit-image    
git clone https://github.com/albarji/proxTV.git    
cd proxTV/src    
git clone https://github.com/vivianlikeyeah/supp-files.git   
cd supp-files   
mv * ../    
cd ../../   
python setup.py install    

2.	SimpleElastix 

sudo pip install cmake    
git clone https://github.com/SuperElastix/SimpleElastix   
mkdir build   
cd build   
cmake ../SimpleElastix/SuperBuild   
make -j4   

Note: -j4 means we are compiling SimpleElastix with 4 cores. You need 4GB of free memory per core. This process takes 1 hour on a quad core machine.    

To install the python module into your system go to the following directory:    
{BUILD_DIRECTORY}/SimpleITK-build/Wrapping/Python/Packaging    
and type:    
sudo python setup.py install    

3.	Download owl     
Git clone https://github.com/VCCRI/Owl.git    

### Windows:    

If you are utilising Windows 10, it is recommended that you enable Bash and then follow the Linux instructions outlined above. A tutorial on enabling bash can be found here: https://www.windowscentral.com/how-install-bash-shell-command-line-windows-10   

Otherwise, multiple alternatives have been outlined below.    
1.	ProxTV     
Detailed installation instructions can be found at:  https://github.com/albarji/proxTV     

python –m pip install numpy scipy cffi matplotlib scikit-image    
git clone https://github.com/albarji/proxTV.git   
cd proxTV/src    
git clone https://github.com/vivianlikeyeah/supp-files.git   
cd supp-files  
move * ../   
cd ../../  
python setup.py install   

Please note, depending on your set-up you may have to modify the above commands.   
For example C:\Python34\python –m pip install …    
2.	SimpleElastix  
Alternative 1:   
You can download SimpleElastix Directly as a whl file at   
http://simpleelastix.github.io/#download  

Then   
pip install SimpleITK-0.9.1rc1.dev163-cp34-cp34m-win_amd64.whl    

Alternative 2:  
Detailed installation instructions can be found at: http://simpleelastix.readthedocs.io/GettingStarted.html
To compile the SuperBuild you need CMake, git and a compiler toolchain. Here, we use the compiler that comes with the free Visual Studio Community 2017 package. https://www.visualstudio.com/downloads/

The steps are outlined below:   

1.	Download CMake, git and code, and setup directories.   
2.	Download and install CMake GUI. Be sure to select Add CMake to the system PATH option.   
3.	Download SimpleElastix into a source folder of your choice   
git clone https://github.com/kaspermarstal/SimpleElastix  
4.	Make a new directory named build    
mkdir build   
5.	Enter the folder     
cd build    
Here we will assume that the build directory and the source directory is in the same folder.    

Compile the project.

1.	Open “Developer Command Prompt for VS2015” (or equivalent depending on your version of Visual Studio)   
2.	Run    
cmake ../SimpleElastix/SuperBuild    
3.	Run    
msbuild /p:configuration=release ALL_BUILD.vcxproj   

To install the python module into your system go to the following directory:     
{BUILD_DIRECTORY}/SimpleITK-build/Wrapping/Python/Packaging   
and type:    
sudo python setup.py install   

A third alternative utilising the visual studio gui is outlined http://simpleelastix.readthedocs.io/GettingStarted.html   


3.	Owl   
Git clone https://github.com/VCCRI/Owl.git   


## Usage 

Try a demonstration run!  
python owl.py --demo 

#### Single Image 

python owl.py --file SampleImage  


#### 3D Stack 

python owl.py --folder SampleFolder    

