# Owl
Owl is an automated denoising application utilising ITV Algorithm and Image Registration. 
 
Traditionally, user input is provided to identify a stopping points for extent of denoising. Owl utilises image registration to calculate the optimal level of denoising.  

### How it is calculated: 

When two images undergo im age registration, they are compared and a similarity metric is derived. An image is iteratively denoised, obtaining a similarity metric at each stage. A graph is derived plotting the similarity metric against the lambda value used to denoise the image. The turning point from this the optimal lambda value. 


