# Targeted Adversarial Example Generator

![1](https://user-images.githubusercontent.com/48395704/91305142-8f8b0080-e7e5-11ea-9913-7bae83dd9646.JPG)

Create an adversarial example for targeted misclassification using iterative least likely class method  
This code is for Keras MobilenetV2 ImageNet and written in Python


# Instructions 
### Requirements   
- Python 3+
- tensorflow
- numpy
- matplotlib

### Usage
Save the input image under the name **input.jpg**  
The output image will be named **output.jpg**  
<code>main.py</code> and **input.jpg** must be in the same path  
  
Run <code>main.py</code>  
While main is running, original image, noise, adversarial example are displayed in screen
If the run is finished, you can see **output.jpg**  
  
You can change a target of misclassification by changing target_index value
