# Targeted Adversarial Example Generator

![1](https://user-images.githubusercontent.com/48395704/91305142-8f8b0080-e7e5-11ea-9913-7bae83dd9646.JPG)

Create an adversarial example for targeted misclassification using iterative least likely class method.  
The target neural network is Keras MobilenetV2 ImageNet.  


# Instructions 
### Requirements   
- Python 3+
- tensorflow
- numpy
- matplotlib

### Usage
Input image name : **input.jpg**  
Output image name : **output.jpg**  
<code>main.py</code> and **input.jpg** must be in the same path.  
  
Run <code>main.py</code>  
While main is running, original image, noise, adversarial example are displayed in screen.  
  
You can change a target of misclassification by changing <code>target_index value</code>
