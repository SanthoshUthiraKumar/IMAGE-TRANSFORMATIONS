# EX-4 IMAGE TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import necessary libraries (NumPy, OpenCV, Matplotlib).
<br>

### Step2:
Read an image, convert it to RGB format, and display it using Matplotlib.Define translation parameters 
(e.g., shifting by 100 pixels horizontally and 200 pixels vertically).Perform translation using cv2.warpAffine().
Display the translated image using Matplotlib.
<br>

### Step3:
Obtain the dimensions (rows, cols, dim) of the input image.
Define a scaling matrix M with scaling factors of 1.5 in the x-direction and 1.8 in the y-direction
.Perform perspective transformation using cv2.warpPerspective(), scaling the image by a factor of 1.5 in the x-direction and 1.8 in the y-direction.
Display the scaled image using Matplotlib.
<br>

### Step4:
Define shear matrices M_x and M_y for shearing along the x-axis and y-axis, respectively.
Perform perspective transformation using cv2.warpPerspective() with the shear matrices to shear the image along the x-axis and y-axis.
Display the sheared images along the x-axis and y-axis using Matplotlib.
<br>

### Step5:
Define reflection matrices M_x and M_y for reflection along the x-axis and y-axis, respectively.
Perform perspective transformation using cv2.warpPerspective() with the reflection matrices to reflect the image along the x-axis and y-axis.
Display the reflected images along the x-axis and y-axis using Matplotlib.
<br>

### Step 6 :
Define an angle of rotation in radians (here, 10 degrees).
Construct a rotation matrix M using the sine and cosine of the angle.
Perform perspective transformation using cv2.warpPerspective() with the rotation matrix to rotate the image.Display the rotated image using Matplotlib.
<br>
### Step 7 :
Define a region of interest by specifying the desired range of rows and columns to crop the image (here, from row 100 to row 300 and from column 100 to column 300).
Use array slicing to extract the cropped region from the input image.Display the cropped image using Matplotlib.
<br>


## Program:
```

#### Developed By: Santhosh U
#### Register Number: 212222240092
```
     
### i)Image Translation
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("image.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()
rows,cols,dim=input_image.shape
M = np.float32([[1, 0, 50],[0, 1, 50],[0, 0, 1]])

translated_image = cv2.warpPerspective(input_image, M, (cols, rows))
plt.axis('off')
plt.imshow(translated_image)
plt.show()
```
  
### ii) Image Scaling
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("image.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()

rows, cols, dim = input_image.shape 
M = np.float32([[1.5, 0, 0],[0, 1.8, 0],[0, 0, 1]])

scaled_img=cv2.warpPerspective (input_image, M, (cols*2, rows*2))
plt.imshow(scaled_img)
plt.show()
```

### iii)Image shearing
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("image.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()

M_x = np.float32([[1, 0.5, 0],[0, 1 ,0],[0,0,1]])
M_y =np.float32([[1, 0, 0],[0.5, 1, 0],[0, 0, 1]])

sheared_img_xaxis=cv2.warpPerspective(input_image,M_x, (int(cols*1.5), int(rows *1.5)))
sheared_img_yaxis = cv2.warpPerspective(input_image,M_y,(int(cols*1.5), int(rows*1.5)))

plt.imshow(sheared_img_xaxis)
plt.show()

plt.imshow(sheared_img_yaxis)
plt.show()
```

### iv)Image Reflection
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("image.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()

M_x= np.float32([[1,0, 0],[0, -1, rows],[0, 0, 1]])
M_y =np.float32([[-1, 0, cols],[ 0, 1, 0 ],[ 0, 0, 1 ]])
# Apply a perspective transformation to the image
reflected_img_xaxis=cv2.warpPerspective (input_image, M_x,(int(cols), int(rows)))
reflected_img_yaxis= cv2.warpPerspective (input_image, M_y, (int(cols), int(rows)))

                                         
plt.imshow(reflected_img_xaxis)
plt.show()

plt.imshow(reflected_img_yaxis)
plt.show()
```

   ### v)Image Rotation
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("image.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()

angle=np.radians(10)
M=np.float32([[np.cos(angle),-(np.sin(angle)),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
rotated_img = cv2.warpPerspective(input_image,M,(int(cols),int(rows)))

plt.imshow(rotated_img)
plt.show()
```

### vi)Image Cropping
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("image.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()

cropped_img= input_image[100:300,100:300]

plt.imshow(cropped_img)
plt.show()
```
### Output:
### i)Image Translation
![Output1](https://github.com/SanthoshUthiraKumar/IMAGE-TRANSFORMATIONS/assets/119477975/18528db2-7348-4ec0-bcf5-d8e89bee1e78)

### ii) Image Scaling
![Output2](https://github.com/SanthoshUthiraKumar/IMAGE-TRANSFORMATIONS/assets/119477975/bd096255-3f15-49cf-89bb-4de4ae04a1f9)

### iii)Image shearing
![Output3](https://github.com/SanthoshUthiraKumar/IMAGE-TRANSFORMATIONS/assets/119477975/cd7b3e68-26f4-4e73-9e80-ab30ab6af4e3)

### iv)Image Reflection
![Output4](https://github.com/SanthoshUthiraKumar/IMAGE-TRANSFORMATIONS/assets/119477975/5e19e11e-c8fb-467e-84a1-c12b52d0bcbd)

### v)Image Rotation
![Output5](https://github.com/SanthoshUthiraKumar/IMAGE-TRANSFORMATIONS/assets/119477975/af6338a6-6cfc-46d7-990c-5908b9d2bd24)

### vi)Image Cropping
![Output6](https://github.com/SanthoshUthiraKumar/IMAGE-TRANSFORMATIONS/assets/119477975/1087e2ae-54da-4be6-a1ff-9234be32f006)

### Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
