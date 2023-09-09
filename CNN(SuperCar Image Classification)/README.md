<h1>CNN - SuperCar Image classification</h1><br>
<h2>Author: Anthony Rodrigues</h2>

<b><pre>
I have divided The Project into Four Parts :   
                                                1)Data Collection - Web Scraping 
                                                2)Image Augmentation  
                                                3)Model Building, Training and Evaluating
                                                4)Model Testing and Pipeline Building
</b></pre>

[I have also created a Web using Flask which lets user upload an image from his local and my model predicts the image which
I will upload later on after adding additional features to it .]

Diving into The Project 

<h3>1)Data Collection - Web Scraping (<a href="https://github.com/Sharkytony/Deep-Learning-                                  Projects/blob/main/CNN(SuperCar%20Image%20Classification)/Image_Classifier_Webscraper.ipynb" target="_blank">Image_Classifier_Webscraper.ipynb</a>)</h3>
The Begin with We have web scraping where I scrape two websites Both containing names of Few Top SuperCar selling Brands
across the World . Searched for the Same Car Brands on a popular website selling SuperCars
with the filter only Search in UAE (for further use in web), and Scraped those too. Now I use random function to randomly
pick two cars of each brand as classes for the Image Classification along with the 
Car model Year. and store the data in CSV file <a href="https://github.com/Sharkytony/Deep-Learning-Projects/blob/main/CNN(SuperCar%20Image%20Classification)/filtered_cars_data.csv" target="_blank">"filtered_cars_data.csv"
</a>. Again imported the csv and Downloaded 20 images of each Car_model in the data using pygoogle_image library which searches on google and downloads images which appear on the Images section.


<h3>2)Image Augmentation (<a href="https://github.com/Sharkytony/Deep-Learning-Projects/blob/main/CNN(SuperCar%20Image%20Classification)/Image_Classifier_Preprocessor.ipynb" target="_blank"> Image_Classifier_Preprocessor.ipynb</a>)</h3>
As 20 images is very less to train a model to be efficient, I have Loaded the images from the Train images folder(Not uploaded here due to large size) and applied Augmentation including Flipping, Cropping,Applying Linear Contrast, GrayScaling,  Blurring, Adding Noise, Shearing, Rotating, Scaling, Horizontal and Vertical movement(translate percent)on each Image upto 8 times (To not fall into overfitting by generating same images multiple times) and saving in the same directory from where the images are imported.

<h3>3)Model Building, Training and Evaluating (<a href="https://github.com/Sharkytony/Deep-Learning-Projects/blob/main/CNN(SuperCar%20Image%20Classification)/Image_Classifier_Model.ipynb" target="_blank">Image_Classifier_Model.ipynb</a>)</h3>
In the Training part, I have first imported the Training images path then listing the labels of each classes from the folder names in Train_images and saving the labels according to their index in the folder as a dict (<a href="https://github.com/Sharkytony/Deep-Learning-Projects/blob/main/CNN(SuperCar%20Image%20Classification)/checkpoint/label_to_index.txt">label_to_index.txt</a>). Created functions for reading the image from its path and another for decoding resizing and scaling the images . Further on Ive Defined and Splitted the Date into train and validation split
Applied the functions created for preprocessing on images in the trainand val data. Used Transfer Learning technique for this Task using InceptionV3 which is moderate in size and has good accuracy. Learning Rate Schedular To prevent Gradient Explosion and Early Stopping for Overfitting has also been applied while training the Model .Run the training for about 15 epochs and saw the model was Undergoing Overfitting condition thus resumed training for about 5 more epochs and The overfitting condition was dismissed and the model improved from 77% to 90% jump also having the training accuracy stable around 90%. Saved the model using tensorflow in  <a href="https://github.com/Sharkytony/Deep-Learning-Projects/blob/main/CNN(SuperCar%20Image%20Classification)/checkpoint/weightings.h5" target="_blank">checkpoint/weightings.h5</a>.


<h3>4)Model Testing and Pipeline Building (<a href="https://github.com/Sharkytony/Deep-Learning-Projects/blob/main/CNN(SuperCar%20Image%20Classification)/Image_Classifier_Tester.ipynb" target="_blank">Image_Classifier_Tester.ipynb</a>)</h3>
Here I have loaded the model and labels first assigned the label names as values and their index as keys. Preprocessing the Images from Test file and predicting those by the model . Created a .py file with all the preprocessing and predicting functions imported the functions in the current file saved in a pipeline and dumped the pipeline using joblib into <a href="https://github.com/Sharkytony/Deep-Learning-Projects/blob/main/CNN(SuperCar%20Image%20Classification)/pred_model.pkl" target="_blank">"pred_model.pkl"</a>.
