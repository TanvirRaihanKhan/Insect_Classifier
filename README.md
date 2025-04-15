# Insect_Classifier
An image classification model that includes data collection,cleaning,model training,deployment and API integration. <br/>
The model can recognize and classify 12 different types of Insects found in Crop fields that are harmful to Crops and Farmers <br/>
They are following: <br/>
1. Africanized Bees
2. Aphids
3. Armyworms
4. Stink Bugs
5. Potato Beetles
6. Cabbage Loopers
7. Corn Borers
8. Corn Earworms
9. Fruit Flies
10. Thrips
11. Tomato Hornworms
12. Corn Rootworms

# Dataset preparation
**Data Collection:** Downloaded from DuckDuckGo using term name <br/>
**Dataloader:** Used fastai DataBlock API to set uo the DataLoader <br/>
**Data Augmentation:** fastai provides data augmentation which operates in GPU <br/>
Details can be found in 'Notebooks\Insect_Classifier_Data_Collection.ipynb'

# Training & Data Cleaning
**Training:** Fine-tuned a Resnet34 model, Resnet50 model, VGG-16 and DenseNet-121 model 3 times with 5 epochs, 4 epochs and 2 epochs respectively. The DenseNet-121 model got the highest accuracy with ~87% accuracy
**Data Cleaning:** THis part took the highest time. Since I collected data from browser, there were many noises. Also there were images that contained misinformation like just Crop images, Insect logos, Animated image etc.. I cleaned and updated data using fastai ImageClassifierCleaner. I cleaned data each time after training or finetuning, except for the last time which was the final iteration of the model. <br/>

# Model deployment
I deployed model to HuggingFace Spaces Gradio App. The deployment canbe found in 'deployment' folder or [here](https://huggingface.co/spaces/Tanvirtrk/Crop_Insects_Classifier). <br/>
<img src='Deployment\Hugging_face_demo.PNG' width='800' height='400'>

# API Integration with Github Pages
The deployed model API is integrated into [here]('tanvirraihankhan.github.io/Jersey_Recognizer/') into Github Pages website. Implementation and other details can be found in 'docs' folder. 
