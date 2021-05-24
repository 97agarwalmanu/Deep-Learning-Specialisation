# Deep Learning Specialization on Coursera (offered by deeplearning.ai)

Programming assignments from all courses in the Coursera [Deep Learning specialization](https://www.coursera.org/specializations/deep-learning) offered by `deeplearning.ai`.

Instructor: [Andrew Ng](http://www.andrewng.org/)

## Notes

## Cloning Instructions

1. `git-lfs` is used to handle large dataset files in this repo. As such, please make sure `git-lfs` is installed before cloning this repo. 
2. Steps to install `git-lfs`:
`curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash`
`sudo apt-get install git-lfs`
3. Clone repo:
`git clone <repo_path>`
4. Download  a pre-trained VGG-19 dataset and extract the zip'd pre-trained models and datasets that are needed for all the assignments.
5. Create a custom conda environment with python = 3.6 `conda create -n myenv python=3.6`
6. Activate new conda environment using `conda activate myenv`
7. Install Tensorflow version = 1.15 as these asignments use older version of tensorflow using `conda install tensorflow=1.15`
8. After using the environment deactivate environment using `conda deactivate`

Note that if you `git clone`'d before installing `git-lfs` (which downloaded only pointers to lfs files), install `git-lfs` and then run `git lfs pull`.

## Credits

This repo contains my work for this specialization. The code base is taken from the [Deep Learning Specialization on Coursera](https://www.coursera.org/specializations/deep-learning), unless specified otherwise.

## Programming Assignments

### Course 1: Neural Networks and Deep Learning

  - [Week 2 - PA 1 - Python Basics with Numpy](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Neural%20Networks%20and%20Deep%20Learning/Week%202/Python%20Basics%20with%20Numpy/Python_Basics_With_Numpy_v3a.ipynb)
  - [Week 2 - PA 2 - Logistic Regression with a Neural Network mindset](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Neural%20Networks%20and%20Deep%20Learning/Week%202/Logistic%20Regression%20as%20a%20Neural%20Network/Logistic_Regression_with_a_Neural_Network_mindset_v6a.ipynb)
  - [Week 3 - PA 3 - Planar data classification with one hidden layer](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Neural%20Networks%20and%20Deep%20Learning/Week%203/Planar%20data%20classification%20with%20one%20hidden%20layer/Planar_data_classification_with_onehidden_layer_v6c.ipynb)
  - [Week 4 - PA 4 - Building your Deep Neural Network: Step by Step](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Neural%20Networks%20and%20Deep%20Learning/Week%204/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/Building_your_Deep_Neural_Network_Step_by_Step_v8a.ipynb)
  - [Week 4 - PA 5 - Deep Neural Network for Image Classification: Application](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Neural%20Networks%20and%20Deep%20Learning/Week%204/Deep%20Neural%20Network%20Application_%20Image%20Classification/Deep%20Neural%20Network%20-%20Application%20v8.ipynb)

### Course 2: Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

  - [Week 1 - PA 1 - Initialization](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Week%201/Initialization/Initialization.ipynb)
  - [Week 1 - PA 2 - Regularization](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Week%201/Regularization/Regularization_v2a.ipynb)
  - [Week 1 - PA 3 - Gradient Checking](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Week%201/Gradient%20Checking/Gradient%20Checking%20v1.ipynb)
  - [Week 2 - PA 4 - Optimization Methods](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Week%202/Optimization_methods_v1b.ipynb)
  - [Week 3 - PA 5 - TensorFlow Tutorial](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Week%203/TensorFlow_Tutorial_v3b.ipynb)

### Course 3: Structuring Machine Learning Projects

  - There are no PAs for this course. But this course comes with very interesting case study quizzes.
  
### Course 4: Convolutional Neural Networks

  - [Week 1 - PA 1 - Convolutional Model: step by step](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Convolutional%20Neural%20Networks/Week%201/Convolution_model_Step_by_Step_v2a.ipynb)
  - [Week 1 - PA 2 - Convolutional Neural Networks: Application](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Convolutional%20Neural%20Networks/Week%201/Convolution_model_Application_v1a.ipynb)
  - [Week 2 - PA 1 - Keras - Tutorial - Happy House](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Convolutional%20Neural%20Networks/Week%202/KerasTutorial/Keras_Tutorial_v2a.ipynb)
  - [Week 2 - PA 2 - Residual Networks](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Convolutional%20Neural%20Networks/Week%202/ResNets/Residual_Networks_v2a.ipynb)
  - [Week 3 - PA 1 - Car detection with YOLO for Autonomous Driving](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Convolutional%20Neural%20Networks/Week%203/Car%20detection%20for%20Autonomous%20Driving/Autonomous_driving_application_Car_detection_v3a.ipynb)
  - [Week 4 - PA 1 - Art Generation with Neural Style Transfer](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Convolutional%20Neural%20Networks/Week%204/Neural%20Style%20Transfer/Art_Generation_with_Neural_Style_Transfer_v3a.ipynb)    
  - [Week 4 - PA 2 - Face Recognition](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Convolutional%20Neural%20Networks/Week%204/Face%20Recognition/Face_Recognition_v3a.ipynb)
  
### Course 5: Sequence Models

  - [Week 1 - PA 1 - Building a Recurrent Neural Network - Step by Step](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Sequence%20Models/Week%201/Building%20a%20Recurrent%20Neural%20Network%20-%20Step%20by%20Step/Building_a_Recurrent_Neural_Network_Step_by_Step_v3a.ipynb)
  - [Week 1 - PA 2 - Dinosaur Land -- Character-level Language Modeling](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Sequence%20Models/Week%201/Dinosaur%20Island%20--%20Character-level%20language%20model/Dinosaurus_Island_Character_level_language_model_final_v3a.ipynb)
  - [Week 1 - PA 3 - Jazz improvisation with LSTM](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Sequence%20Models/Week%201/Jazz%20improvisation%20with%20LSTM/Improvise_a_Jazz_Solo_with_an_LSTM_Network_v3a.ipynb)  
  - [Week 2 - PA 1 - Word Vector Representation and Debiasing](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Sequence%20Models/Week%202/Word%20Vector%20Representation/Operations_on_word_vectors_v2a.ipynb)  
  - [Week 2 - PA 2 - Emojify!](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Sequence%20Models/Week%202/Emojify/Emojify_v2a.ipynb)  
  - [Week 3 - PA 1 - Neural Machine Translation with Attention](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Sequence%20Models/Week%203/Machine%20Translation/Neural_machine_translation_with_attention_v4a.ipynb)  
  - [Week 3 - PA 2 - Trigger Word Detection](https://github.com/97agarwalmanu/Deep-Learning-Specialisation/blob/master/Sequence%20Models/Week%203/Trigger%20word%20detection/Trigger_word_detection_v1a.ipynb)   

## Disclaimer

I recognize the hard time people spend on building intuition, understanding new concepts and debugging assignments. The solutions uploaded here are **only for reference**. They are meant to unblock you if you get stuck somewhere. I hope you don't copy any part of the code as-is (the programming assignments are fairly easy if you read the instructions carefully). Similarly, try out the quizzes yourself before you refer to the quiz solutions. This course is the most straight-forward deep learning course I have ever taken, with fabulous course content and structure. It's a treasure by the deeplearning.ai team.
