# DLProject
## How to Setup ?
1. Clone the repository
2. Move to the directory where the repository is cloned
3. Make an environment and install the requirements.txt file

## Structure of the repository
    - experiments
        - Contains the experiments that are run
            - models : Contains the models that are run (Training Models)
            - helpers : Contains the helper functions that are used in the experiments (Training Functions)
            - optimizers : Contains the self defined optimizers that are used in the experiments (Training Optimizers)
    - poc 
        - Contains the different POCs done on different testing functions for single objective optimization (https://en.wikipedia.org/wiki/Test_functions_for_optimization)
        - test1.ipynb : **Final optimisation** rule that is used and tested on the test functions
        - other notebooks : Contains the different optimization rules that are experimented for testing.

    - reports 
        Contains the mid term reports and final report of the project
    
## How to run the experiments ?
1. To run the models move to the models directory inside the experiments directory
2. Run the model that you want to run according to the name of the ipynb file (For example cifar10_adam_cnn.ipynb contains the cnn model trained on cifar10 dataset using our experminetal optimizers and other standard optimizers)


## Experiments Directory Description
Helpers : 
    Contains the helper files that are used in the experiments (Training Functions)
    1. train_all.py : Helps to run all the optmisers on the model without scheduling of learning rate
    2. train_all_2.py : Helps to run all the optimisers on the model with scheduling of learning rate
    3. train_GAN.py : Helps to run the GAN model with the optimisers
    4. trainer.py : Helps to make your own custom trainer for the model

Models :
    Contains the models that are run (Training Models)
    1. cifar10_adam_cnn.ipynb : Contains the cnn model trained on cifar10 dataset using our experminetal optimizers and other standard optimizers
    2. cifar10_GAN.ipynb : Contains the GAN model trained on cifar10 dataset using our experminetal optimizers and other standard optimizers
    3. cifar10_resnet.ipynb : Contains the resnet18 model trained on cifar10 dataset using our experminetal optimizers and other standard optimizers
    4. imagenet_resnet18.ipynb : Contains the resnet18 model trained on imagenet dataset using our experminetal optimizers and other standard optimizers without scheduling of learning rate
    5. imagenet_resnet18_scheduled.ipynb : Contains the resnet18 model trained on imagenet dataset using our experminetal optimizers and other standard optimizers with scheduling of learning rate
    6. mnist_adam_cnn.ipynb : Contains the cnn model trained on mnist dataset using our experminetal optimizers and other standard optimizers

Optimizers :
    Contains the self defined optimizers that were used in the intital experiments (Training Optimizers)
    1. customAdam.py : Contains the custom Adam optimizer same as the original Adam optimizer
    2. customAdam2.py : Contains the custom Adam optimizer with the modification as changing of betas acording to certain rules mentioned in the report
    3. customAdam3.py : Contains the custom Adam optimizer with the modification as incorporating the third order moment in the optimizer along with the first and second order moments
    4. customAdam4.py : Contains the custom Adam optimizer with the modification as replacing the second order moment with third order moement in the optimizer 
