# AC-gan
![Image](https://github.com/arshagarwal/AC-gan/blob/master/Results/grid_result_14thin.png)
![Image2](https://github.com/arshagarwal/AC-gan/blob/master/Results/grid_result_240thin.png)

## Quick guide illustrating  the function of the most important files.
0. **models/t_cgan.py** - The generator and discriminator models are declared in this file.
1. **torch_train_cgan.py**- To Train the conditional Gan network. The network is declared in **models/Cgan.py**.
2. **torch_test_cgan.py** - To test the trained conditional Gan network.
3. **Options/cgan_train_options.py** - Command-line options/arguments for train_cgan0.py file.
4. **Options/cgan_test_options.py**- Command-line options/arguments for test_cgan.py file.
5. **slim_dataset**- Dataset consisting of **Train_dataset** and **Test_dataset** for training and testing respectively.
6. **utils/Image_processing** - Code to extract and preprocess the Images.
7. **visualize.py** - Code to visualize the results.
8. **docker** - docker file to containerize and lock the dependencies.

## Steps to run the code using docker container(recommended way):
1. Install the docker cli tools by following the [docker installation steps](https://docs.docker.com/engine/install/ubuntu/).
2. Clone the repository by running `!git clone https://github.com/arshagarwal/AC-gan.git` .
3. Run the command `docker build -f docker -t acgan:latest .` to build the docker container. 
4. Run the command `docker run -it acgan:latest` to start the docker container.
5. Run the command `python torch_train_cgan.py --data_path slim_dataset/Test_dataset`. 
**Alternatively to train on custom dataset replace the `slim_dataset/Train_dataset` string with the path of your custom dataset.** 
6. Run the command `!python visualize.py  2` to visualize results for **2** categories. 
### Note: After running this command the results will be stored in **grid_results_n_epochs** directory. 


## Steps to run the code on Google colab
0. Clone the code by running the command `!git clone https://github.com/arshagarwal/AC-gan.git` . 
   **Replace the username and password with your github username and password respectively.**
1. Run the command `cd Ac-gan` to navigate to the **Ac-gan_gan** directory.
2. Run the command `!bash import_dataset.sh` to import the **Big slim dataset**. 
3. Run the command `!python torch_train_cgan.py --data_path Big_slim` to train on the **Big_slim_dataset**. 
**Alternatively to train on custom dataset replace the `slim_dataset/Train_dataset` string with the path of your custom dataset.**  
  For further options such as **number of epochs, batch_size etc** refer **Options/cgan_train_options.py**
4. Run the command `!python visualize.py  2` to visualize results for **2** categories. 
### Note: After running this command the results will be stored in **grid_results_n_epochs** directory. 

## Steps to run locally
0. Clone the code by running the command `git clone https://github.com/arshagarwal/AC-gan.git` . 
   **Replace the username and password with your github username and password respectively.**
1. Navigate into the directory where the repository is downloaded.
2. Run the command `!bash import_dataset.sh` to import the **Big slim dataset**.
3. Run the command `python torch_train_cgan.py --data_path Big_slim` to train on the **Big_slim_dataset**. 
**Alternatively to train on custom dataset replace the `slim_dataset/Train_dataset` string with the path of your custom dataset.**  
  For further options such as **number of epochs, batch_size etc** refer **Options/cgan_train_options.py**
4. Run the command `python visualize.py  2` to visualize the results for **2** categories. 
### Note: After running this command the results will be stored in **grid_results_n_epochs** directory.

## Steps for continuing training from Latest Checkpoint:
1. To train from previous checkpoint after training with the code add the `--continue_train` flag in the train command.
2. For ex you trained for say 5 epochs with the code 
`! python torch_train_cgan.py --epochs 5 --batch_size 50 --data_path Big_slim`
3. To continue training from 6th epoch run the command `! python torch_train_cgan.py --epochs 5 --batch_size 50 --data_path Big_slim --continue_train`, and this should start training from the last checkpoint.
