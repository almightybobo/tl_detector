
# Dataset
https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0&fbclid=IwAR3ej7pxj0__2uAWymsGzLT2YYjqaIZwyupVx1AhZynhTCNXuMnqG8P6vmo

# How to
1. download the above dataset, unzip, put in this working directory as udacity-traffic-light-dataset
2. python data_preprocess.py udacity-traffic-light-dataset/sim_data.txt udacity-traffic-light-dataset/udacity_testarea_rgb udacity-traffic-light-dataset/sim_proc udacity-traffic-light-dataset/test
- the processed meta info would be sim_data.txt
- the processed images would be in the sim_proc/
- the test results would be in the test/
3. python main.py -r -e 1 -d udacity-traffic-light-dataset/sim_data.txt -l logs
- -r means training phase
- -e 1 means training for 1 epoch
- -d to specify the path to <data.txt>
- -l logs means saved model would be in logs/
4. python main.py -p -c <path_to_ckpt, excluding file extension>
- -p means predicting phase
- -c means restore from the specified checkpoint

