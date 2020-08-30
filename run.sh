!git clone https://$1:$2@github.com
!cd C_slim_gan
!python train_cgan0.py --data_path $3 
!python test_cgan.py 2