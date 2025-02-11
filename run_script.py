import subprocess
import os  

train_data_dir = input("input your training dataset path:")
if train_data_dir is None or train_data_dir == '':
    print("invalid path!")
    exit()
if not train_data_dir.endswith('/'):  
    train_data_dir = os.path.join(train_data_dir, '') 

val_data_dir = input("input your testing dataset path:")
if val_data_dir is None or val_data_dir == '':
    print("invalid path!")
    exit()
if not val_data_dir.endswith('/'):  
    val_data_dir = os.path.join(val_data_dir, '') 

save_place = input("input the saving path for checkpoints:")
if save_place is None or val_data_dir == '':
    print("invalid path!")
    exit()
if not save_place.endswith('/'):  
    save_place = os.path.join(save_place, '')


commands = [
    f'python train_step.py -train_data_dir {train_data_dir} -val_data_dir {val_data_dir} -learning_rate 0.0003 -num_steps 50000 -train_batch_size 8 -crop_size 128 128 -save_step 2000 -exp_name {save_place}',
    f'python train_step.py -train_data_dir {train_data_dir} -val_data_dir {val_data_dir} -learning_rate 0.0002 -num_steps 120000 -train_batch_size 8 -crop_size 128 128 -save_step 2000 -exp_name {save_place} -checkpoint {save_place}50000_ckpt',
    f'python train_step.py -train_data_dir {train_data_dir} -val_data_dir {val_data_dir} -learning_rate 0.0001 -num_steps 220000 -train_batch_size 5 -crop_size 144 144 -save_step 2000 -exp_name {save_place} -checkpoint {save_place}120000_ckpt',
    f'python train_step.py -train_data_dir {train_data_dir} -val_data_dir {val_data_dir} -learning_rate 0.000075 -num_steps 320000 -train_batch_size 4 -crop_size 160 160 -save_step 1000 -exp_name {save_place} -checkpoint {save_place}220000_ckpt',
    f'python train_step.py -train_data_dir {train_data_dir} -val_data_dir {val_data_dir} -learning_rate 0.00005 -num_steps 410000 -train_batch_size 4 -crop_size 160 160 -save_step 1000 -exp_name {save_place} -checkpoint {save_place}320000_ckpt',
    f'python train_step.py -train_data_dir {train_data_dir} -val_data_dir {val_data_dir} -learning_rate 0.000025 -num_steps 500000 -train_batch_size 3 -crop_size 192 192 -save_step 1000 -exp_name {save_place} -checkpoint {save_place}410000_ckpt',
    f'python train_step.py -train_data_dir {train_data_dir} -val_data_dir {val_data_dir} -learning_rate 0.00001 -num_steps 580000 -train_batch_size 3 -crop_size 192 192 -save_step 1000 -exp_name {save_place} -checkpoint {save_place}500000_ckpt',
    f'python train_step.py -train_data_dir {train_data_dir} -val_data_dir {val_data_dir} -learning_rate 0.0000075 -num_steps 660000 -train_batch_size 2 -crop_size 224 224 -save_step 1000 -exp_name {save_place} -checkpoint {save_place}580000_ckpt',
    f'python train_step.py -train_data_dir {train_data_dir} -val_data_dir {val_data_dir} -learning_rate 0.000005 -num_steps 680000 -train_batch_size 1 -crop_size 256 256 -save_step 1000 -exp_name {save_place} -checkpoint {save_place}660000_ckpt',
    f'python train_step.py -train_data_dir {train_data_dir} -val_data_dir {val_data_dir} -learning_rate 0.00000025 -num_steps 700000 -train_batch_size 1 -crop_size 320 320 -save_step 1000 -exp_name {save_place} -checkpoint {save_place}680000_ckpt',
    f'python train_step.py -train_data_dir {train_data_dir} -val_data_dir {val_data_dir} -learning_rate 0.0000001 -num_steps 720000 -train_batch_size 1 -crop_size 320 320 -save_step 1000 -exp_name {save_place} -checkpoint {save_place}700000_ckpt',
    f'python train_step.py -train_data_dir {train_data_dir} -val_data_dir {val_data_dir} -learning_rate 0.00002 -num_steps 800000 -train_batch_size 1 -crop_size 320 320 -save_step 1000 -exp_name {save_place} -checkpoint {save_place}720000_ckpt',
]

# execute each command
for cmd in commands:
    try:
        print(f"now running command: {cmd}")
        result = subprocess.run(cmd, check=True, shell=True)
        if result.returncode != 0:
            print(f"command '{cmd}' had error. error code: {result.returncode}")
            break
        else:
            print(f"command '{cmd}' finish!")
    except subprocess.CalledProcessError as e:
        print(f"command '{cmd}' had error while running. error exception: {e}")
        break