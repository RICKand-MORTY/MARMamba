import subprocess

commands = [
    'python train_step.py -learning_rate 0.0002 -num_steps 100000 -train_batch_size 8 -crop_size 256 256 -warm_up True -save_step 1000 -Tmax 1000 -exp_name checkpoint/syn -train_data_dir /root/autodl-tmp/SynDeepLesion -val_data_dir /root/autodl-tmp/SynDeepLesion',
    'python train_step.py -learning_rate 0.0002 -num_steps 300000 -train_batch_size 4 -crop_size 336 336 -warm_up True -save_step 1000 -Tmax 1000 -exp_name checkpoint/syn -train_data_dir /root/autodl-tmp/SynDeepLesion -val_data_dir /root/autodl-tmp/SynDeepLesion -checkpoint ./checkpoint/syn/100000_ckpt',
    'python train_step.py -learning_rate 0.0002 -num_steps 320000 -train_batch_size 2 -crop_size 416 416 -warm_up True -save_step 1000 -Tmax 1000 -exp_name checkpoint/syn -train_data_dir /root/autodl-tmp/SynDeepLesion -val_data_dir /root/autodl-tmp/SynDeepLesion -checkpoint ./checkpoint/syn/300000_ckpt',
    ##'python train_step.py -learning_rate 0.0001 -num_steps 400000 -train_batch_size 2 -crop_size 416 416 -warm_up True -save_step 1000 -Tmax 2000 -exp_name checkpoint/xma -train_data_dir /root/autodl-tmp/SynDeepLesion -val_data_dir /root/autodl-tmp/SynDeepLesion -checkpoint ./checkpoint/syn/350000_ckpt',
    
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