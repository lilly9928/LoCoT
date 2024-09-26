# Basic settings
* Set the number of GPUs to use: <br/>`NUM_GPUS=num_of_gpus`
* Set the rank of the current node (in multi-node training): <br/>`NODE_RANK=rank_of_node`
* Set the number of nodes to use (in multi-node training): <br/>`NUM_NODES=num_of_nodes`
* Set the address of the master node: <br/>`MASTER_ADDR=addr_of_master`
* Set the port of master: <br/>`MASTER_PORT=port_of_master`
* Set the GPUs to use: <br/>`CUDA_VISIBLE_DEVICES=gpu_devices`

# Training
To train the model, run the following command:
```
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=${NUM_NODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    main.py \
    --name={EXPERIMENT_NAME}\
    --data={DATA_NAME}\
    --data_path={DATA_PATH}\
    --split={SPLIT}         #   {SPLIT} = {'train', 'test', 'val', 'testdev'}
```
