gpu_id=$1
shift
CUDA_VISIBLE_DEVICES=$gpu_id python train_rnn.py --network_type pc "$@"