clean-logs:
	rm -r lightning_logs/
clean-models:
	rm -r model_weights

train-miniimagenet:
	python3 train.py --dataset='miniimagenet' --train-n=30 --val-n=5 --test-n=5 --n-s=1 --n-q=15 --trainsize=1_000 --valsize=64 --testsize=64 --device='cuda:1' --batch-size=4

train-miniimagenet-log:
	python3 train.py --dataset='miniimagenet' --train-n=30 --val-n=5 --test-n=5 --n-s=1 --n-q=15 --trainsize=1_000 --valsize=64 --testsize=64 --device='cuda:1' --batch-size=4 > train-log.txt 2> train-err.txt
