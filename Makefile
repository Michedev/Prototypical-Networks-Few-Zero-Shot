train-paper-omniglot:
	python train.py --dataset=omniglot --classes=60 -s=1 -q=5 --device=cuda:0 --eval-steps=10 --batch-size=64
	python train.py --dataset=omniglot --classes=60 -s=5 -q=5 --device=cuda:0 --eval-steps=10 --batch-size=64
train-paper-miniimagenet:
	python train.py --dataset=miniimagenet --classes=30 -s=1 -q=15
	python train.py --dataset=miniimagenet --classes=20 -s=5 -q=15
