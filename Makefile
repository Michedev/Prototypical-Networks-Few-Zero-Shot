train-paper-omniglot:
	python train.py --dataset=omniglot --classes=60 -s=1 -q=5 --device=cuda:0 --batch-size=64 --lr=1e-3 --lr-decay=True --lr-decay-gamma=0.5 --lr-decay-steps=3000
	python train.py --dataset=omniglot --classes=60 -s=5 -q=5 --device=cuda:0 --batch-size=16 --lr=1e-3 --lr-decay=True --lr-decay-gamma=0.5 --lr-decay-steps=3000
train-paper-miniimagenet:
	python train.py --dataset=miniimagenet --classes=30 -s=1 -q=15 --lr=1e-3 --lr-decay=True --lr-decay-gamma=0.5 --lr-decay-steps=3000 --device=cuda:0 --batch-size=4 --early-stop=True --early-stop-patience=5
	python train.py --dataset=miniimagenet --classes=20 -s=5 -q=15 --lr=1e-3 --lr-decay=True --lr-decay-gamma=0.5 --lr-decay-steps=3000 --device=cuda:0 --batch-size=4 --early-stop=True --early-stop-patience=5
train-paper-zero-shot-cub:
	python train.py --dataset=cub -s=0 --classes=50 -q=10 \
                    --lr=1e-4 --weight-decay=1e-6 --seed=13 \
                    --image-features=1024 --metadata-features=312 \
                    --device=cuda:0 --batch-size=16 --lr-decay=False \
                    --lr-decay-steps=100 --lr-decay-gamma=0.5 \
                    --early-stop=True --early-stop-patience=2 \
                    --early-stop-metric=loss
