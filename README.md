# Transformer-Based Deep Survival Analysis
The PyTorch implementation of the paper: [Transformer-Based Deep Survival Analysis](http://proceedings.mlr.press/v146/hu21a/hu21a.pdf).

# Run

This is an example command to train this model

```
python sa_transformer.py --lr 0.0001 --max_time=160 --num_heads=4 --d_model=512 --d_ff=2048 --N=4 --data_parallel --train_batch_size=64 --coeff=0.1 --coeff2=0.1
```
