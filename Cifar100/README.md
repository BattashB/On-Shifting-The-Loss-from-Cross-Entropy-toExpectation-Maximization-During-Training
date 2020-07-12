## Cifar 100:
The models were trained on one gpu, data parallel of course can effect the results.
### Examples:
##### Train base model:
```
CUDA_VISIBLE_DEVICES=0 python3 trainer.py --arch <resnet20/resnet32/resnet44> --save-dir <dest_dir> --data_path <data_path>
```
##### Train CE+EL:
```
CUDA_VISIBLE_DEVICES=0 python3 trainer.py --arch <resnet20/resnet32/resnet44> --save-dir <dest_dir> --accloss True --data_path <data_path>
```
##### Train CE+EL, F = {0,0.5}, C=1.9:
```
CUDA_VISIBLE_DEVICES=0 python3 trainer.py --arch <resnet20/resnet32/resnet44>  --save-dir <dest_dir> --accloss True --F "0,0.5" --hf 1.9 --data_path <data_path>
```
##### Train CE+EL, F = {0-0.5}, C=1:
```
CUDA_VISIBLE_DEVICES=0 python3 trainer.py --arch <resnet20/resnet32/resnet44>  --save-dir <dest_dir> --accloss True --F "0,0.1,0.2,0.3,0.4,0.5" --hf 1 --data_path <data_path>
```

In order to run all the configuration with multiple seeds, you can use:
```
CUDA_VISIBLE_DEVICES=0 python3 run_model.py  <resnet20/resnet32/resnet44>
```
