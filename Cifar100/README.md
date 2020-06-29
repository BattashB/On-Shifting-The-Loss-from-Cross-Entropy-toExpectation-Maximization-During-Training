## Cifar 100:

### Examples:
##### Train base model:
```
python3 trainer.py --arch <resnet20/resnet32/resnet44> --save-dir <dest_dir> --data_path <data_path>
```
##### Train CE+EL:
```
python3 trainer.py --arch <resnet20/resnet32/resnet44> --save-dir <dest_dir> --accloss True --data_path <data_path>
```
##### Train CE+EL, F = {0,0.5}, C=1.9:
```
python3 trainer.py --arch <resnet20/resnet32/resnet44>  --save-dir <dest_dir> --accloss True --F "0,0.5" --hf 1.9 --data_path <data_path>
```
##### Train CE+EL, F = {0-0.5}, C=1:
```
python3 trainer.py --arch <resnet20/resnet32/resnet44>  --save-dir <dest_dir> --accloss True --F "0,0.1,0.2,0.3,0.4,0.5" --hf 1 --data_path <data_path>
```

In order to run all the configuration with multiple seeds, you can use:
```
python3 run_model.py  <resnet20/resnet32/resnet44>
```
