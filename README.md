# README

## Installation

This coordinate mapping requires the following dependencies:

```bash
pytorch=1.6.0
opencv=3.4.2
scikit-image=0.16.2
```



Set up the ESAC extensions:

```bash
coordinate_mapping> cd src/esac
esac> python setup.py install
```



Download the 7Scenes dataset:

```bash
coordinate_mapping> cd datasets
datasets> python download.py
```



## Coordinate Mapping Test

For example, test the mapping from chess to chess scene:

```bash
coordinate_mapping> cd envs/chess_chess
chess_chess> CUDA_VISIBLE_DEVICES=0 python ../../src/test.py -apd chess --tree chess
```





## Set Up a Mapping

First, create a directory under `envs`, e.g. `my_dir`.

Next, train the ESAC pipeline:

```bash
my_dir> CUDA_VISIBLE_DEVICES=0 python ../../src/train_gating.py -apd [src_scene]
my_dir> CUDA_VISIBLE_DEVICES=0 python ../../src/train_expert.py -apd [src_scene]
my_dir> CUDA_VISIBLE_DEVICES=0 python ../../src/train_esac.py -apd [src_scene]
```

which typically takes 1-2 days.



Finally, set up the kd-tree:

```bash
my_dir> CUDA_VISIBLE_DEVICES=0 python ../../src/train_octree.py -apd [src_scene] --tree [target_scene]
```

which typically takes a few seconds.





