# A Novel CNet-assisted Evolutionary Level Repairer  
```T. Shu, Z. Wang, J. Liu and X. Yao, "A Novel CNet-assisted Evolutionary Level Repairer and Its Applications to Super Mario Bros," 2020 IEEE Congress on Evolutionary Computation (CEC), Glasgow, United Kingdom, 2020, pp. 1-10, doi: 10.1109/CEC48606.2020.9185538.```

## Brief introduction
Applying latent variable evolution to game level design has become more and more popular as little human expert knowledge is required. However, defective levels with illegal patterns may be generated due to the violation of constraints for level design. A traditional way of repairing the defective levels is implementing specific rule-based repairers to patch the flaw. However, implementing these constraints is sometimes complex and not straightforward. An autonomous level repairer which is capable of learning the constraints is needed. In this project, we consider tile-based game level design. We propose a novel approach, CNet, to learn the probability distribution of tiles giving its surrounding tiles on a set of real levels, and then detect the illegal tiles in the generated new levels. Then, an evolutionary repairer is designed to search for optimal replacement schemes equipped with a novel search space being constructed with the help of CNet and a novel heuristic function. The proposed approaches are proved to be effective in our case study of repairing GAN-generated and artificially destroyed levels of Super Mario Bros. game. Our CNet-assisted evolutionary repairer can also be easily applied to other games of which the levels can be represented by a matrix of elements or tiles.  

For more information, please see the following paper. This paper should also be cited if code from this project is used in any way:

```
@INPROCEEDINGS{shu2020novel,
  title={A Novel CNet-assisted Evolutionary Level Repairer and Its Applications to Super Mario Bros},
  author={Shu, Tianye and Wang, Ziqi and Liu, Jialin and Yao, Xin},
  booktitle={2020 IEEE Congress on Evolutionary Computation (CEC)}, 
  doi={10.1109/CEC48606.2020.9185538},
  pages={1-10},
  year={2020}
}
```

You can also find the pdf of this paper for free on ArXiv: https://arxiv.org/abs/2005.06148

## Requirements

Requirments to run this project and the tested version are as follow:

* python (3.7.6)

* numpy (1.18.1)

* pytorch (1.5.0)

* pygame (2.0.1)

* matplotlib (3.3.2)


## Project structure
* **Assets/Tiles**: contains resources to render the level
* **CNET** folder: contains codes about CNet	* **CNET** folder: contains codes about CNet
* **GA** folder: repairs defective levels by Genetic Algorithm	* **GA** folder: repairs defective levels by Genetic Algorithm
* **LevelGenerator** folder: generates defective levels (from GAN or random destroyed)	* **LevelGenerator** folder: generates defective levels (from GAN or random destroyed)
* **LevelText** folder: contains mario level text  	* **LevelText** folder: contains mario level text  
* **utiles** folder: contains commonly used methods	* **utiles** folder: contains commonly used methods
* **root.py**: use to get root path	* **root.py**: use to get root path

## How to use
### Quick start without re-training
The generated data and trained model is given in this project. You can run the repairer and visualise the results directly by the following steps.
* **Run GA to repair level**: Run ***GA/run.py*** to repair a defective level. The best invidivual at each epoch will be saved as an image in "*GA/result*" folder. In addition, you can run "*GA/clear.py*" to clean up the old results.

```python GA/clear.py```

```python GA/run.py```

* **Result Visulization:** Run ***draw_graph.py*** to draw the graph from repair results. Run ***evaluate.py*** to see how many true(wrong) tiles was changed to true(wrong) tiles after repair. What's more, you can run "render.py" to see the visuliized repair progress.

```python GA/draw_graph.py```

```python GA/evaluate.py```

```python GA/render.py```

### Start from the very beginning (data generation, model training)
* **Generate Data for CNet**: 

```python CNet/data/generate.py```

The generated data will be located in the folder ***./CNet/data/***.

* **Train/Test CNet**: Run ***CNet/model.py*** to train CNet and run ***CNet/test.py*** to test the model. Please make sure the data is generated already.

```python CNet/model.py```

```python CNet/test.py```

* **Generate levels to repair**: You can run ***RandomDestroyed/generate.py*** to generate random destroyed levels for the further test. As we mension in the paper, GAN will generate broken pipes sometimes. We prepared a trained GAN model in ***LevelGeneratir/GAN/generator.pth***. You can run ***LevelGenerator/generate_level.py*** to generate some levels by GAN to see how many defective level will it generate.  

```python LevelGenerator/RandomDestroyed/generate.py```

```python LevelGenerator/GAN/generate_level.py```

* **Run GA to repair level**: Run ***GA/run.py*** to repair a defective level. The best invidivual at each epoch will be saved as an image in "*GA/result*" folder. In addition, you can run "*GA/clear.py*" to clean up the old results.

```python GA/clear.py```

```python GA/run.py```

* **Result Visulization:** Run ***draw_graph.py*** to draw the graph from repair results. Run ***evaluate.py*** to see how many true(wrong) tiles was changed to true(wrong) tiles after repair. What's more, you can run "render.py" to see the visuliized repair progress.

```python GA/draw_graph.py```

```python GA/evaluate.py```

```python GA/render.py```



