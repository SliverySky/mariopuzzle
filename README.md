# MarioPuzzle
This is the code for the paper "Experience-Driven PCG via Reinforcement Learning: A Super Mario Bros Study" accepted by the 2021 IEEE Conference on Games.

Please use this bibtex if you use this repository in your work:

````
@inproceedings{shu2021experience,
  title={Experience-Driven PCG via Reinforcement Learning: A Super Mario Bros Study},
  author={Shu, Tianye and Liu, Jialin and Yannakakis, Georgios N.},
  booktitle={2021 IEEE Conference on Games (CoG)},
  pages={accepted},
  year={2021},
  organization={IEEE}
}
````

## Update (June 29th, 2021)

TODOs added on June 29th, 2021
* Add instructions about how to use it.
* Make this repository more clean and readable in the near future.

### Requirements:

```
pytorch (1.7.1)
gym (0.18.0)
java (9)
```



### How to use mariopuzzle game environment:

```sh
cd project/pcg-gym/
pip install -e .
```

You can modify the mariopuzzle.py in /pcg-gym/pcg_gym/envs/ by using new reward function, adding new component...

Now the mariopuzzle contains:

- Generator: /DCGAN
- Repairer: /pcg-gym/pcg_gym/envs/MarioLevelRepairer
- AI  player:  /pcg-gym/pcg_gym/envs/Mario-AI-Framework-master.jar

Use random agent to play mariopuzzle (generate levels):

```python
cd project/pcg-gym/pcg_gym/envs/
python visual_play.py
```

### Train a agent

```sh
cd project/pytorch-a2c-ppo-acktr-gail/
python main.py --env-name "mario_puzzle-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 16 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.1 --save-dir "./trained_models/experiment99/" --recurrent-policy --log-dir "./logs/experiment99/" --num-env-steps 1000000 --experiment 7 --cuda-id 1 
```

Since each environment contains a GAN, a repairer and a tetst agent, multiple GPU (at least four) are needed to run these components. You can decrease the num-processes parameter.

### Generate levels with pre-trained agent

```sh
cd project/data
python generate_level.py --exp 7 # generate levels with pretrained agent 
python online_generate.py --exp 7 # online generate playable levels
```



  

