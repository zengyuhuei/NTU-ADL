# ADL HW3


## Installation
Type the following command to install OpenAI Gym Atari environment.

`$ pip3 install opencv-python gym gym[atari]`

## How to run :
training policy gradient:
* `$ python3 main.py --train_pg`

testing policy gradient:
* `$ python3 test.py --test_pg`

training DQN:
* `$ python3 main.py --train_dqn`

testing DQN:
* `$ python3 test.py --test_dqn`

plot the learning curve of policy gradient:
* `$ python3 pg_plot_learning_curve.py `

plot the learning curve of DQN:
* `$ python3 dqn_plot_learning_curve.py `

plot the different settings of the target network update frequency:
frequency = [500, 1000, 1500, 2000]
* `$ python3 plot_compare.py `

plot the improvement of policy gradient:
* `$ python3 plot_pg_cp.py `

plot the improvement of DQN:
* `$ python3 plot_ddqn_dqn_cp.py `

## Code structure

```
.
├── agent_dir 
│   ├── agent.py (x)
│   ├── agent_dqn.py 
│   └── agent_pg.py 
├── plot 
│   ├── pg_learning_curve.png
│   ├── dqn_learning_curve.png
│   ├── pg_cp.png
│   ├── dqn_ddqn_cp.png
│   └── dqn_update_freq_cp.png
└── dqn_update_freq_cp.png
├── json (store average rewards) 
│   ├── pg_learning_curve.json
│   ├── dqn_learning_curve.json
│   ├── dqn_learning_curve_500.json
│   ├── dqn_learning_curve_1500.json
│   └── dqn_learning_curve_2000.json
├── dqn_plot_learning_curve.py
├── pg_plot_learning_curve.py
├── plot_ddqn_dqn_cp.py
├── plot_compare.py
├── plot_pg_cp.py
├── pg.cpt (model state of policy gradient)
├── dqn_target.cpt (model state of DQN)
├── dqn_online.cpt (model state of DQN)
├── README.md
├── Report.pdf
├── argument.py (optional)
├── atari_wrapper.py (x)
├── environment.py (x) 
├── main.py (x) 
└── test.py (x)
```
