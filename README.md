## Usage

Install genesis, rsl_rl and legged_gym that are in the root directory.

Bash `cd legged_gym/legged_gym/scripts`.

Replace wandb login key and entity in `train.py` by yours.

Run `python train.py 000-00-ANYTHING --task go2_wtw --headless --max_iterations 8000` to train. (10000 to converge, could early terminate at 4000 to finetune)

Run `python play_wtw.py 001-22 --task go2_wtw` to evaluate the existant checkpoint. Replace `001-22` by your exptid (i.e. `000-00`).

Check `domain_rand`, `commands`, `reward_scales` in `legged_gym/legged_gym/wtw/legged_robot_config.py` and `control` in `legged_gym/legged_gym/wtw/go2_config.py` for detailed configuration. In `commands`, remember to modify `lin_vel_x` as well as `limit_vel_x` (and other command ranges) together unless you are sure about how the curriculum works.

Currently I'm using kp 40 and kv 0.5. Kp 20 seems to be too soft for intense command.