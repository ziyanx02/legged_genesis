## TODO

Domain randomization.

## Usage

After adding any assets, **REMEMBER TO CHECK**:
- joint names and body names, including foot_name etc.
- default dof pos, and rearrange them in the other used in Genesis

Run `python train.py 000-00-ANYTHING --task a1 --entity YOUR_WANDB_ENTITY --headless` to train.

Run `python play.py 000-00 --task a1` to evaluate. `000-00` can be replaced to any exptid (automatically match the description).