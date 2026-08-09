import torch


ckpt = torch.load("/home/robotlab/isaaclab_ur_reach_sim2real/sample/ur_reach/ur5e_reach_policy.pt")
print(ckpt.keys())