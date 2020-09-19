
import pickle as pkl
import numpy as np


# load saved rb
rb = '50k_norm'
rb_filename = f'output/catcher-short/data/{rb}/replay_buffer.pkl'
with open(rb_filename, 'rb') as f:
    replay_buffer = pkl.load(f)['replay_buffer']
f.close()

dsize = replay_buffer._transitions_stored
rdict, adict = {}, {}
for x in range(dsize // 256):
    indices = np.arange(x * 256, x* 256 + 256)
    batch = replay_buffer.sample_transitions(256, idx=indices)
    rewards = batch['reward']
    for rval in np.unique(rewards):
        rdict[rval] = rdict.get(rval, 0) + (rewards == rval).astype(np.float32).sum()
    actions = batch['action']
    actions = np.argmax(actions, axis=-1)
    for rval in np.unique(actions):
        adict[rval] = adict.get(rval, 0) + (actions == rval).astype(np.float32).sum()

for key in rdict.keys():
    rdict[key] = rdict[key] / dsize
for key in adict.keys():
    adict[key] = adict[key] / dsize

print('Reward stats', rdict)
print('Action stats', adict)



