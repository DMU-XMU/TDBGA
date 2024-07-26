import os
os.environ['MUJOCO_GL']='egl'
import numpy as np
import gym
import mujoco_py as mjc
import imageio


def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)


def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    qstate_dim = qpos_dim + qvel_dim

    if 'ant' in env.name:
        ypos = np.zeros(1)
        state = np.concatenate([ypos, state])

    if state.size == qpos_dim - 1 or state.size == qstate_dim - 1:
        xpos = np.zeros(1)
        state = np.concatenate([xpos, state])

    if state.size == qpos_dim:
        qvel = np.zeros(qvel_dim)
        state = np.concatenate([state, qvel])

    if 'ant' in env.name and state.size > qpos_dim + qvel_dim:
        xpos = np.zeros(1)
        state = np.concatenate([xpos, state])[:qstate_dim]

    assert state.size == qpos_dim + qvel_dim

    env.set_state(state[:qpos_dim], state[qpos_dim:])


def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions)+1):
        ## if terminated early, pad with zeros
        observations.append( np.zeros(obs.size) )
    return np.stack(observations)


def save_video(filename, video_frames, fps=30):
    assert fps == int(fps), fps
    _make_dir(filename)

    imageio.mimsave(filename, video_frames, 'mp4', fps=fps)


def save_videos(filename, *video_frames, **kwargs):
    ## video_frame : [ N x H x W x C ]
    video_frames = np.concatenate(video_frames, axis=2)
    save_video(filename, video_frames, **kwargs)


class Render(object):
    def __init__(self, env, env_id: str, observation_dim=None, action_dim=None):
        if type(env) is str:
            self.env = gym.make(env)
            self.env.name = env.replace("-v2", "").lower()
        else:
            self.env = env
            self.env.name = env_id

        self.observation_dim = observation_dim or np.prod(self.env.observation_space.shape)
        self.action_dim = action_dim or np.prod(self.env.action_space.shape)
        # self.viewer = mjc.MjRenderContextOffscreen(self.env.sim,device_id=0)
        # self.viewer = mjc.MjViewer(self.env.sim)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)
    def render(self, observation, dim=256, render_kwargs=None):
        if render_kwargs is None:
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [0, -0.5, 1],
                'elevation': -20
            }

        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        set_state(self.env, observation)

        if type(dim) == int:
            dim = (dim, dim)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    
    def render_plan(self, savepath, sequence, state, fps=30):
        '''
            state : np.array[ observation_dim ]
            sequence : np.array[ horizon x transition_dim ]
                as usual, sequence is ordered as [ s_t, a_t, r_t, V_t, ... ]
        '''

        if len(sequence) == 1:
            return

        ## compare to ground truth rollout using actions from sequence
        actions = sequence[:-1, self.observation_dim : self.observation_dim + self.action_dim]
        rollout_states = rollout_from_state(self.env, state, actions)

        videos = [
            self.renders(sequence[:, :self.observation_dim]),
            self.renders(rollout_states),
        ]

        save_videos(savepath, *videos, fps=fps)


    def render_rollout(self, savepath, states, **video_kwargs):
        images = self(states)
        save_video(savepath, images, **video_kwargs)
        
    def render_group_rollout(self, save_path, states_lst):
        images = []
        for states in states_lst:
            images.append(self(states))
        images = np.concatenate(images, axis=2)
        
        save_video(save_path, images)