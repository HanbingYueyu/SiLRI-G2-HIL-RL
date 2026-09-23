"""Gymnasium adapter for local insertion with explicit backend selection."""
import uuid
import logging
from copy import deepcopy
import cv2
import gymnasium as gym
import numpy as np
from .contract import CAMERA_KEYS, EpisodeContext
from .episode import EpisodeRunner, StepResult


class SyntheticBackend:
    """Software transport test only: no contact physics or success simulation."""
    name = 'synthetic'

    def __init__(self):
        self.state = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)

    def observe(self):
        return dict(state=self.state.copy(), **{
            key: np.full((128, 128, 3), 64 + i * 32, dtype=np.uint8)
            for i, key in enumerate(CAMERA_KEYS)})

    def execute(self, action):
        self.state[:3] += np.asarray(action[:3], dtype=np.float32) * .001
        return StepResult(self.observe(), tuple(action), 0., False)

    def stop(self):
        pass

    def close(self):
        pass


class G2LocalEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, backend, *, max_steps=100, image_size=128, camera_rois=None,
                 intervention=None):
        self.backend = backend
        self.runner = EpisodeRunner(backend, max_steps=max_steps)
        self.image_size = image_size
        self.camera_rois = dict(camera_rois or {})
        if not set(self.camera_rois) <= set(CAMERA_KEYS):
            raise ValueError('Unknown camera ROI key')
        self.last_crop_boxes = {}
        self.intervention = intervention
        self.action_space = gym.spaces.Box(-1., 1., (6,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
            **{key: gym.spaces.Box(0, 255, (image_size, image_size, 3), dtype=np.uint8)
               for key in CAMERA_KEYS}})

    def _observation(self, raw):
        if set(raw) != set(self.observation_space.spaces):
            raise ValueError('Observation keys do not match dual-camera schema')
        result = {'state': np.asarray(raw['state'], dtype=np.float32).copy()}
        crop_boxes = {}
        for key in CAMERA_KEYS:
            image = raw[key]
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f'{key} must be HWC uint8 RGB')
            if key in self.camera_rois:
                box = self.camera_rois[key]
                if (not isinstance(box, (tuple, list)) or len(box) != 4 or
                        any(type(value) is not int for value in box)):
                    raise ValueError(f'{key} ROI must have integer pixel bounds')
                x0, y0, x1, y1 = box
                if not (0 <= x0 < x1 <= image.shape[1] and
                        0 <= y0 < y1 <= image.shape[0]):
                    raise ValueError(f'{key} ROI is outside acquired frame')
                crop_boxes[key] = tuple(box)
                image = image[y0:y1, x0:x1]
            result[key] = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        if not np.isfinite(result['state']).all() or not self.observation_space.contains(result):
            raise ValueError('Invalid observation shape or values')
        self.last_crop_boxes = crop_boxes
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        context = (options or {}).get('context')
        if context is None:
            if not isinstance(self.backend, SyntheticBackend):
                raise ValueError('Real reset requires explicit EpisodeContext after scene reset')
            context = EpisodeContext(str(uuid.uuid4()), (0, 0, 0), 'synthetic', 'synthetic')
        raw = self.runner.reset(context)
        try:
            observation = self._observation(raw)
        except Exception:
            self.runner.close()
            raise
        return observation, {'backend': getattr(self.backend, 'name', 'gdk')}

    def step(self, action):
        if not self.runner.active:
            raise RuntimeError('Reset required before stepping')
        try:
            active, human = self.intervention() if self.intervention else (False, None)
        except Exception:
            try:
                self.runner.close()
            except Exception:
                logging.exception('Backend stop failed after intervention error')
            raise
        row = self.runner.step(action, human_active=active, human=human)
        try:
            obs = self._observation(row['next_state'])
        except Exception:
            self.runner.close()
            raise
        success_label = row['complementary_info']['success_label']
        succeed = (success_label if success_label is not None else
                    bool(row['done'] and row['reward'] > 0))
        info = dict(row['complementary_info'],
                    executed_action=np.asarray(row['action'], dtype=np.float32),
                    intervene_action=np.asarray(row['action'], dtype=np.float32),
                    reward_source=row['complementary_info']['reward_source'],
                    success_label=success_label,
                    succeed=succeed,
                    backend=getattr(self.backend, 'name', 'gdk'))
        return obs, row['reward'], row['done'], row['truncated'], info

    def refresh_observation(self):
        """Acquire a fresh predecessor through the backend's freshness guard."""
        if not self.runner.active:
            raise RuntimeError('Reset required before refreshing observation')
        try:
            raw = self.backend.observe()
            policy_obs = self._observation(raw)
            self.runner.observation = deepcopy(raw)
            return policy_obs
        except BaseException:
            try:
                self.close()
            except Exception:
                logging.exception('Backend close failed after observation refresh error')
            raise

    def close(self):
        try:
            self.runner.close()
        finally:
            self.backend.close()
