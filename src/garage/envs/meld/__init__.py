"""Garage wrappers for MELD environments."""
from garage.envs.meld.cheetah.meld_cheetah_vel import HalfCheetahVelEnv
from garage.envs.meld.cheetah.meld_cheetah_wrapper import MeldCheetahWrapper

from garage.envs.meld.reacher.sawyer_reacher import SawyerReachingEnvMultitask
from garage.envs.meld.peg.sawyer_peg import SawyerPegInsertionEnv4Box
from garage.envs.meld.shelf.sawyer_shelf import SawyerPegShelfEnvMultitask
from garage.envs.meld.button.sawyer_button import SawyerButtonsEnv
from garage.envs.meld.sawyer_wrapper import MeldReachingWrapper, MeldPegWrapper, MeldShelfWrapper, MeldButtonWrapper


__all__ = [
    'HalfCheetahVelEnv',
    'MeldCheetahWrapper',
    'SawyerReachingEnvMultitask',
    'SawyerPegInsertionEnv4Box',
    'SawyerPegShelfEnvMultitask',
    'SawyerButtonsEnv',
    'MeldReachingWrapper',
    'MeldPegWrapper',
    'MeldShelfWrapper',
    'MeldButtonWrapper',
]
