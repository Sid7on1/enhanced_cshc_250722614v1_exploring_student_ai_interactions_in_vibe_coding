import logging
import numpy as np
from typing import Dict, List, Tuple
from config import Config
from utils import load_config, get_logger
from data_models import RewardData, StudentData
from constants import REWARD_TYPES, STUDENT_STATUS

class RewardSystem:
    """
    Reward calculation and shaping system.

    This class is responsible for calculating rewards based on student interactions
    with the Vibe Coding platform. It uses the Flow Theory and velocity-threshold
    algorithms to determine the rewards.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward system.

        Args:
            config (Config): Configuration object.
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.reward_types = REWARD_TYPES
        self.student_status = STUDENT_STATUS

    def calculate_reward(self, student_data: StudentData, interaction_data: Dict) -> RewardData:
        """
        Calculate the reward for a student based on their interaction data.

        Args:
            student_data (StudentData): Student data.
            interaction_data (Dict): Interaction data.

        Returns:
            RewardData: Reward data.
        """
        try:
            # Calculate the velocity
            velocity = self.calculate_velocity(interaction_data)

            # Calculate the reward based on the velocity and student status
            reward = self.calculate_reward_based_on_velocity(velocity, student_data)

            # Add additional rewards based on the interaction data
            reward = self.add_additional_rewards(reward, interaction_data)

            return reward

        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            return RewardData()

    def calculate_velocity(self, interaction_data: Dict) -> float:
        """
        Calculate the velocity based on the interaction data.

        Args:
            interaction_data (Dict): Interaction data.

        Returns:
            float: Velocity.
        """
        try:
            # Get the time difference between interactions
            time_diff = interaction_data['time_diff']

            # Calculate the velocity
            velocity = interaction_data['num_interactions'] / time_diff

            return velocity

        except Exception as e:
            self.logger.error(f"Error calculating velocity: {str(e)}")
            return 0.0

    def calculate_reward_based_on_velocity(self, velocity: float, student_data: StudentData) -> RewardData:
        """
        Calculate the reward based on the velocity and student status.

        Args:
            velocity (float): Velocity.
            student_data (StudentData): Student data.

        Returns:
            RewardData: Reward data.
        """
        try:
            # Get the student status
            student_status = student_data.status

            # Calculate the reward based on the velocity and student status
            if student_status == self.student_status['active']:
                reward = self.calculate_active_reward(velocity)
            elif student_status == self.student_status['inactive']:
                reward = self.calculate_inactive_reward(velocity)
            else:
                reward = self.calculate_unknown_reward(velocity)

            return reward

        except Exception as e:
            self.logger.error(f"Error calculating reward based on velocity: {str(e)}")
            return RewardData()

    def calculate_active_reward(self, velocity: float) -> RewardData:
        """
        Calculate the reward for an active student based on the velocity.

        Args:
            velocity (float): Velocity.

        Returns:
            RewardData: Reward data.
        """
        try:
            # Calculate the reward based on the velocity
            if velocity > self.config.velocity_threshold:
                reward = self.config.active_reward * self.config.velocity_multiplier
            else:
                reward = self.config.active_reward

            return RewardData(reward_type=self.reward_types['active'], reward=reward)

        except Exception as e:
            self.logger.error(f"Error calculating active reward: {str(e)}")
            return RewardData()

    def calculate_inactive_reward(self, velocity: float) -> RewardData:
        """
        Calculate the reward for an inactive student based on the velocity.

        Args:
            velocity (float): Velocity.

        Returns:
            RewardData: Reward data.
        """
        try:
            # Calculate the reward based on the velocity
            if velocity > self.config.velocity_threshold:
                reward = self.config.inactive_reward * self.config.velocity_multiplier
            else:
                reward = self.config.inactive_reward

            return RewardData(reward_type=self.reward_types['inactive'], reward=reward)

        except Exception as e:
            self.logger.error(f"Error calculating inactive reward: {str(e)}")
            return RewardData()

    def calculate_unknown_reward(self, velocity: float) -> RewardData:
        """
        Calculate the reward for an unknown student based on the velocity.

        Args:
            velocity (float): Velocity.

        Returns:
            RewardData: Reward data.
        """
        try:
            # Calculate the reward based on the velocity
            if velocity > self.config.velocity_threshold:
                reward = self.config.unknown_reward * self.config.velocity_multiplier
            else:
                reward = self.config.unknown_reward

            return RewardData(reward_type=self.reward_types['unknown'], reward=reward)

        except Exception as e:
            self.logger.error(f"Error calculating unknown reward: {str(e)}")
            return RewardData()

    def add_additional_rewards(self, reward: RewardData, interaction_data: Dict) -> RewardData:
        """
        Add additional rewards based on the interaction data.

        Args:
            reward (RewardData): Reward data.
            interaction_data (Dict): Interaction data.

        Returns:
            RewardData: Reward data.
        """
        try:
            # Get the interaction type
            interaction_type = interaction_data['interaction_type']

            # Add additional rewards based on the interaction type
            if interaction_type == 'correct':
                reward = self.add_correct_reward(reward)
            elif interaction_type == 'incorrect':
                reward = self.add_incorrect_reward(reward)
            else:
                reward = self.add_unknown_reward(reward)

            return reward

        except Exception as e:
            self.logger.error(f"Error adding additional rewards: {str(e)}")
            return RewardData()

    def add_correct_reward(self, reward: RewardData) -> RewardData:
        """
        Add a correct reward.

        Args:
            reward (RewardData): Reward data.

        Returns:
            RewardData: Reward data.
        """
        try:
            # Calculate the correct reward
            correct_reward = self.config.correct_reward

            # Add the correct reward to the total reward
            reward.reward += correct_reward

            return reward

        except Exception as e:
            self.logger.error(f"Error adding correct reward: {str(e)}")
            return RewardData()

    def add_incorrect_reward(self, reward: RewardData) -> RewardData:
        """
        Add an incorrect reward.

        Args:
            reward (RewardData): Reward data.

        Returns:
            RewardData: Reward data.
        """
        try:
            # Calculate the incorrect reward
            incorrect_reward = self.config.incorrect_reward

            # Add the incorrect reward to the total reward
            reward.reward += incorrect_reward

            return reward

        except Exception as e:
            self.logger.error(f"Error adding incorrect reward: {str(e)}")
            return RewardData()

    def add_unknown_reward(self, reward: RewardData) -> RewardData:
        """
        Add an unknown reward.

        Args:
            reward (RewardData): Reward data.

        Returns:
            RewardData: Reward data.
        """
        try:
            # Calculate the unknown reward
            unknown_reward = self.config.unknown_reward

            # Add the unknown reward to the total reward
            reward.reward += unknown_reward

            return reward

        except Exception as e:
            self.logger.error(f"Error adding unknown reward: {str(e)}")
            return RewardData()


class RewardSystemFactory:
    """
    Reward system factory.

    This class is responsible for creating instances of the RewardSystem class.
    """

    def create_reward_system(self, config: Config) -> RewardSystem:
        """
        Create a reward system instance.

        Args:
            config (Config): Configuration object.

        Returns:
            RewardSystem: Reward system instance.
        """
        return RewardSystem(config)


def main():
    # Load the configuration
    config = load_config()

    # Create a reward system factory
    factory = RewardSystemFactory()

    # Create a reward system instance
    reward_system = factory.create_reward_system(config)

    # Create some sample interaction data
    interaction_data = {
        'time_diff': 10,
        'num_interactions': 5,
        'interaction_type': 'correct'
    }

    # Create some sample student data
    student_data = StudentData(status='active')

    # Calculate the reward
    reward = reward_system.calculate_reward(student_data, interaction_data)

    # Print the reward
    print(reward)


if __name__ == '__main__':
    main()