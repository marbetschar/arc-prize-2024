from typing import Dict, List, Tuple

import kaggle
import json
import os
import math
import random
import shutil
import torch

from pathlib import Path
from zipfile import ZipFile


class Arc20204Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            dataset_dir: Path,
            dataset_name: str = 'training',
            permutations_enabled: bool = False,
            download: int = -1
    ):
        """
        Abstract base Arc20204Dataset which provides helper functionality for its concrete child implementations.

        :param dataset_dir:
        :param dataset_name: `training`, `evaluation` or `test`
        :param permutations_enabled: `True` to create permutations of the data
        :param download: `-1` = download if not already exists (default), `0` = don't download, `1` = always download
        """

        if dataset_name not in ['training', 'evaluation', 'test']:
            raise ValueError(f'dataset_name {dataset_name} is not supported')

        if dataset_name in ['evaluation', 'test']:
            self.permutations_enabled = False
        else:
            self.permutations_enabled = permutations_enabled

        if download != 0:
            self.download(dataset_dir, download == 1)

    @classmethod
    def blow(cls, tensor: torch.Tensor, target_shape=(30, 30), pad_value=10):
        padding_right = (target_shape[1] - tensor.shape[1])
        padding_bottom = (target_shape[0] - tensor.shape[0])

        m = torch.nn.ConstantPad2d(
            padding=(
                0,  # padding_left
                padding_right,
                0,  # padding_top
                padding_bottom
            ),
            value=pad_value
        )

        return m(tensor)

    # TODO - We must ensure the image is centered here. Otherwise all operations requiring symmetry
    # will not yield correct results (e.g. vertical mirroring) and are therefore hard to compare.
    # An even better approach would be to get rid of padding completely to avoid this issue
    # in the first place.
    @classmethod
    def pad(cls, tensor: torch.Tensor, target_shape=(30, 30), pad_value=10):
        vertical_pad = (target_shape[0] - tensor.shape[0]) / 2.0
        horizontal_pad = (target_shape[1] - tensor.shape[1]) / 2.0

        m = torch.nn.ConstantPad2d(
            padding=(
                math.floor(horizontal_pad),  # padding_left
                math.ceil(horizontal_pad),  # padding_right
                math.floor(vertical_pad),  # padding_top
                math.ceil(vertical_pad)  # padding_bottom
            ),
            value=pad_value
        )

        return m(tensor)

    @classmethod
    def unpad(cls, input: torch.Tensor, pad_value=10):
        input_mask = input[:, :] == pad_value

        # make sure we only remove rows and columns which contain pad_value only
        dim0 = torch.all(input_mask, dim=0) == False
        dim1 = torch.all(input_mask, dim=1) == False

        input_unpadded = input[dim1, :][:, dim0]

        return input_unpadded

    def generate_permutations(
            self,
            challenge_tensor: torch.Tensor,
            solution_tensor: torch.Tensor,
            map_lambda=lambda x, y: (x, y)
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        permutations = []

        if self.permutations_enabled:
            for i in range(1, 11, 1):
                challenge_permutated = self.unpad(torch.clone(challenge_tensor).squeeze(0))
                challenge_permutated = (challenge_permutated + i) % 10
                challenge_permutated = self.pad(challenge_permutated).unsqueeze(0)

                solution_permutated = self.unpad(torch.clone(solution_tensor).squeeze(0))
                solution_permutated = (solution_permutated + i) % 10
                solution_permutated = self.pad(solution_permutated).unsqueeze(0)

                if not torch.equal(challenge_tensor, challenge_permutated) or (
                        not torch.equal(solution_tensor, solution_permutated)
                ):
                    permutations.append(map_lambda(challenge_permutated, solution_permutated))

        return permutations

    @classmethod
    def download(cls, dataset_dir: Path, force: bool = False, competition='arc-prize-2024'):
        if not force and dataset_dir.is_dir() and len(os.listdir(dataset_dir)) > 1:
            return
        elif dataset_dir.is_dir():
            shutil.rmtree(dataset_dir)

        kaggle.api.authenticate()
        kaggle.api.competition_download_files(
            competition=competition,
            path=dataset_dir,
            force=force
        )

        zip_file_path = dataset_dir / f"{competition}.zip"
        if not zip_file_path.exists() or not zip_file_path.is_file():
            raise f"Error downloading zip file from kaggle competition {competition}"
        else:
            with ZipFile(zip_file_path, 'r') as zip_file_handle:
                zip_file_handle.extractall(dataset_dir)
            os.remove(zip_file_path)

    @classmethod
    def read_json_file(cls, file_path: Path):
        file = open(file_path, 'r')
        json_data = json.loads(file.read())
        file.close()
        return json_data


class ZeroShotDataset(Arc20204Dataset):

    def __init__(
            self,
            dataset_dir: Path,
            dataset_name: str = 'training',
            mode: str = 'train',
            permutations_enabled: bool = False,
            download: int = -1
    ):
        """
        Dataset which loads data from ARC 2024 dataset and provides a challenge along with its solution.
        `ZeroShotDataset` does not differentiate between `train` and `test` samples within a set of challenges.

        :param mode: `train` or `test`
        """
        super().__init__(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            permutations_enabled=permutations_enabled,
            download=download
        )

        if mode not in ['train', 'test']:
            raise ValueError(f'mode {mode} is not supported')

        challenges_json = self.read_json_file(dataset_dir / f"arc-agi_{dataset_name}_challenges.json")
        solutions_json = {}

        solutions_file_path = dataset_dir / f"arc-agi_{dataset_name}_solutions.json"
        if solutions_file_path.exists():
            solutions_json = self.read_json_file(solutions_file_path)

        self.challenges, self.challenge_ids = self.pre_process_challenges_and_solutions(
            mode=mode,
            challenges_json=challenges_json,
            solutions_json=solutions_json
        )

    def pre_process_challenges_and_solutions(
            self,
            mode: str,
            challenges_json,
            solutions_json
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[str]]:
        challenges = []
        challenge_ids = []

        for challenge_id in challenges_json:
            challenge_json = challenges_json[challenge_id]
            solution_json = solutions_json[challenge_id]

            if mode == 'test':
                for i, test in enumerate(challenge_json['test']):
                    challenges += self.pre_process_challenge_and_solution(
                        test['input'],
                        solution_json[i]
                    )
                    challenge_ids += [challenge_id]

            else:
                for train_json in challenge_json['train']:
                    challenges += self.pre_process_challenge_and_solution(
                        train_json['input'],
                        train_json['output']
                    )
                    challenge_ids += [challenge_id]

        return challenges, challenge_ids

    def pre_process_challenge_and_solution(
            self,
            challenge_json,
            solution_json
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        challenge_tensor = self.pad(torch.tensor(challenge_json, dtype=torch.float)).unsqueeze(0)
        solution_tensor = self.pad(torch.tensor(solution_json, dtype=torch.float)).unsqueeze(0)

        challenge_and_solution_tensors = [(challenge_tensor, solution_tensor)]
        challenge_and_solution_tensors += self.generate_permutations(challenge_tensor, solution_tensor)

        return challenge_and_solution_tensors

    def __len__(self) -> int:
        return len(self.challenges)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.challenges[index]

    def get_id(self, index: int) -> str:
        return self.challenge_ids[index]


class FewShotDataset(Arc20204Dataset):
    def __init__(
            self,
            dataset_dir: Path,
            dataset_name: str = 'training',
            permutations_enabled: bool = False,
            download: int = -1
    ):
        """
        Dataset which loads data from ARC 2024 dataset and provides challenges along with their solution.
        `NShotDataset` differentiates between `train` and `test` samples within a set of challenges.
        """

        super().__init__(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            permutations_enabled=permutations_enabled,
            download=download
        )

        challenges_json = self.read_json_file(dataset_dir / f"arc-agi_{dataset_name}_challenges.json")
        solutions_json = {}

        solutions_file_path = dataset_dir / f"arc-agi_{dataset_name}_solutions.json"
        if solutions_file_path.exists():
            solutions_json = self.read_json_file(solutions_file_path)

        self.train_challenges, self.test_challenges, self.test_challenge_ids = (
            self.pre_process_challenges_and_solutions(
                challenges_json=challenges_json,
                solutions_json=solutions_json
            )
        )

    def pre_process_challenge_and_solution(
            self,
            challenge_json,
            solution_json,
            map_lambda=lambda x, y: (x, y)
    ):
        challenge_tensor = self.pad(torch.tensor(challenge_json, dtype=torch.float)).unsqueeze(0)
        solution_tensor = self.pad(torch.tensor(solution_json, dtype=torch.float)).unsqueeze(0)

        challenge_and_solution_tensors = [map_lambda(challenge_tensor, solution_tensor)]
        challenge_and_solution_tensors += self.generate_permutations(
            challenge_tensor=challenge_tensor,
            solution_tensor=solution_tensor,
            map_lambda=map_lambda
        )

        return challenge_and_solution_tensors

    def __len__(self) -> int:
        return len(self.test_challenges)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        a, b, _ = self.test_challenges[index]
        if torch.equal(a, b):
            raise ValueError("Cannot be equal")
        return self.test_challenges[index]

    def get_samples(self, challenge_id: str, max_samples: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        if challenge_id not in self.train_challenges:
            raise KeyError(f'Challenge {challenge_id} not found in training challenges.')

        all_samples = self.train_challenges[challenge_id]
        k = min(max_samples, len(all_samples)) if max_samples > 0 else len(all_samples)

        samples = random.sample(all_samples, k)

        x_sample = []
        y_sample = []
        for sample in samples:
            x, y = sample
            x_sample.append(x)
            y_sample.append(y)

        return torch.stack(x_sample), torch.stack(y_sample)

    def pre_process_challenges_and_solutions(
            self,
            challenges_json,
            solutions_json
    ) -> Tuple[Dict, List, List]:
        train_challenges = {}
        test_challenges = []
        test_challenge_ids = []

        for challenge_id in challenges_json:
            challenge_json = challenges_json[challenge_id]
            solution_json = solutions_json[challenge_id]

            for i, test in enumerate(challenge_json['test']):
                test_challenge_ids.append(challenge_id)
                test_challenges += self.pre_process_challenge_and_solution(
                    challenge_json=test['input'],
                    solution_json=solution_json[i],
                    map_lambda=lambda x, y: (x, y, challenge_id)
                )

            train_challenges[challenge_id] = []
            for train_json in challenge_json['train']:
                train_challenges[challenge_id] += self.pre_process_challenge_and_solution(
                    challenge_json=train_json['input'],
                    solution_json=train_json['output'],
                    map_lambda=lambda x, y: (x, y)
                )

        return train_challenges, test_challenges, test_challenge_ids
