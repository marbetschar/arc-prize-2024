from typing import Dict, List, Tuple

import kaggle
import json
import os
import math
import shutil
import torch

from pathlib import Path
from zipfile import ZipFile


class Arc20204Dataset(torch.utils.data.Dataset):
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

    @classmethod
    def generate_permutations(
            cls,
            challenge_tensor: torch.Tensor,
            solution_tensor: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        permutations = []

        for i in range(1, 11, 1):
            challenge_permutated = cls.unpad(torch.clone(challenge_tensor).squeeze(0))
            challenge_permutated = (challenge_permutated + i) % 10
            challenge_permutated = cls.pad(challenge_permutated).unsqueeze(0)

            solution_permutated = cls.unpad(torch.clone(solution_tensor).squeeze(0))
            solution_permutated = (solution_permutated + i) % 10
            solution_permutated = cls.pad(solution_permutated).unsqueeze(0)

            if not torch.equal(challenge_tensor, challenge_permutated) or (
                    not torch.equal(solution_tensor, solution_permutated)
            ):
                permutations.append(
                    (challenge_permutated, solution_permutated)
                )

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
            permutations: bool = False,
            download: int = -1
    ):
        """
        Dataset which loads data from ARC 2024 dataset and provides a challenge along with its solution.
        `ZeroShotDataset` does not differentiate between `train` and `test` samples within a set of challenges.

        :param dataset_dir:
        :param dataset_name: `training`, `evaluation` or `test`
        :param mode: `train` or `test`
        :param permutations: `True` if you automatically want to create variations of the provided support set data
        :param download: `-1` = download if not already exists (default), `0` = don't download, `1` = always download
        """

        if dataset_name not in ['training', 'evaluation', 'test']:
            raise ValueError(f'dataset_name {dataset_name} is not supported')

        if mode not in ['train', 'test']:
            raise ValueError(f'mode {mode} is not supported')

        if dataset_name in ['evaluation', 'test']:
            permutations = False

        if download != 0:
            self.download(dataset_dir, download == 1)

        challenges_json = self.read_json_file(dataset_dir / f"arc-agi_{dataset_name}_challenges.json")
        solutions_json = {}

        solutions_file_path = dataset_dir / f"arc-agi_{dataset_name}_solutions.json"
        if solutions_file_path.exists():
            solutions_json = self.read_json_file(solutions_file_path)

        self.challenges = self.pre_process_challenges_and_solutions(
            mode=mode,
            challenges_json=challenges_json,
            solutions_json=solutions_json,
            permutations=permutations
        )

    @classmethod
    def pre_process_challenges_and_solutions(
            cls,
            mode: str,
            challenges_json,
            solutions_json,
            permutations: bool = False
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        challenges = []

        for challenge_id in challenges_json:
            challenge_json = challenges_json[challenge_id]
            solution_json = solutions_json[challenge_id]

            if mode == 'test':
                for i, test in enumerate(challenge_json['test']):
                    challenges += cls.pre_process_challenge_and_solution(
                        test['input'],
                        solution_json[i],
                        permutations=permutations
                    )

            else:
                for train_json in challenge_json['train']:
                    challenges += cls.pre_process_challenge_and_solution(
                        train_json['input'],
                        train_json['output'],
                        permutations=permutations
                    )

        return challenges

    @classmethod
    def pre_process_challenge_and_solution(
            cls,
            challenge_json,
            solution_json,
            permutations: bool = False
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        challenge_tensor = cls.pad(torch.tensor(challenge_json, dtype=torch.float)).unsqueeze(0)
        solution_tensor = cls.pad(torch.tensor(solution_json, dtype=torch.float)).unsqueeze(0)

        challenge_and_solution_tensors = [(challenge_tensor, solution_tensor)]
        if permutations:
            challenge_and_solution_tensors += cls.generate_permutations(challenge_tensor, solution_tensor)

        return challenge_and_solution_tensors

    def __len__(self) -> int:
        return len(self.challenges)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.challenges[index]
