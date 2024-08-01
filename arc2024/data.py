from typing import Dict, List, Tuple

import kaggle
import json
import os
import shutil
import torch

from pathlib import Path
from zipfile import ZipFile


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_dir: Path,
        mode: str = 'training',
        permutations: bool = False,
        download: int = -1
    ):
        """
        Dataset for ARC Prize 2024 competition. Can automatically download the dataset from kaggle.com

        :param dataset_dir:
        :param mode: `training`, `evaluation` or `test`
        :param permutations: `True` if you automatically want to create variations of the provided support set data
        :param download: `-1` = download if not already exists (default), `0` = don't download, `1` = always download
        """

        if mode not in ['training', 'evaluation', 'test']:
            raise ValueError(f'mode {mode} is not supported')

        if mode in ['evaluation', 'test']:
            permutations = False

        if download != 0:
            self.__download(dataset_dir, download == 1)

        challenges_json = self.__read_json_file(dataset_dir / f"arc-agi_{mode}_challenges.json")
        solutions_json = {}

        solutions_file_path = dataset_dir / f"arc-agi_{mode}_solutions.json"
        if solutions_file_path.exists():
            solutions_json = self.__read_json_file(solutions_file_path)

        self.challenges = self.__pre_process_challenges_and_solutions(
            challenges_json=challenges_json,
            solutions_json=solutions_json,
            permutations=permutations
        )
        self.challenge_keys = list(self.challenges.keys())

    @classmethod
    def __pre_process_challenges_and_solutions(
        cls,
        challenges_json,
        solutions_json,
        permutations: bool = False
    ) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        challenges = {}

        for challenge_id in challenges_json:
            support_set_inputs, support_set_outputs, query_inputs, query_outputs = (
                cls.__pre_process_challenge_and_solution(
                    challenges_json[challenge_id]['train'],
                    challenges_json[challenge_id]['test'],
                    solutions_json.get(challenge_id, {}),
                    permutations
                )
            )

            challenges[challenge_id] = {
                'support_set_inputs': support_set_inputs,
                'support_set_outputs': support_set_outputs,
                'query_inputs': query_inputs,
                'query_outputs': query_outputs
            }

        return challenges

    @classmethod
    def __pre_process_challenge_and_solution(
        cls,
        train_json,
        test_json,
        solution_json,
        permutations: bool = False
    ):
        support_set_inputs = []
        support_set_outputs = []
        query_inputs = []
        query_outputs = []

        for i, train in enumerate(train_json):
            support_set_input = torch.tensor(train['input'], dtype=torch.uint8).unsqueeze(0)
            support_set_output = torch.tensor(train['output'], dtype=torch.uint8).unsqueeze(0)

            support_set_inputs.append(support_set_input)
            support_set_outputs.append(support_set_output)

        if permutations:
            support_set_inputs += cls.__generate_permutations(support_set_inputs)
            support_set_outputs += cls.__generate_permutations(support_set_outputs)

        for i, test in enumerate(test_json):
            query_input = torch.tensor(test['input'], dtype=torch.uint8).unsqueeze(0)

            query_output = None
            if len(solution_json) > i:
                query_output = torch.tensor(solution_json[i], dtype=torch.uint8).unsqueeze(0)

            query_inputs.append(query_input)
            query_outputs.append(query_output)

        if permutations:
            query_inputs += cls.__generate_permutations(query_inputs)
            query_outputs += cls.__generate_permutations(query_outputs)

        return support_set_inputs, support_set_outputs, query_inputs, query_outputs

    @classmethod
    def __generate_permutations(cls, origins: List[torch.Tensor]) -> List[torch.Tensor]:
        permutations = []

        for i in range(1, 11, 1):
            for origin in origins:
                permutation = (torch.clone(origin).squeeze(0) + i) % 10
                permutation = permutation.unsqueeze(0)

                if torch.equal(origin, permutation):
                    continue

                permutations.append(permutation)

        return permutations

    @classmethod
    def __download(cls, dataset_dir: Path, force: bool = False, competition='arc-prize-2024'):
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
    def __read_json_file(cls, file_path: Path):
        file = open(file_path, 'r')
        json_data = json.loads(file.read())
        file.close()
        return json_data

    def __len__(self) -> int:
        return len(self.challenge_keys)

    def __getitem__(self, index: int) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
    ]:
        key = self.challenge_keys[index]
        challenge = self.challenges[key]

        return (
            challenge['support_set_inputs'],
            challenge['support_set_outputs'],
            challenge['query_inputs'],
            challenge['query_outputs']
        )
