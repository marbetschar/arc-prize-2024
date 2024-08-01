from typing import Dict, List

import kaggle
import json
from zipfile import ZipFile
import os
import shutil

import torch

from pathlib import Path


def json_file_read(file_path: Path):
    file = open(file_path, 'r')
    json_data = json.loads(file.read())
    file.close()
    return json_data


def download_files(
    dataset_dir: Path,
    competition: str = 'arc-prize-2024',
    force: bool = False
):
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


def tensor_permutations(input: torch.Tensor) -> List[torch.Tensor]:
    tensors_permutated = []

    for i in range(1, 11, 1):
        input_permutated = (torch.clone(input).squeeze(0) + i) % 10
        input_permutated = input_permutated.unsqueeze(0)

        if torch.equal(input, input_permutated):
            continue

        tensors_permutated.append(input_permutated)

    return tensors_permutated


def challenge_and_solution_to_tensors(
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
            support_set_inputs += tensor_permutations(support_set_input)
            support_set_outputs += tensor_permutations(support_set_output)

    for i, test in enumerate(test_json):
        query_input = torch.tensor(test['input'], dtype=torch.uint8).unsqueeze(0)
        query_output = torch.tensor(solution_json[i], dtype=torch.uint8).unsqueeze(0)

        query_inputs.append(query_input)
        query_outputs.append(query_output)

        if permutations:
            query_inputs += tensor_permutations(query_input)
            query_outputs += tensor_permutations(query_output)

    return support_set_inputs, support_set_outputs, query_inputs, query_outputs


def challenges_and_solutions_to_tensors(
    challenges_json,
    solutions_json,
    permutations: bool = False
) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    tensors = {}

    for challenge_id in challenges_json:
        support_set_inputs, support_set_outputs, query_inputs, query_outputs = challenge_and_solution_to_tensors(
            challenges_json[challenge_id]['train'],
            challenges_json[challenge_id]['test'],
            solutions_json[challenge_id],
            permutations
        )

        tensors[challenge_id] = {
            'support_set_inputs': support_set_inputs,
            'support_set_outputs': support_set_outputs,
            'query_inputs': query_inputs,
            'query_outputs': query_outputs
        }

    return tensors
