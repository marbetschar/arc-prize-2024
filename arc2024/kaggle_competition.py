import kaggle
import json
from zipfile import ZipFile
import os
import shutil

import torch
import torchvision

from pathlib import Path


def json_file_read(file_path: Path):
    file = open(file_path, 'r')
    json_data = json.loads(file.read())
    file.close()
    return json_data


def image_to_tensor(path: Path):
    return torchvision.io.read_image(
        path=str(path),
        mode=torchvision.io.image.ImageReadMode.UNCHANGED
    )


def tensor_to_image(image_path_without_extension: Path, input: torch.Tensor, permutations: bool = False):
    torchvision.io.write_png(
        input=input,
        filename=str(
            image_path_without_extension.with_name(
                f"{image_path_without_extension.name}-0"
            ).with_suffix('.png')
        ),
        compression_level=0
    )

    if permutations:
        k = 0
        for i in range(1, 11, 1):
            input_permutated = (torch.clone(input).squeeze(0) + i) % 10
            input_permutated = input_permutated.unsqueeze(0)

            if torch.equal(input, input_permutated):
                continue
            k += 1

            torchvision.io.write_png(
                input=input_permutated,
                filename=str(
                    image_path_without_extension.with_name(
                        f"{image_path_without_extension.name}-{k}"
                    ).with_suffix('.png')
                ),
                compression_level=0
            )


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
            zip_file_handle.extractall(dataset_dir / 'json')
        os.remove(zip_file_path)

    training_challenges = json_file_read(dataset_dir / 'json/arc-agi_training_challenges.json')
    training_solutions = json_file_read(dataset_dir / 'json/arc-agi_training_solutions.json')
    challenges_and_solutions_to_png(
        working_dir=dataset_dir / 'training',
        challenges_json=training_challenges,
        solutions_json=training_solutions,
        permutations=True,
        force=force
    )

    evaluation_challenges = json_file_read(dataset_dir / 'json/arc-agi_evaluation_challenges.json')
    evaluation_solutions = json_file_read(dataset_dir / 'json/arc-agi_evaluation_solutions.json')
    challenges_and_solutions_to_png(
        working_dir=dataset_dir / 'evaluation',
        challenges_json=evaluation_challenges,
        solutions_json=evaluation_solutions,
        permutations=False,
        force=force
    )


def challenge_and_solution_to_png(
        working_dir: Path,
        challenge_id: str,
        train_json,
        test_json,
        solution_json,
        permutations: bool = False
):
    challenge_dir = working_dir / challenge_id

    support_set_inputs_dir = challenge_dir / 'support-set_inputs'
    support_set_outputs_dir = challenge_dir / 'support-set_outputs'

    query_inputs_dir = challenge_dir / 'query_inputs'
    query_outputs_dir = challenge_dir / 'query_outputs'

    for d in [support_set_inputs_dir, support_set_outputs_dir, query_inputs_dir, query_outputs_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    for i, train in enumerate(train_json):
        tensor_to_image(
            support_set_inputs_dir / str(i),
            torch.tensor(train['input'], dtype=torch.uint8).unsqueeze(0),
            permutations
        )
        tensor_to_image(
            support_set_outputs_dir / str(i),
            torch.tensor(train['output'], dtype=torch.uint8).unsqueeze(0),
            permutations
        )

    for i, test in enumerate(test_json):
        tensor_to_image(
            query_inputs_dir / str(i),
            torch.tensor(test['input'], dtype=torch.uint8).unsqueeze(0),
            permutations
        )
        tensor_to_image(
            query_outputs_dir / str(i),
            torch.tensor(solution_json[i], dtype=torch.uint8).unsqueeze(0),
            permutations
        )


def challenges_and_solutions_to_png(
        working_dir: Path,
        challenges_json,
        solutions_json,
        permutations: bool = False,
        force: bool = False
):
    if not force and working_dir.is_dir():
        return
    elif working_dir.is_dir():
        shutil.rmtree(working_dir)

    for challenge_id in challenges_json:
        challenge_and_solution_to_png(
            working_dir,
            challenge_id,
            challenges_json[challenge_id]['train'],
            challenges_json[challenge_id]['test'],
            solutions_json[challenge_id],
            permutations
        )
