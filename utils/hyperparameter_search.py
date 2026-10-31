"""Validation-only ridge-lambda selection for Fashion-MNIST."""
from dataclasses import replace
import numpy as np
import torch
import torch.nn.functional as F
from .datasets import random_coreset
from .distillation import distill_coreset
from .evaluation import evaluate_coreset
from .reservoir_onn import tm_ensemble


def validation_split(x_train, y_train, validation_size=5000, seed=424242):
    """Reproduce the historical split on the selected device."""
    generator = torch.Generator(device=x_train.device).manual_seed(seed)
    permutation = torch.randperm(len(x_train), generator=generator, device=x_train.device)
    validation, training = permutation[:validation_size], permutation[validation_size:]
    return x_train[training], y_train[training], x_train[validation], y_train[validation]


def search_lambda(x_train, y_train, x_validation, y_validation, config,
                  lambda_grid=(0.01, 0.1, 1.0, 10.0), method='x_plus_y',
                  output_dim=1024, tm_count=50):
    """Train each distilled candidate from common seeds and score validation data.

    The TM ranges, support seeds and final-step selection match the historical
    FMNIST protocol. The full-data and random candidates do not run distillation.
    Test images and labels are never passed to this function.
    """
    device = x_train.device
    input_dim = x_train.shape[1]
    train_tms = tm_ensemble(input_dim, output_dim, range(40000, 40000+tm_count), device)
    validation_start = {'random':46000, 'full':47000}.get(method,45000)
    validation_tms = tm_ensemble(input_dim, output_dim, range(validation_start, validation_start+tm_count), device)
    scores = []
    for ridge_lambda in lambda_grid:
        if method == 'full':
            support_x = x_train
            support_y = F.one_hot(y_train, config.num_classes).float()
        elif method == 'random':
            support_x, support_y, _, _ = random_coreset(x_train, y_train, config.images_per_class, 22025, config.num_classes)
        elif method in {'x_plus_y','only_x','only_y'}:
            candidate = replace(config, ridge_lambda=float(ridge_lambda),
                                coreset_seed=17, batch_seed=123, eval_every=0,
                                optimize_x=method!='only_y', optimize_y=method!='only_x',
                                lr_x=0.005 if method=='only_x' else 0.01,
                                lr_y=0.005 if method=='only_x' else 0.01)
            result = distill_coreset(x_train, y_train, train_tms, candidate)
            support_x, support_y = result.x_coreset, result.y_coreset
        else:
            raise ValueError(f'Unknown selection method: {method}')
        score = evaluate_coreset(support_x, support_y, x_validation, y_validation,
                                validation_tms, float(ridge_lambda), config.alpha)['accuracy_mean']
        scores.append(score)
        print(f'{method}, IPC={config.images_per_class}, lambda={ridge_lambda:g}: validation accuracy={score:.5f}',flush=True)
    return {'lambda_grid':np.asarray(lambda_grid), 'validation_accuracy':np.asarray(scores),
            'selected_lambda':float(lambda_grid[int(np.argmax(scores))])}
