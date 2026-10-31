"""Simulated random support sizes matching measured CCPP and NARMA10 R2."""
from .datasets import flat_result_path, flat_result_directory
from pathlib import Path
import json
import time

import h5py
import numpy as np
import torch

from .reservoir_onn import tm_ensemble
from .regression import evaluate_regression, query_feature_cache, SUPPORT_ONLY_TRAIN_CALIBRATION
from .forecasting import ReservoirReadoutEnsemble, autonomous_rollout, quantize_mask, r2_per_tm

CONFIGS={'ccpp':dict(name='CCPP',size=2,coarse=20,fine=2),
         'narma10':dict(name='NARMA10',size=24,coarse=20,fine=2)}
SEEDS=(0,1,2)


def save_json(path,data):
    path=Path(path)
    temp=path.with_suffix('.tmp.json')
    flat_result_path(temp).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    flat_result_path(temp).replace(flat_result_path(path))


def _narma_origins(root,archive):
    with h5py.File(root/'experimental_results/narma10/sim_source_data/source.mat','r') as f:
        result=dict(starts=np.asarray(f['origin_test_origins']).reshape(-1).astype(int),
            initial_histories=np.asarray(f['origin_initial_histories']).T,
            truth=np.asarray(f['origin_truth']).T)
    state=archive['state_full'].astype(np.float64).reshape(-1).copy()
    for i,start in enumerate(result['starts']):
        # Use the exact initial history presented in the physical recording.
        state[start-9:start+1]=result['initial_histories'][i]
    result['state']=state
    return result


def regression_inputs(root, task):
    """Recreate sample-efficiency inputs from raw data and fixed simulation seeds."""
    from .regression import preprocess_ccpp
    from .forecasting import preprocess_narma10

    root = Path(root)
    def array(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

    if task == 'ccpp':
        runs = [preprocess_ccpp(root/'dataset/CCPP/Folds5x2_pp.xlsx', device='cpu',
            split_seed=20260821+100*rep, projection_seed=20330821) for rep in range(5)]
        names = {'x_train_mask_flat_per_rep':'x_train', 'y_train_standardized_per_rep':'y_train',
            'x_validation_mask_flat_per_rep':'x_val', 'y_validation_standardized_per_rep':'y_val',
            'x_test_mask_flat_per_rep':'x_test', 'y_test_standardized_per_rep':'y_test',
            'y_test_raw_per_rep':'y_test_raw', 'y_mean_per_rep':'y_mean', 'y_std_per_rep':'y_std'}
        result = {key:np.stack([array(data[value]) for data in runs]) for key,value in names.items()}
        result.update(support_sizes=np.array([2,4,8,12,16,24,32]),
            random_beta=np.array([[.001,.001,.1,1.,1.,1.,1.],
                [100.,1.,.1,1.,1.,1.,1.],[100.,.001,1.,1.,1.,1.,1.],
                [.001,.001,1.,1.,1.,1.,1.],[.001,.001,1.,1.,1.,1.,1.]]),
            full_beta=np.full(5,.01),
            test_tm_seeds=np.stack([20369821+10000*rep+np.arange(50) for rep in range(5)]))
        return result

    if task != 'narma10':
        raise ValueError(task)
    config = dict(washout=100,train_origins=6000,validation_origins=1200,test_origins=1500,
        base_seed=20260823,validation_limit=512,optical_dim=64,projection_seed=20260864)
    data,raw,drive,state,_ = preprocess_narma10(config,'cpu')
    names = {'train_masks':'x_train','y_train_scaled':'y_train','x_native_lower':'x_lower',
        'x_native_scale':'x_scale','y_mean':'y_mean','y_std':'y_std',
        'projection_matrix':'projection_matrix','projection_lower':'projection_lower',
        'projection_scale':'projection_scale'}
    result = {key:array(data[value]) for key,value in names.items()}
    result.update(drive_full=drive,state_full=state,
        support_sizes=np.array([4,8,12,16,24,32,48,64]),random_beta_m64=np.full(5,.001),
        full_beta=np.array([.001,.001,.001,.01,.01],dtype=np.float32),
        test_tm_seeds=np.stack([20263823+100000*rep+np.arange(50) for rep in range(5)]))
    return result


@torch.no_grad()
def run_regression_task(root,task,*,device='cuda:1',recompute=False,output_dir=None):
    root=Path(root).resolve()
    config=CONFIGS[task]
    base=Path(output_dir) if output_dir else root/'results/SI/sample_efficiency/regression'
    out=base/f'{task}_size{config["size"]}'
    flat_result_directory(out).mkdir(parents=True,exist_ok=True)
    experiment_path=root/f'results/experimental/{task}_experiment.npz'
    summary_path=out/'summary.json'
    if flat_result_path(summary_path).exists() and not recompute:
        result=json.loads(flat_result_path(summary_path).read_text())
        if result['config'] == config:return result
    started=time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    def tensor(value):return torch.as_tensor(value,dtype=torch.float32,device=device)
    archive=regression_inputs(root,task)
    with np.load(flat_result_path(experiment_path),allow_pickle=False) as f:
        column=list(f['support_sizes']).index(config['size'])
        targets=f['coreset_r2'][:,column].astype(float)
        equal_random=f['random_r2'][:,column].astype(float)
    assert targets.shape==(10,) and np.isfinite(targets).all()
    reps=np.tile(np.arange(1,6),2)
    acquisitions=np.repeat(['acquisition1','acquisition2'],5)
    # Fix lambda before the size search using the largest archived random
    # support's validation selection; never tune a parameter on target R2.
    anchor=int(np.max(archive['support_sizes']))
    betas=archive['random_beta'][:,-1] if task=='ccpp' else archive[f'random_beta_m{anchor}']
    if task=='ccpp':
        assert np.all(betas==1.0)
        train_size=archive['x_train_mask_flat_per_rep'].shape[1]
        test_size=archive['x_test_mask_flat_per_rep'].shape[1]
        origins=None
    else:
        assert np.allclose(betas,.001,atol=1e-9)
        train_size=len(archive['train_masks'])
        test_size=200
        origins=_narma_origins(root,archive)
    permutations=np.stack([np.random.default_rng(seed).permutation(train_size) for seed in SEEDS])
    matched=np.full(10,-1,dtype=int)
    matched_scores=np.full(10,np.nan)
    rows,curves=[],[]
    for rep in range(5):
        beta=float(betas[rep])
        tms=tm_ensemble(64,1024,archive['test_tm_seeds'][rep],device)
        if task=='ccpp':
            x=tensor(archive['x_train_mask_flat_per_rep'][rep])
            y=tensor(archive['y_train_standardized_per_rep'][rep])
            query=tensor(archive['x_test_mask_flat_per_rep'][rep])
            query_y=tensor(archive['y_test_standardized_per_rep'][rep])
            query_cache=query_feature_cache(query,tms,optical_intensity_normalization_policy=SUPPORT_ONLY_TRAIN_CALIBRATION)
            raw_truth=archive['y_test_raw_per_rep'][rep].astype(float)
            y_mean=archive['y_mean_per_rep'][rep].astype(float)
            y_std=archive['y_std_per_rep'][rep].astype(float)
            def score_support(xs,ys):
                value=evaluate_regression(xs,ys,query,query_y,tms,beta,query_cache=query_cache,
                    return_predictions=True,optical_intensity_normalization_policy=SUPPORT_ONLY_TRAIN_CALIBRATION)
                prediction=value['prediction_per_tm'].astype(float)*y_std+y_mean
                denominator=np.square(raw_truth-raw_truth.mean()).sum()
                per_tm=1-np.square(prediction-raw_truth[None]).sum(axis=(1,2))/denominator
                return per_tm,None
        else:
            x=quantize_mask(tensor(archive['train_masks']))
            y=tensor(archive['y_train_scaled'])
            data=dict(x_train=x,x_lower=archive['x_native_lower'],x_scale=archive['x_native_scale'],
                y_mean=archive['y_mean'],y_std=archive['y_std'],
                projection_matrix=tensor(archive['projection_matrix']),
                projection_lower=tensor(archive['projection_lower']),
                projection_scale=tensor(archive['projection_scale']))
            drive=archive['drive_full'].astype(float).reshape(-1)
            def score_support(xs,ys):
                model=ReservoirReadoutEnsemble(xs,ys,beta,tms)
                predictions=np.stack([autonomous_rollout(model,drive,origins['state'],int(start),200,data,quantize8=True)
                                      for start in origins['starts']])
                scores=np.asarray([r2_per_tm(p,t) for p,t in zip(predictions,origins['truth'])])
                return scores,predictions
        cache, trajectories, phases, order={},{},{},[]
        def evaluate(n,phase='fine'):
            n=int(n)
            if n not in cache:
                values,predictions=[],[]
                for permutation in permutations:
                    indices=torch.as_tensor(permutation[:n],dtype=torch.long,device=device)
                    value,prediction=score_support(x[indices],y[indices])
                    values.append(value)
                    if prediction is not None:predictions.append(prediction)
                cache[n]=np.stack(values)
                assert np.isfinite(cache[n]).all(),(task,rep,n)
                if predictions:trajectories[n]=np.stack(predictions)
                phases[n]=phase
                order.append(n)
                print(f'{task} rep={rep+1} phase={phase} size={n} random_R2={cache[n].mean():.7f}',flush=True)
                save_json(out/'progress.json',dict(task=task,rep=rep+1,budget=n,
                    random_r2=float(cache[n].mean()),evaluated_candidates=len(cache),
                    completed_matches=rows,elapsed_seconds=time.monotonic()-started))
            return float(cache[n].mean())
        goal_indices=np.flatnonzero(reps==rep+1)
        maximum=float(targets[goal_indices].max())
        evaluate(config['size'],'equal_size')
        upper=train_size
        for n in list(range(config['coarse'],train_size+1,config['coarse']))+[train_size]:
            if evaluate(n,'coarse')>=maximum:
                upper=n
                break
        # Evaluate all fine points to avoid assuming monotonic learning curves.
        fine_grid=list(range(config['fine'],train_size+1,config['fine']))
        if fine_grid[-1]!=train_size:fine_grid.append(train_size)
        for n in fine_grid:
            if n>upper:break
            evaluate(n)
        for index in goal_indices:
            goal=float(targets[index])
            found=next((n for n in fine_grid if n in cache and evaluate(n)>=goal),None)
            if found is not None:
                matched[index]=found
                matched_scores[index]=evaluate(found)
            rows.append(dict(rep=rep+1,acquisition=str(acquisitions[index]),experimental_r2=goal,
                matched_random_size=found,matched_random_r2=evaluate(found) if found is not None else None,
                gain=found/config['size'] if found is not None else None))
        ordered=sorted(cache)
        values=np.stack([cache[n] for n in ordered])
        arrays=dict(candidate_size=np.asarray(ordered),candidate_r2=values,
            candidate_phase=np.asarray([phases[n] for n in ordered]),evaluation_order=np.asarray(order),
            test_tm_seeds=archive['test_tm_seeds'][rep],random_lambda=np.asarray(beta))
        if trajectories:arrays['candidate_predictions_by_draw_origin_tm']=np.stack([trajectories[n] for n in ordered])
        np.savez_compressed(flat_result_path(out/f'rep{rep+1}_curves.npz'),**arrays)
        curves.append(dict(rep=rep+1,candidate_size=ordered,mean_r2=values.reshape(len(ordered),-1).mean(1).tolist(),
                           evaluation_order=order))
        del tms,x,y
        if task=='ccpp':del query_cache,query,query_y
        torch.cuda.empty_cache()
    complete=bool((matched>0).all())
    gains=matched/config['size']
    summary=dict(task=task,evaluation_protocol='independent_tm_feedback' if task=='narma10' else 'per_tm_regression',config=config,device=device,coreset_total=config['size'],train_size=train_size,test_size=test_size,
        metric='R2',experimental_r2_mean=float(targets.mean()),experimental_r2_sd=float(targets.std(ddof=1)),
        experimental_equal_random_r2_mean=float(equal_random.mean()),
        experimental_equal_random_r2_sd=float(equal_random.std(ddof=1)),
        measurement_count=10,independent_coreset_repetitions=5,acquisition_rounds=2,
        random_sampling='uniform nested sampling without replacement from the original training split',
        random_seeds=list(SEEDS),test_tms_per_rep=50,
        optical_normalization='support_only_train_calibration',
        random_beta_by_rep=betas.tolist(),random_beta_anchor_size=anchor,
        beta_policy='fixed before search: validation-selected random lambda at largest archived support budget',
        fine_search='first crossing over all positive even budgets below coarse upper bound; no interpolation',
        narma_origin_indices=origins['starts'].tolist() if origins else None,
        narma_quantization='8-bit support and query masks' if origins else None,
        score_aggregation=('mean raw-scale per-TM R2, then mean of 3 random draws' if task=='ccpp' else
            'Independent feedback per TM; score each 200-step trajectory, then average 50 TMs, 2 origins and 3 random draws'),
        all_matched=complete,matched_count=int((matched>0).sum()),
        matched_random_size=matched.tolist(),gains=[float(g) if m>0 else None for g,m in zip(gains,matched)],
        gain_arithmetic_mean=float(gains.mean()) if complete else None,
        gain_sample_sd=float(gains.std(ddof=1)) if complete else None,
        gain_geometric_mean=float(np.exp(np.log(gains).mean())) if complete else None,
        source_reps=reps.tolist(),acquisitions=acquisitions.tolist(),matches=rows,curves=curves,
        runtime_seconds=time.monotonic()-started)
    np.savez_compressed(flat_result_path(out/'arrays.npz'),experimental_r2=targets,experimental_equal_random_r2=equal_random,
        matched_random_size=matched,matched_random_r2=matched_scores,gain=np.where(matched>0,gains,np.nan),
        source_reps=reps,source_acquisitions=acquisitions,random_permutations=permutations,random_seeds=np.asarray(SEEDS))
    save_json(summary_path,summary)
    return summary


