"""Tune candidate ranking using only training-trip cross-validation."""
import json
from importlib.metadata import version
from pathlib import Path
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from train_baselines import ROOT, MODELS, load_split, fit_model, top_candidates, OBS_KEY, run
from score_baseline import score_candidates


def tune(name, trials=6):
    timeline, rows = load_split(ROOT / "data/processed", "train")
    folds = list(GroupKFold(3).split(timeline, groups=timeline.trip_id))
    output = ROOT / "experiments" / (name + "_tuned")
    output.mkdir(parents=True, exist_ok=True)
    original = MODELS[name].copy()
    pd.DataFrame([
        {"trip_id": trip, "validation_fold": i}
        for i, (_, val) in enumerate(folds, 1)
        for trip in sorted(set(timeline.iloc[val].trip_id))
    ]).to_csv(output / "search_folds.csv", index=False)
    (output / "search_protocol.json").write_text(json.dumps({
        "seed": 42, "search_folds": 3, "trials_requested": trials,
        "objective": "mean fold observation top1 on aligned training trips",
        "student_test_used_for_selection": False,
        "optuna_version": version("optuna"),
        "baseline_parameters": original,
        "validation_note": "Five-fold refit evaluation is not nested CV."
    }, indent=2))
    def objective(trial):
        params = original.copy()
        if name == "random_forest":
            params.update(n_estimators=original["n_estimators"],
                max_depth=trial.suggest_int("max_depth", 8, 24, step=4),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 80, log=True),
                max_features=trial.suggest_categorical("max_features", ["sqrt", 1.0]),
                class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]))
        else:
            params.update(n_estimators=trial.suggest_int("n_estimators", 100, 400, step=100),
                max_depth=trial.suggest_int("max_depth", 2, 8),
                learning_rate=trial.suggest_float("learning_rate", .02, .2, log=True),
                min_child_weight=trial.suggest_float("min_child_weight", 5, 100, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1, 100, log=True))
        MODELS[name] = params
        scores=[]
        for fit, val in folds:
            fit_ids=set(timeline.iloc[fit].trip_id); val_ids=set(timeline.iloc[val].trip_id)
            model, _, _ = fit_model(name, rows[rows.trip_id.isin(fit_ids)])
            validation=rows[rows.trip_id.isin(val_ids)]
            predicted=top_candidates(validation, score_candidates(model,validation))
            aligned=timeline.iloc[val].query("alignment_status == 'aligned'")
            joined=aligned.merge(predicted,on=OBS_KEY,how="left",validate="one_to_one")
            scores.append(float(joined.arc_id.eq(joined.reference_arc_id).fillna(False).mean()))
        trial.set_user_attr("fold_top1", scores)
        trial.set_user_attr("parameters", params)
        return float(np.mean(scores))
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=3))
    study.enqueue_trial({"max_depth": original["max_depth"], "min_samples_leaf": original["min_samples_leaf"], "max_features": original["max_features"], "class_weight": None} if name == "random_forest" else {"n_estimators": original["n_estimators"], "max_depth": original["max_depth"], "learning_rate": original["learning_rate"], "min_child_weight": original["min_child_weight"], "reg_lambda": 1.0})
    study.optimize(objective,n_trials=trials,callbacks=[lambda study, trial: study.trials_dataframe().to_csv(output / "search_trials.csv",index=False)])
    study.trials_dataframe().to_csv(output / "search_trials.csv",index=False)
    best=study.best_trial.user_attrs["parameters"]
    (output / "selection.json").write_text(json.dumps({"objective":"mean three-fold training-trip top1", "best_cv_top1":study.best_value,"parameters":best,"trials":trials},indent=2))
    MODELS[name]=best
    run(ROOT / "data/processed", ROOT / "data/raw/Student folder.zip",output,[name])

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=["random_forest","xgboost"],required=True)
    parser.add_argument("--trials",type=int,default=6)
    args=parser.parse_args()
    tune(args.model,args.trials)
