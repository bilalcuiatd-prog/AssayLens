from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def build_model(task, params):
    seed = params.get('random_state', 42)
    if task == 'classification':
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)
