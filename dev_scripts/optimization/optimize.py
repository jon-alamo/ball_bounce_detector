from bounce_detector.objective_function import objective_function
from scipy.optimize import minimize


def optimize_parameters(df, tolerance, initial_params, bounds, n_calls=50):
    """
    Optimize parameters for the bounce detection pipeline using Bayesian optimization.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing ball position data.
    tolerance (int): Tolerance window for matching detected bounces with ground truth.
    initial_params (dict): Initial parameters for the pipeline.
    bounds (dict): Bounds for each parameter to be optimized.
    n_calls (int): Number of calls to the objective function.

    Returns:
    dict: Best parameters found during optimization.
    float: Best precision score achieved.
    """
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args

    # Define the search space
    dimensions = []
    for param, (low, high) in bounds.items():
        if isinstance(initial_params[param], int):
            dimensions.append(Integer(low, high, name=param))
        else:
            dimensions.append(Real(low, high, name=param))

    @use_named_args(dimensions=dimensions)
    def objective(**params):
        return -objective_function(df, tolerance, **params)

    # Run Bayesian optimization
    result = gp_minimize(objective, dimensions, n_calls=n_calls, random_state=42)

    # Extract best parameters and precision
    best_params = {dim.name: val for dim, val in zip(dimensions, result.x)}
    best_precision = -result.fun

    return best_params, best_precision


def optimize_parameters_2(df, tolerance, initial_params, bounds, n_calls=50):
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args

    dimensions = []
    x0 = []
    
    # Asegurar orden consistente
    param_names = list(bounds.keys())
    
    for param in param_names:
        low, high = bounds[param]
        if isinstance(initial_params[param], int):
            dimensions.append(Integer(low, high, name=param))
        else:
            dimensions.append(Real(low, high, name=param))
        x0.append(initial_params[param])

    @use_named_args(dimensions=dimensions)
    def objective(**params):
        return -objective_function(df, tolerance, **params)

    result = gp_minimize(
        objective, 
        dimensions, 
        n_calls=n_calls,
        n_initial_points=5, # Reduce puntos aleatorios si x0 es fiable
        x0=x0,              # Evalúa tus parámetros iniciales primero
        random_state=42,
        acq_func="EI"       # Puede ayudar a una búsqueda menos errática
    )

    best_params = {dim.name: val for dim, val in zip(dimensions, result.x)}
    best_precision = -result.fun

    return best_params, best_precision


def optimize_parameters_forest(df, tolerance, initial_params, bounds, n_calls=500):
    from skopt import forest_minimize  # Cambio a Random Forest
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args

    param_names = list(bounds.keys())
    dimensions = []
    x0 = []

    for name in param_names:
        low, high = bounds[name]
        if isinstance(initial_params[name], int):
            dimensions.append(Integer(low, high, name=name))
        else:
            dimensions.append(Real(low, high, name=name))
        x0.append(initial_params[name])

    @use_named_args(dimensions=dimensions)
    def objective(**params):
        # El signo negativo se mantiene porque buscamos maximizar la precisión
        return -objective_function(df, tolerance, **params)

    res = forest_minimize(
        objective,
        dimensions,
        n_calls=n_calls,
        n_initial_points=5,
        x0=x0,
        random_state=42,
        base_estimator="RF", # Random Forest
        acq_func="EI"        # Centrado en mejora esperada
    )

    best_params = {dim.name: val for dim, val in zip(dimensions, res.x)}
    return best_params, -res.fun



import numpy as np
from scipy.optimize import differential_evolution

import numpy as np
from scipy.optimize import differential_evolution

def optimize_parameters_de(df, tolerance, initial_params, bounds_dict):
    param_names = list(bounds_dict.keys())
    bounds_list = [bounds_dict[name] for name in param_names]
    
    # 1. Definir integrality (0: continuo, 1: entero)
    integrality = [1 if isinstance(initial_params[name], int) else 0 for name in param_names]

    def objective(x):
        # x ya vendrá respetando la integrality si se configura en el solver
        params = {name: (int(val) if integrality[i] else val) 
                  for i, (name, val) in enumerate(zip(param_names, x))}
        return -objective_function(df, tolerance, **params)

    # 2. Construir población inicial (S > 4)
    # Usamos popsize=15 (defecto), la población total será popsize * len(x)
    n_params = len(param_names)
    pop_size = 15 * n_params
    
    # Inicializar matriz de población con valores aleatorios dentro de bounds
    init_pop = np.random.uniform(
        [b[0] for b in bounds_list], 
        [b[1] for b in bounds_list], 
        (pop_size, n_params)
    )
    
    # Inyectar tus parámetros conocidos en el primer individuo
    x0 = np.array([initial_params[name] for name in param_names])
    init_pop[0] = x0

    # 3. Ejecutar optimización
    result = differential_evolution(
        objective,
        bounds_list,
        integrality=integrality,  # Manejo nativo de enteros
        init=init_pop,            # Población con tu x0 incluido
        strategy='best1bin',
        mutation=(0.5, 1),
        recombination=0.7,
        tol=0.01,
        seed=42
    )

    best_params = {name: (int(val) if integrality[i] else val) 
                   for i, (name, val) in enumerate(zip(param_names, result.x))}
    
    return best_params, -result.fun



def optimize_parameters_local(df, tolerance, initial_params, bounds_dict):
    param_names = list(bounds_dict.keys())
    x0 = np.array([initial_params[name] for name in param_names])
    bounds_list = [bounds_dict[name] for name in param_names]
    
    # Identificar índices de enteros para el redondeo interno
    int_indices = [i for i, name in enumerate(param_names) 
                   if isinstance(initial_params[name], int)]

    def objective(x):
        # Aplicar restricciones de bounds manualmente (Powell a veces sale de ellos)
        x_clipped = np.clip(x, [b[0] for b in bounds_list], [b[1] for b in bounds_list])
        
        params = {}
        for i, name in enumerate(param_names):
            if i in int_indices:
                params[name] = int(np.round(x_clipped[i]))
            else:
                params[name] = x_clipped[i]
        
        score = objective_function(df, tolerance, **params)
        # Log de debug para ver si realmente estamos evaluando x0
        # print(f"Precision: {score:.4f} | Params: {params}")
        return -score

    # El método 'Powell' es robusto para funciones no diferenciables
    res = minimize(
        objective,
        x0,
        method='Powell',
        bounds=bounds_list,
        options={'xtol': 1e-3, 'disp': True}
    )

    best_params = {name: (int(np.round(res.x[i])) if i in int_indices else res.x[i]) 
                   for i, name in enumerate(param_names)}
    
    return best_params, -res.fun


def optimize_parameters_grid(df, tolerance, parameter_ranges):
    from itertools import product
    import time

    param_names = list(parameter_ranges.keys())
    param_values = [parameter_ranges[name] for name in param_names]
    total_combinations = np.prod([len(vals) for vals in param_values])
    print(f"Total combinations to evaluate: {total_combinations}")
    best_precision = -1
    best_params = None
    i = 0
    t0 = time.time()
    for values in product(*param_values):
        params = {name: values[i] for i, name in enumerate(param_names)}
        
        precision = objective_function(df, tolerance, **params)
        i += 1
        avg_iter_time = (time.time() - t0) / i

        if precision > best_precision:
            best_precision = precision
            best_params = params
            print(f"New best precision: {best_precision:.2f}% with params: {best_params}")
            
        if i % 100 == 0 or i == total_combinations:
            # Calculate remaining time
            remaining_time = total_combinations * avg_iter_time - (time.time() - t0)
            print(f"Estimated remaining time: {remaining_time/60:.2f} minutes")
            print(f"Progress: {i}/{total_combinations} ({(i/total_combinations)*100:.2f}%)")



    return best_params, best_precision



import optuna

def optimize_with_optuna_xy(df, tolerance, initial_params, bounds, n_trials=500):
    """
    Optimización avanzada utilizando TPE y gestión nativa de tipos.
    """
    from bounce_detector.pipeline import run_pipeline_xy
    
    def objective(trial):
        # Sugerencia de parámetros respetando tipos y rangos
        params = {
            'xw': trial.suggest_int('xw', bounds['xw'][0], bounds['xw'][1]),
            'yw': trial.suggest_int('yw', bounds['yw'][0], bounds['yw'][1]),
            'xvw': trial.suggest_int('xvw', bounds['xvw'][0], bounds['xvw'][1]),
            'yvw': trial.suggest_int('yvw', bounds['yvw'][0], bounds['yvw'][1]),
            'xaw': trial.suggest_int('xaw', bounds['xaw'][0], bounds['xaw'][1]),
            'yaw': trial.suggest_int('yaw', bounds['yaw'][0], bounds['yaw'][1]),
            'mf': trial.suggest_float('mf', bounds['mf'][0], bounds['mf'][1]),
            'dfx': trial.suggest_float('dfx', bounds['dfx'][0], bounds['dfx'][1]),
            'dfy': trial.suggest_float('dfy', bounds['dfy'][0], bounds['dfy'][1]),
            'xat': trial.suggest_float('xat', bounds['xat'][0], bounds['xat'][1]),
            'yat': trial.suggest_float('yat', bounds['yat'][0], bounds['yat'][1]),
            'sha': trial.suggest_int('sha', bounds['sha'][0], bounds['sha'][1]),
            'mew': trial.suggest_int('mew', bounds['mew'][0], bounds['mew'][1]),
            'sb': trial.suggest_float('sb', bounds['sb'][0], bounds['sb'][1])
        }
        
        # Optuna maximiza o minimiza según se configure el estudio
        return objective_function(df, tolerance, pipeline=run_pipeline_xy, **params)

    # 1. Crear el estudio (dirección maximizar para precisión)
    # El sampler TPESampler es el por defecto y el mejor para este caso
    study = optuna.create_study(direction='maximize')

    # 2. Inyectar parámetros iniciales (el set de 66.17% o 75%)
    # Esto garantiza que la primera iteración sea tu punto de referencia
    study.enqueue_trial(initial_params)

    # 3. Ejecutar optimización
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value



def optimize_with_optuna_ang(df, tolerance, initial_params, bounds, n_trials=500):
    """
    Optimización avanzada utilizando TPE y gestión nativa de tipos.
    """
    from bounce_detector.pipeline import run_pipeline_polar
    
    def objective(trial):
        # Sugerencia de parámetros respetando tipos y rangos
        params = {
            'xw': trial.suggest_int('xw', bounds['xw'][0], bounds['xw'][1]),
            'yw': trial.suggest_int('yw', bounds['yw'][0], bounds['yw'][1]),
            'xvw': trial.suggest_int('xvw', bounds['xvw'][0], bounds['xvw'][1]),
            'yvw': trial.suggest_int('yvw', bounds['yvw'][0], bounds['yvw'][1]),
            'avw': trial.suggest_int('avw', bounds['avw'][0], bounds['avw'][1]),
            'mvw': trial.suggest_int('mvw', bounds['mvw'][0], bounds['mvw'][1]),
            'aaw': trial.suggest_int('aaw', bounds['aaw'][0], bounds['aaw'][1]),
            'maw': trial.suggest_int('maw', bounds['maw'][0], bounds['maw'][1]),
            'mf': trial.suggest_float('mf', bounds['mf'][0], bounds['mf'][1]),
            'dfa': trial.suggest_float('dfa', bounds['dfa'][0], bounds['dfa'][1]),
            'dfm': trial.suggest_float('dfm', bounds['dfm'][0], bounds['dfm'][1]),
            'aat': trial.suggest_float('aat', bounds['aat'][0], bounds['aat'][1]),
            'mat': trial.suggest_float('mat', bounds['mat'][0], bounds['mat'][1]),
            'sha': trial.suggest_int('sha', bounds['sha'][0], bounds['sha'][1]),
            'mew': trial.suggest_int('mew', bounds['mew'][0], bounds['mew'][1]),
            'sb': trial.suggest_float('sb', bounds['sb'][0], bounds['sb'][1])
        }
        
        # Optuna maximiza o minimiza según se configure el estudio
        return objective_function(df, tolerance, pipeline=run_pipeline_polar, **params)

    # 1. Crear el estudio (dirección maximizar para precisión)
    # El sampler TPESampler es el por defecto y el mejor para este caso
    study = optuna.create_study(direction='maximize')

    # 2. Inyectar parámetros iniciales (el set de 66.17% o 75%)
    # Esto garantiza que la primera iteración sea tu punto de referencia
    study.enqueue_trial(initial_params)

    # 3. Ejecutar optimización
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value



def optimize_with_optuna_full(df, tolerance, initial_params, bounds, n_trials=500):
    """
    Optimización avanzada utilizando TPE y gestión nativa de tipos.
    """
    from bounce_detector.pipeline import run_pipeline_full
    
    def objective(trial):
        # Sugerencia de parámetros respetando tipos y rangos
        params = {
            'xw': trial.suggest_int('xw', bounds['xw'][0], bounds['xw'][1]),
            'yw': trial.suggest_int('yw', bounds['yw'][0], bounds['yw'][1]),
            'xvw': trial.suggest_int('xvw', bounds['xvw'][0], bounds['xvw'][1]),
            'yvw': trial.suggest_int('yvw', bounds['yvw'][0], bounds['yvw'][1]),
            'avw': trial.suggest_int('avw', bounds['avw'][0], bounds['avw'][1]),
            'xaw': trial.suggest_int('xaw', bounds['xaw'][0], bounds['xaw'][1]),
            'yaw': trial.suggest_int('yaw', bounds['yaw'][0], bounds['yaw'][1]),
            'mvw': trial.suggest_int('mvw', bounds['mvw'][0], bounds['mvw'][1]),
            'aaw': trial.suggest_int('aaw', bounds['aaw'][0], bounds['aaw'][1]),
            'maw': trial.suggest_int('maw', bounds['maw'][0], bounds['maw'][1]),
            'mf': trial.suggest_float('mf', bounds['mf'][0], bounds['mf'][1]),
            'dfx': trial.suggest_float('dfx', bounds['dfx'][0], bounds['dfx'][1]),
            'dfy': trial.suggest_float('dfy', bounds['dfy'][0], bounds['dfy'][1]),
            'dfa': trial.suggest_float('dfa', bounds['dfa'][0], bounds['dfa'][1]),
            'dfm': trial.suggest_float('dfm', bounds['dfm'][0], bounds['dfm'][1]),
            'xat': trial.suggest_float('xat', bounds['xat'][0], bounds['xat'][1]),
            'yat': trial.suggest_float('yat', bounds['yat'][0], bounds['yat'][1]),
            'aat': trial.suggest_float('aat', bounds['aat'][0], bounds['aat'][1]),
            'mat': trial.suggest_float('mat', bounds['mat'][0], bounds['mat'][1]),
            'sha': trial.suggest_int('sha', bounds['sha'][0], bounds['sha'][1]),
            'mew': trial.suggest_int('mew', bounds['mew'][0], bounds['mew'][1]),
            'sb': trial.suggest_float('sb', bounds['sb'][0], bounds['sb'][1])
        }
        
        # Optuna maximiza o minimiza según se configure el estudio
        return objective_function(df, tolerance, pipeline=run_pipeline_full, **params)

    # 1. Crear el estudio (dirección maximizar para precisión)
    # El sampler TPESampler es el por defecto y el mejor para este caso
    study = optuna.create_study(direction='maximize')

    # 2. Inyectar parámetros iniciales (el set de 66.17% o 75%)
    # Esto garantiza que la primera iteración sea tu punto de referencia
    study.enqueue_trial(initial_params)

    # 3. Ejecutar optimización
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value