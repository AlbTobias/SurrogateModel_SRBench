# This example submission shows the submission of FEAT (cavalab.org/feat).
try:
    from geml.regressors import GeneticProgrammingRegressor
    from geml.regressors import model
    _LEGACY_IMAGE_API = False
except ImportError:
    # The published SRBench image predates the separate `geml` package and
    # contains the regressor under GeneticEngine itself.
    from geneticengine.off_the_shelf.regressors import GeneticProgrammingRegressor
    _LEGACY_IMAGE_API = True
from sklearn.base import RegressorMixin

"""
est: a sklearn-compatible regressor. 
    if you don't have one they are fairly easy to create. 
    see https://scikit-learn.org/stable/developers/develop.html
"""
est: RegressorMixin = GeneticProgrammingRegressor(
    **(
        {'timer_stop_criteria': True, 'timer_limit': 60*60}
        if _LEGACY_IMAGE_API
        else {'max_time': 60*60}
    )
)


if _LEGACY_IMAGE_API:
    def model(est, X=None):
        return est.sympy_compatible_phenotype


def get_population(est) -> list[RegressorMixin]:
    """
    Return the final population of the model. This final population should
    be a list with at most 100 individuals. Each of the individuals must
    be compatible with scikit-learn, so they should have a predict method.

    Also, it is expected that the `model()` function can operate with them,
    so they should have a way of getting a simpy string representation.

    Returns
    -------
    A list of scikit-learn compatible estimators
    """

    return est.get_population()


def get_best_solution(est) -> RegressorMixin:
    """
    Return the best solution from the final model.

    Returns
    -------
    A scikit-learn compatible estimator
    """

    return est.get_best_solution()


# define eval_kwargs.
eval_kwargs = {}
