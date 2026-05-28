from src.models import LinearRegression


def test_linear_regression():
    model = LinearRegression(weight=2, intercept=3)
    data = [1, 2, 3]
    expected = [5, 7, 9]  # Calculated as (value * weight) + intercept
    assert model.predict(data) == expected
    assert model._calculate(4) == 11  # Calculated as (4 * weight) + intercept
    assert model.weight == 2
    assert model.intercept == 3
    assert isinstance(model.predict(data), list)
    assert all(isinstance(x, (int, float)) for x in model.predict(data))
    assert isinstance(model, LinearRegression)
