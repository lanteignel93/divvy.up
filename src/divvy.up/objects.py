class Individual:
    def __init__(self, name: str):
        self.name = name


class Expense:
    def __init__(self, payer: Individual, distribution: str, cost: float):
        self.Individual = Individual
        self.distribution = distribution
        self.cost = cost


class Budget:
    def __init__(self, expenses: list[Expense]):
        self.expenses = expenses
