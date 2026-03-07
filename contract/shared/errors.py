class ContractError(Exception):
    pass


class ContractViolation(ContractError):
    pass


class SchemaError(ContractError):
    pass


class OrderingError(ContractError):
    pass

