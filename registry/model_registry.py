class ModelRegistry:
    _models = {}

    @classmethod
    def register(cls, name):
        def wrapper(model_cls):
            cls._models[name] = model_cls
            return model_cls
        return wrapper

    @classmethod
    def get(cls, name):
        if name not in cls._models:
            raise ValueError(f"Model {name} not registered. Available: {list(cls._models.keys())}")
        return cls._models[name]