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
        return cls._models.get(name)