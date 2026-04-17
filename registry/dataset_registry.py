class DatasetRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def wrapper(dataset_cls):
            cls._registry[name] = dataset_cls
            return dataset_cls
        return wrapper

    @classmethod
    def get(cls, name):
        if name not in cls._registry:
            raise ValueError(f"Dataset {name} not registered. Available: {list(cls._registry.keys())}")
        return cls._registry[name]